#!/usr/bin/env python3
"""Diagnose SID_a/SID_b/SID_c LWC gradient conflicts at the final layer.

The diagnostic reconstructs the controlled AD experiment without changing any
checkpoint.  It uses the original FP model for teacher targets, restores the
shared quantized prefix for student inputs, and decomposes the final-layer
objective into one gradient contribution per SID slot.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate_lfq_boundary_diagnostics import _checkpoint_config
from .omniquant.runtime import (
    LFQ_SLOT_NAMES,
    OmniQuantConfig,
    _build_lfq_boundary_targets,
    _first_tensor,
    _LFQOutputProjector,
    _lfq_boundary_loss,
    _lfq_soft_cross_entropy_from_logits,
    _load_omniquant_checkpoint,
    _restore_train_block_parameters,
    _TrainableOmniBlock,
    _TrainableSymmetricLinear,
    _tree_cpu,
    restore_omniquant_layers_from_checkpoints,
)
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    build_lfq_sid_slot_batches,
    capture_layer_input_batches,
    default_calib_split,
    dtype_from_name,
    get_task_config,
    get_transformer_layers,
    load_task_data,
    resolve_repo_path,
    set_seed,
    sid_slot_token_ids,
)
from .support.runtime_utils import _move_tree_to_device
from .support.smoothquant_runtime import Batch, _batch_to_args_kwargs


PAIR_NAMES = (("a", "b"), ("a", "c"), ("b", "c"))
STATE_NAMES = ("initial", "trained")
OBJECTIVE_NAMES = ("ce", "boundary", "composite")
DEFAULT_EXPERIMENT_ROOT = Path(
    "artifacts/results/fake_quant/recommender/"
    "1p7b_ad_fp4w_fp8a_lwc_abc_boundary_prefix128_final512_heldout512"
)
DEFAULT_PREFIX_CHECKPOINT_DIR = (
    DEFAULT_EXPERIMENT_ROOT
    / "shared_mse_lwc_prefix_calib128/1.7B/ad/omniquant_calibration"
)
DEFAULT_BOUNDARY_CHECKPOINT_DIR = (
    DEFAULT_EXPERIMENT_ROOT
    / "abc_lfq_boundary_w0.3_train512_heldout512/1.7B/ad/omniquant_calibration"
)
DEFAULT_OUTPUT_PATH = DEFAULT_EXPERIMENT_ROOT / "abc_gradient_conflicts_train512.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pairwise SID_a/SID_b/SID_c final-layer LWC gradient "
            "conflicts for CE, boundary, and their trained composite objective."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=("ad", "product", "video"))
    parser.add_argument("--calib_split", default="auto")
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument(
        "--prefix_checkpoint_dir",
        default=str(DEFAULT_PREFIX_CHECKPOINT_DIR),
    )
    parser.add_argument(
        "--boundary_checkpoint_dir",
        default=str(DEFAULT_BOUNDARY_CHECKPOINT_DIR),
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=STATE_NAMES,
        default=list(STATE_NAMES),
        help="Parameter states to inspect. Initial is before final-layer LWC training.",
    )
    parser.add_argument("--expected_boundary_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("bfloat16", "float16"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("--sample_size must be positive.")
    if args.sample_offset < 0:
        raise ValueError("--sample_offset must be non-negative.")
    if not math.isfinite(args.expected_boundary_weight) or args.expected_boundary_weight <= 0.0:
        raise ValueError("--expected_boundary_weight must be finite and positive.")
    if len(set(args.states)) != len(args.states):
        raise ValueError("--states must not contain duplicates.")


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.reshape(-1).double()
    right_flat = right.reshape(-1).double()
    denominator = float(torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat))
    if denominator == 0.0:
        return None
    return float(torch.dot(left_flat, right_flat) / denominator)


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "valid_count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "negative_fraction": None,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "valid_count": len(values),
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "p25": float(torch.quantile(tensor, 0.25)),
        "p75": float(torch.quantile(tensor, 0.75)),
        "negative_fraction": float((tensor < 0.0).double().mean()),
    }


class GradientConflictAccumulator:
    """Streaming statistics for one objective's per-slot gradients."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.loss_sums = {slot: 0.0 for slot in LFQ_SLOT_NAMES}
        self.grad_norm_sums = {slot: 0.0 for slot in LFQ_SLOT_NAMES}
        self.grad_sums: dict[str, torch.Tensor | None] = {
            slot: None for slot in LFQ_SLOT_NAMES
        }
        self.sample_cosines = {
            f"{left}_{right}": [] for left, right in PAIR_NAMES
        }
        self.zero_norm_counts = {
            f"{left}_{right}": 0 for left, right in PAIR_NAMES
        }

    def update(
        self,
        *,
        losses: Mapping[str, float],
        gradients: Mapping[str, torch.Tensor],
    ) -> None:
        if set(losses) != set(LFQ_SLOT_NAMES) or set(gradients) != set(LFQ_SLOT_NAMES):
            raise ValueError("Losses and gradients must contain exactly slots a, b, and c.")
        self.sample_count += 1
        for slot in LFQ_SLOT_NAMES:
            gradient = gradients[slot].detach().reshape(-1)
            gradient_cpu = gradient.cpu().double()
            self.loss_sums[slot] += float(losses[slot])
            self.grad_norm_sums[slot] += float(torch.linalg.vector_norm(gradient_cpu))
            if self.grad_sums[slot] is None:
                self.grad_sums[slot] = torch.zeros_like(gradient_cpu)
            assert self.grad_sums[slot] is not None
            self.grad_sums[slot].add_(gradient_cpu)

        for left, right in PAIR_NAMES:
            name = f"{left}_{right}"
            cosine = _safe_cosine(gradients[left], gradients[right])
            if cosine is None:
                self.zero_norm_counts[name] += 1
            else:
                self.sample_cosines[name].append(cosine)

    def mean_gradient(self, slot: str) -> torch.Tensor:
        if self.sample_count == 0 or self.grad_sums[slot] is None:
            raise RuntimeError("No gradients have been accumulated.")
        return self.grad_sums[slot] / self.sample_count

    def finalize(self) -> dict[str, Any]:
        if self.sample_count == 0:
            raise RuntimeError("No gradients have been accumulated.")
        mean_gradients = {
            slot: self.mean_gradient(slot) for slot in LFQ_SLOT_NAMES
        }
        total_gradient = sum(mean_gradients.values(), torch.zeros_like(mean_gradients["a"]))
        pairwise: dict[str, Any] = {}
        for left, right in PAIR_NAMES:
            name = f"{left}_{right}"
            left_gradient = mean_gradients[left]
            right_gradient = mean_gradients[right]
            dot = float(torch.dot(left_gradient, right_gradient))
            pairwise[name] = {
                "mean_gradient_cosine": _safe_cosine(left_gradient, right_gradient),
                "mean_gradient_dot": dot,
                "mean_gradient_conflict": dot < 0.0,
                "per_sample_cosine": {
                    **_distribution_summary(self.sample_cosines[name]),
                    "zero_norm_count": self.zero_norm_counts[name],
                },
            }

        total_alignment: dict[str, Any] = {}
        for slot in LFQ_SLOT_NAMES:
            gradient = mean_gradients[slot]
            dot = float(torch.dot(gradient, total_gradient))
            total_alignment[slot] = {
                "cosine_with_total_mean_gradient": _safe_cosine(gradient, total_gradient),
                "dot_with_total_mean_gradient": dot,
                "first_order_descent_compatible": dot > 0.0,
            }

        return {
            "sample_count": self.sample_count,
            "slot_statistics": {
                slot: {
                    "mean_weighted_loss_contribution": (
                        self.loss_sums[slot] / self.sample_count
                    ),
                    "mean_sample_gradient_norm": (
                        self.grad_norm_sums[slot] / self.sample_count
                    ),
                    "mean_gradient_norm": float(
                        torch.linalg.vector_norm(mean_gradients[slot])
                    ),
                }
                for slot in LFQ_SLOT_NAMES
            },
            "pairwise": pairwise,
            "total_mean_gradient_norm": float(torch.linalg.vector_norm(total_gradient)),
            "slot_alignment_with_total": total_alignment,
        }


def summarize_component_alignment(
    ce_accumulator: GradientConflictAccumulator,
    boundary_accumulator: GradientConflictAccumulator,
    sample_cosines: Mapping[str, Sequence[float]],
    zero_norm_counts: Mapping[str, int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for slot in LFQ_SLOT_NAMES:
        ce_gradient = ce_accumulator.mean_gradient(slot)
        boundary_gradient = boundary_accumulator.mean_gradient(slot)
        dot = float(torch.dot(ce_gradient, boundary_gradient))
        output[slot] = {
            "mean_gradient_cosine": _safe_cosine(ce_gradient, boundary_gradient),
            "mean_gradient_dot": dot,
            "mean_gradient_conflict": dot < 0.0,
            "per_sample_cosine": {
                **_distribution_summary(sample_cosines[slot]),
                "zero_norm_count": int(zero_norm_counts[slot]),
            },
        }
    return output


def _flatten_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flattened = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(flattened).float()


def _load_model(model_path: str, *, dtype: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(dtype),
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model


def _release_module(module: torch.nn.Module) -> None:
    module.to(torch.device("cpu"))
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_checkpoint_protocol(
    *,
    prefix_checkpoint_dir: Path,
    boundary_checkpoint_dir: Path,
    final_layer_idx: int,
    expected_boundary_weight: float,
    task: str,
    seed: int,
    sample_offset: int,
    sample_size: int,
) -> tuple[OmniQuantConfig, OmniQuantConfig]:
    for layer_idx in range(final_layer_idx):
        checkpoint = prefix_checkpoint_dir / f"layer_{layer_idx:02d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing shared-prefix checkpoint: {checkpoint}")
    final_checkpoint = boundary_checkpoint_dir / f"layer_{final_layer_idx:02d}.pt"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained final-layer checkpoint: {final_checkpoint}")

    prefix_config = _checkpoint_config(
        prefix_checkpoint_dir,
        final_layer_idx=final_layer_idx - 1,
    )
    boundary_config = _checkpoint_config(
        boundary_checkpoint_dir,
        final_layer_idx=final_layer_idx,
    )
    if prefix_config.final_objective != "mse":
        raise ValueError("The shared prefix must use the MSE objective.")
    if boundary_config.final_objective != "lfq_ce":
        raise ValueError("The final-layer checkpoint must use the LFQ CE objective.")
    if not boundary_config.use_lwc or boundary_config.use_let or boundary_config.learn_let:
        raise ValueError("This diagnostic requires the LWC-only final-layer experiment.")
    if boundary_config.lfq_loss_weight <= 0.0:
        raise ValueError("The selected checkpoint is boundary-only, not CE+boundary.")
    if not math.isclose(
        boundary_config.lfq_boundary_loss_weight,
        expected_boundary_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Boundary checkpoint weight mismatch: "
            f"{boundary_config.lfq_boundary_loss_weight} != {expected_boundary_weight}."
        )
    shared_fields = (
        "weight_quant_format",
        "activation_quant_format",
        "weight_quant_scheme",
        "weight_group_size",
        "use_lwc",
        "use_let",
        "learn_let",
    )
    for field in shared_fields:
        if getattr(prefix_config, field) != getattr(boundary_config, field):
            raise ValueError(f"Prefix/final checkpoint mismatch for {field}.")

    run_config_path = boundary_checkpoint_dir.parent / "omniquant_config.json"
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        recorded_prefix = run_config.get("omni_prefix_checkpoint_dir")
        if recorded_prefix and Path(str(recorded_prefix)).resolve() != prefix_checkpoint_dir.resolve():
            raise ValueError(
                "The final-layer run records a different shared prefix: "
                f"{recorded_prefix}."
            )
        if run_config.get("task") != task:
            raise ValueError(
                f"Checkpoint task={run_config.get('task')!r}, requested task={task!r}."
            )
        if int(run_config.get("seed", -1)) != seed:
            raise ValueError(
                f"Checkpoint seed={run_config.get('seed')!r}, requested seed={seed}."
            )
        train_sample_size = int(run_config.get("omni_train_sample_size", 0))
        if train_sample_size <= 0 or sample_offset + sample_size > train_sample_size:
            raise ValueError(
                f"Requested samples [{sample_offset},{sample_offset + sample_size}) "
                f"are not contained in the recorded training range [0,{train_sample_size})."
            )
        recorded_init_logit = float(
            run_config.get("omni_init_lwc_logit", boundary_config.init_lwc_logit)
        )
        if not math.isclose(
            recorded_init_logit,
            boundary_config.init_lwc_logit,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "The recorded initial LWC logit cannot be reconstructed by this "
                f"checkpoint config: {recorded_init_logit} != "
                f"{boundary_config.init_lwc_logit}."
            )
    return prefix_config, boundary_config


def _capture_final_layer_streams(
    *,
    model: torch.nn.Module,
    model_batches: Sequence[Mapping[str, torch.Tensor]],
    token_ids: Mapping[str, Sequence[int]],
    prefix_checkpoint_dir: Path,
    prefix_config: OmniQuantConfig,
) -> tuple[list[Batch], list[Batch], torch.nn.Module, _LFQOutputProjector, int]:
    layers = get_transformer_layers(model)
    final_layer_idx = len(layers) - 1
    final_block_template = copy.deepcopy(layers[final_layer_idx]).to(torch.device("cpu"))
    backbone = getattr(model, "model", None)
    final_norm = getattr(backbone, "norm", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if not isinstance(final_norm, torch.nn.Module) or not isinstance(output_head, torch.nn.Module):
        raise TypeError("LFQ gradient diagnostics require the final norm and output head.")
    projector = _LFQOutputProjector(
        final_norm=final_norm,
        output_head=output_head,
        token_ids=token_ids,
    )
    projector = projector.to(next(model.parameters()).device)

    print(f"[gradient diagnostic] capturing FP inputs to layer {final_layer_idx}")
    fp_inputs = [
        _tree_cpu(batch)
        for batch in capture_layer_input_batches(
            model=model,
            layer=layers[final_layer_idx],
            model_batches=tqdm(
                model_batches,
                desc="FP final-layer inputs",
                unit="sample",
            ),
        )
    ]
    print(f"[gradient diagnostic] restoring shared prefix layers 0-{final_layer_idx - 1}")
    restore_omniquant_layers_from_checkpoints(
        model=model,
        layer_indices=list(range(final_layer_idx)),
        config=prefix_config,
        checkpoint_dir=prefix_checkpoint_dir,
        act_quant_mode="shared_input",
    )
    layers = get_transformer_layers(model)
    print(f"[gradient diagnostic] capturing quantized-prefix inputs to layer {final_layer_idx}")
    quant_inputs = [
        _tree_cpu(batch)
        for batch in capture_layer_input_batches(
            model=model,
            layer=layers[final_layer_idx],
            model_batches=tqdm(
                model_batches,
                desc="Quant-prefix final-layer inputs",
                unit="sample",
            ),
        )
    ]
    return fp_inputs, quant_inputs, final_block_template, projector, final_layer_idx


def _cache_teacher_targets(
    *,
    teacher_block: torch.nn.Module,
    projector: _LFQOutputProjector,
    fp_inputs: Sequence[Batch],
    config: OmniQuantConfig,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    teacher_block.eval()
    probabilities: list[dict[str, torch.Tensor]] = []
    boundary_targets: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(fp_inputs, desc="Teacher targets", unit="sample"):
            args, kwargs = _batch_to_args_kwargs(batch)
            args = _move_tree_to_device(args, device)
            kwargs = _move_tree_to_device(kwargs, device)
            hidden = _first_tensor(teacher_block(*args, **kwargs))
            teacher_logits = {
                slot: logits.float() for slot, logits in projector(hidden).items()
            }
            probabilities.append(
                {
                    slot: F.softmax(teacher_logits[slot], dim=-1).detach().cpu()
                    for slot in LFQ_SLOT_NAMES
                }
            )
            boundary_targets.append(
                _build_lfq_boundary_targets(
                    teacher_logits,
                    topk=config.lfq_boundary_topk,
                    negative_count=config.lfq_boundary_negative_count,
                    tie_threshold=config.lfq_boundary_tie_threshold,
                    gap_scale=config.lfq_boundary_gap_scale,
                )
            )
    return probabilities, boundary_targets


def _trainable_lwc_parameters(
    train_block: _TrainableOmniBlock,
) -> tuple[list[str], list[torch.nn.Parameter]]:
    wrappers = [
        module
        for module in train_block.modules()
        if isinstance(module, _TrainableSymmetricLinear)
    ]
    expected = [parameter for wrapper in wrappers for parameter in wrapper.lwc_parameters()]
    named = [
        (name, parameter)
        for name, parameter in train_block.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _name, parameter in named]
    if not parameters or {id(parameter) for parameter in parameters} != {
        id(parameter) for parameter in expected
    }:
        raise ValueError("Expected exactly the final-layer LWC parameters to be trainable.")
    return [name for name, _parameter in named], parameters


def analyze_parameter_state(
    *,
    state_name: str,
    final_block_template: torch.nn.Module,
    projector: _LFQOutputProjector,
    quant_inputs: Sequence[Batch],
    teacher_probabilities: Sequence[Mapping[str, torch.Tensor]],
    boundary_targets: Sequence[Mapping[str, Any]],
    config: OmniQuantConfig,
    boundary_checkpoint_dir: Path,
    final_layer_idx: int,
    device: torch.device,
) -> dict[str, Any]:
    train_block = _TrainableOmniBlock(
        copy.deepcopy(final_block_template).to(device),
        config=config,
        init_scales={},
    )
    checkpoint_path: Path | None = None
    if state_name == "trained":
        checkpoint_path = boundary_checkpoint_dir / f"layer_{final_layer_idx:02d}.pt"
        state = _load_omniquant_checkpoint(checkpoint_path, layer_idx=final_layer_idx)
        _restore_train_block_parameters(
            train_block,
            state,
            checkpoint_path=checkpoint_path,
            layer_idx=final_layer_idx,
        )
    elif state_name != "initial":
        raise ValueError(f"Unsupported state: {state_name}")
    train_block.eval()
    parameter_names, parameters = _trainable_lwc_parameters(train_block)
    normalized_slot_weights = {
        slot: float(weight) / float(sum(config.lfq_slot_weights))
        for slot, weight in zip(LFQ_SLOT_NAMES, config.lfq_slot_weights)
    }
    accumulators = {
        objective: GradientConflictAccumulator() for objective in OBJECTIVE_NAMES
    }
    component_sample_cosines = {slot: [] for slot in LFQ_SLOT_NAMES}
    component_zero_norm_counts = {slot: 0 for slot in LFQ_SLOT_NAMES}

    iterator = zip(quant_inputs, teacher_probabilities, boundary_targets)
    for quant_batch, teacher, boundary in tqdm(
        iterator,
        total=len(quant_inputs),
        desc=f"Gradient conflicts ({state_name})",
        unit="sample",
    ):
        args, kwargs = _batch_to_args_kwargs(quant_batch)
        args = _move_tree_to_device(args, device)
        kwargs = _move_tree_to_device(kwargs, device)
        prediction = _first_tensor(train_block(*args, **kwargs))
        student_logits = {
            slot: logits.float() for slot, logits in projector(prediction).items()
        }
        _ce_total, ce_slot_losses = _lfq_soft_cross_entropy_from_logits(
            student_logits,
            teacher,
            normalized_slot_weights,
        )
        _boundary_total, boundary_slot_losses = _lfq_boundary_loss(
            student_logits,
            boundary,
            normalized_slot_weights,
        )
        weighted_losses: dict[str, dict[str, torch.Tensor]] = {
            "ce": {},
            "boundary": {},
        }
        for slot in LFQ_SLOT_NAMES:
            slot_weight = normalized_slot_weights[slot]
            weighted_losses["ce"][slot] = (
                slot_weight * config.lfq_loss_weight * ce_slot_losses[slot]
            )
            weighted_losses["boundary"][slot] = (
                slot_weight
                * config.lfq_boundary_loss_weight
                * boundary_slot_losses[slot]
            )

        gradient_calls = [
            (objective, slot, weighted_losses[objective][slot])
            for objective in ("ce", "boundary")
            for slot in LFQ_SLOT_NAMES
        ]
        gradients: dict[str, dict[str, torch.Tensor]] = {
            "ce": {},
            "boundary": {},
        }
        for call_idx, (objective, slot, loss) in enumerate(gradient_calls):
            gradients[objective][slot] = _flatten_gradients(
                loss,
                parameters,
                retain_graph=call_idx < len(gradient_calls) - 1,
            )

        composite_gradients = {
            slot: gradients["ce"][slot] + gradients["boundary"][slot]
            for slot in LFQ_SLOT_NAMES
        }
        composite_losses = {
            slot: float(
                weighted_losses["ce"][slot].detach()
                + weighted_losses["boundary"][slot].detach()
            )
            for slot in LFQ_SLOT_NAMES
        }
        accumulators["ce"].update(
            losses={
                slot: float(weighted_losses["ce"][slot].detach())
                for slot in LFQ_SLOT_NAMES
            },
            gradients=gradients["ce"],
        )
        accumulators["boundary"].update(
            losses={
                slot: float(weighted_losses["boundary"][slot].detach())
                for slot in LFQ_SLOT_NAMES
            },
            gradients=gradients["boundary"],
        )
        accumulators["composite"].update(
            losses=composite_losses,
            gradients=composite_gradients,
        )
        for slot in LFQ_SLOT_NAMES:
            cosine = _safe_cosine(
                gradients["ce"][slot],
                gradients["boundary"][slot],
            )
            if cosine is None:
                component_zero_norm_counts[slot] += 1
            else:
                component_sample_cosines[slot].append(cosine)

    result = {
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_tensors": len(parameters),
        "trainable_parameter_names": parameter_names,
        "objectives": {
            objective: accumulators[objective].finalize()
            for objective in OBJECTIVE_NAMES
        },
        "ce_boundary_alignment_within_slot": summarize_component_alignment(
            accumulators["ce"],
            accumulators["boundary"],
            component_sample_cosines,
            component_zero_norm_counts,
        ),
    }
    _release_module(train_block)
    return result


def _format_number(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):+.4f}"


def print_summary(states: Mapping[str, Mapping[str, Any]]) -> None:
    for state_name, state in states.items():
        print(f"[gradient summary] state={state_name}")
        for objective in OBJECTIVE_NAMES:
            result = state["objectives"][objective]
            pair_text = []
            for left, right in PAIR_NAMES:
                pair = result["pairwise"][f"{left}_{right}"]
                pair_text.append(
                    f"{left.upper()}{right.upper()}:mean_cos="
                    f"{_format_number(pair['mean_gradient_cosine'])},"
                    f"sample_neg={_format_number(pair['per_sample_cosine']['negative_fraction'])}"
                )
            print(f"  objective={objective} " + " ".join(pair_text))
            alignment = result["slot_alignment_with_total"]
            print(
                "    total_alignment "
                + " ".join(
                    f"{slot.upper()}:cos={_format_number(alignment[slot]['cosine_with_total_mean_gradient'])},"
                    f"compatible={alignment[slot]['first_order_descent_compatible']}"
                    for slot in LFQ_SLOT_NAMES
                )
            )
        component = state["ce_boundary_alignment_within_slot"]
        print(
            "  ce_boundary_within_slot "
            + " ".join(
                f"{slot.upper()}:mean_cos={_format_number(component[slot]['mean_gradient_cosine'])},"
                f"sample_neg={_format_number(component[slot]['per_sample_cosine']['negative_fraction'])}"
                for slot in LFQ_SLOT_NAMES
            )
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This full-model diagnostic requires an available CUDA device.")

    output_path = resolve_repo_path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
    model_path = str(resolve_repo_path(args.model_path))
    data_dir = str(resolve_repo_path(args.data_dir))
    prefix_checkpoint_dir = resolve_repo_path(args.prefix_checkpoint_dir)
    boundary_checkpoint_dir = resolve_repo_path(args.boundary_checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    task_config = get_task_config(args.task)
    prompt_token = str(task_config.get("generation_config", {}).get("prompt_token", ""))
    calib_split = (
        default_calib_split(data_dir, DEFAULT_SPLIT, task_name=args.task)
        if args.calib_split == "auto"
        else args.calib_split
    )
    calibration_data = load_task_data(
        task_name=args.task,
        tokenizer=tokenizer,
        data_dir=data_dir,
        split=calib_split,
        sample_size=args.sample_size,
        sample_offset=args.sample_offset,
        require_answer=False,
        drop_final_assistant=True,
    )
    items = list(calibration_data.items())
    if len(items) != args.sample_size:
        raise ValueError(f"Loaded {len(items)} samples, expected {args.sample_size}.")
    sample_ids = [sample_id for sample_id, _sample in items]
    samples = [sample for _sample_id, sample in items]
    model_batches = build_lfq_sid_slot_batches(
        tokenizer=tokenizer,
        samples=samples,
        prompt_token=prompt_token,
        device=device,
    )

    print(
        f"[gradient diagnostic] task={args.task} split={calib_split} "
        f"samples=[{args.sample_offset},{args.sample_offset + args.sample_size}) "
        f"device={device}"
    )
    model = _load_model(model_path, dtype=args.dtype, device=device)
    final_layer_idx = len(get_transformer_layers(model)) - 1
    prefix_config, boundary_config = _validate_checkpoint_protocol(
        prefix_checkpoint_dir=prefix_checkpoint_dir,
        boundary_checkpoint_dir=boundary_checkpoint_dir,
        final_layer_idx=final_layer_idx,
        expected_boundary_weight=args.expected_boundary_weight,
        task=args.task,
        seed=args.seed,
        sample_offset=args.sample_offset,
        sample_size=args.sample_size,
    )
    (
        fp_inputs,
        quant_inputs,
        final_block_template,
        projector,
        captured_final_layer_idx,
    ) = _capture_final_layer_streams(
        model=model,
        model_batches=model_batches,
        token_ids={
            slot: sid_slot_token_ids(tokenizer, slot) for slot in LFQ_SLOT_NAMES
        },
        prefix_checkpoint_dir=prefix_checkpoint_dir,
        prefix_config=prefix_config,
    )
    if captured_final_layer_idx != final_layer_idx:
        raise RuntimeError("Final-layer index changed while restoring the prefix.")
    del model_batches
    _release_module(model)

    teacher_block = copy.deepcopy(final_block_template).to(device)
    teacher_probabilities, boundary_targets = _cache_teacher_targets(
        teacher_block=teacher_block,
        projector=projector,
        fp_inputs=fp_inputs,
        config=boundary_config,
        device=device,
    )
    _release_module(teacher_block)
    del fp_inputs

    state_results: dict[str, Any] = {}
    for state_name in args.states:
        state_results[state_name] = analyze_parameter_state(
            state_name=state_name,
            final_block_template=final_block_template,
            projector=projector,
            quant_inputs=quant_inputs,
            teacher_probabilities=teacher_probabilities,
            boundary_targets=boundary_targets,
            config=boundary_config,
            boundary_checkpoint_dir=boundary_checkpoint_dir,
            final_layer_idx=final_layer_idx,
            device=device,
        )

    payload = {
        "protocol": {
            "model_path": model_path,
            "data_dir": data_dir,
            "task": args.task,
            "calib_split": calib_split,
            "sample_offset": args.sample_offset,
            "sample_size": args.sample_size,
            "sample_ids": sample_ids,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": args.device,
            "final_layer_idx": final_layer_idx,
            "prefix_checkpoint_dir": str(prefix_checkpoint_dir),
            "boundary_checkpoint_dir": str(boundary_checkpoint_dir),
            "states": list(args.states),
            "gt_prefix_conditioned": True,
            "gradient_scope": "final_layer_lwc_parameters_only",
            "lfq_loss_weight": boundary_config.lfq_loss_weight,
            "boundary_loss_weight": boundary_config.lfq_boundary_loss_weight,
            "normalized_slot_weights": {
                slot: float(weight) / float(sum(boundary_config.lfq_slot_weights))
                for slot, weight in zip(LFQ_SLOT_NAMES, boundary_config.lfq_slot_weights)
            },
        },
        "definitions": {
            "mean_gradient_conflict": "Dot product of two dataset-mean slot gradients is negative.",
            "per_sample_negative_fraction": "Fraction of samples whose two slot gradients have negative cosine.",
            "first_order_descent_compatible": "A step along the negative total mean gradient locally decreases this slot loss when true.",
            "composite": "The exact weighted CE plus boundary objective stored in the selected checkpoint.",
        },
        "states": state_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print_summary(state_results)
    print(f"[gradient diagnostic] output={output_path}")


if __name__ == "__main__":
    main()
