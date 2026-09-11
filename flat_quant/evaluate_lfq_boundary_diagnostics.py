#!/usr/bin/env python3
"""Evaluate held-out SID-slot diagnostics for matched OmniQuant checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .omniquant import OmniQuantConfig, restore_omniquant_layers_from_checkpoints
from .omniquant.runtime import _load_omniquant_checkpoint
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    SID_SLOT_NAMES,
    build_lfq_sid_slot_batches,
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


METHOD_NAMES = ("mse", "abc", "boundary")
PRIMARY_METRICS = (
    "cross_entropy",
    "kl",
    "top1_agreement_rate",
    "teacher_top1_in_student_top5_rate",
    "teacher_top1_in_student_top10_rate",
    "teacher_top1_in_student_topk_rate",
    "teacher_top1_student_rank",
    "student_top1_teacher_rank",
    "top5_retention",
    "top10_retention",
    "topk_retention",
    "topk_jaccard",
    "boundary_pair_violation_rate",
    "boundary_pair_weighted_violation_rate",
    "boundary_gap_mae",
    "boundary_gap_weighted_mae",
    "intruder_rank_k1_to_kplusn_rate",
    "intruder_below_rank_kplusn_rate",
    "teacher_cutoff_gap",
    "student_teacher_cutoff_gap",
)
METRIC_DIRECTIONS = {
    "cross_entropy": "lower",
    "kl": "lower",
    "top1_agreement_rate": "higher",
    "teacher_top1_in_student_top5_rate": "higher",
    "teacher_top1_in_student_top10_rate": "higher",
    "teacher_top1_in_student_topk_rate": "higher",
    "teacher_top1_student_rank": "lower",
    "student_top1_teacher_rank": "lower",
    "top5_retention": "higher",
    "top10_retention": "higher",
    "topk_retention": "higher",
    "topk_jaccard": "higher",
    "boundary_pair_violation_rate": "lower",
    "boundary_pair_weighted_violation_rate": "lower",
    "boundary_gap_mae": "lower",
    "boundary_gap_weighted_mae": "lower",
    "intruder_rank_k1_to_kplusn_rate": "lower",
    "intruder_below_rank_kplusn_rate": "lower",
    "teacher_cutoff_gap": "diagnostic",
    "student_teacher_cutoff_gap": "higher",
}

HEAD_METRIC_DEFINITIONS = {
    "top1_agreement_rate": "Teacher and student argmax token are identical.",
    "teacher_top1_in_student_top5_rate": "Teacher argmax is in the student top-5.",
    "teacher_top1_in_student_top10_rate": "Teacher argmax is in the student top-10.",
    "teacher_top1_in_student_topk_rate": (
        "Teacher argmax is in the student top-k configured by --topk."
    ),
    "teacher_top1_student_rank": (
        "One-based competition rank of the teacher argmax under student logits."
    ),
    "student_top1_teacher_rank": (
        "One-based competition rank of the student argmax under teacher logits."
    ),
    "top5_retention": "Teacher/student top-5 set intersection divided by 5.",
    "top10_retention": "Teacher/student top-10 set intersection divided by 10.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MSE-LWC, ABC-LFQ, and ABC-LFQ+boundary checkpoints on a "
            "fixed held-out calibration tail using GT-prefix SID-slot logits."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=("ad", "product", "video"))
    parser.add_argument("--calib_split", default="auto")
    parser.add_argument("--calib_sample_size", type=int, default=1024)
    parser.add_argument("--prefix_calib_sample_size", type=int, default=128)
    parser.add_argument("--train_sample_size", type=int, default=512)
    parser.add_argument("--heldout_sample_size", type=int, default=512)
    parser.add_argument("--mse_checkpoint_dir")
    parser.add_argument("--abc_checkpoint_dir")
    parser.add_argument("--boundary_checkpoint_dir")
    parser.add_argument(
        "--single_checkpoint_dir",
        help=(
            "Evaluate one LFQ checkpoint instead of the controlled three-arm "
            "comparison. Omit the three arm-specific checkpoint arguments."
        ),
    )
    parser.add_argument(
        "--single_method_name",
        default="boundary",
        help="JSON method key and progress label for --single_checkpoint_dir.",
    )
    parser.add_argument(
        "--expected_boundary_lfq_loss_weight",
        type=float,
        default=1.0,
        help="Expected ABC soft-CE weight stored in the boundary-arm checkpoint.",
    )
    parser.add_argument(
        "--expected_boundary_loss_weight",
        type=float,
        default=None,
        help="Optional exact boundary-loss weight required in single-checkpoint mode.",
    )
    parser.add_argument(
        "--expected_omni_let_mode",
        default="none",
        choices=("none", "fixed", "learned"),
        help="Expected LET mode shared by the prefix and all three final-layer arms.",
    )
    parser.add_argument(
        "--expected_omni_let_init",
        default="smoothquant",
        choices=("smoothquant", "ones"),
        help="Expected LET initialization shared by the controlled runs.",
    )
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--negative_count", type=int, default=32)
    parser.add_argument("--tie_threshold", type=float, default=1e-2)
    parser.add_argument("--gap_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("bfloat16", "float16"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    arm_checkpoint_dirs = (
        args.mse_checkpoint_dir,
        args.abc_checkpoint_dir,
        args.boundary_checkpoint_dir,
    )
    if args.single_checkpoint_dir is not None:
        if any(path is not None for path in arm_checkpoint_dirs):
            raise ValueError(
                "--single_checkpoint_dir cannot be combined with the three "
                "arm-specific checkpoint arguments."
            )
        if not args.single_method_name.strip():
            raise ValueError("--single_method_name cannot be empty.")
    elif not all(path is not None for path in arm_checkpoint_dirs):
        raise ValueError(
            "Provide --single_checkpoint_dir, or provide all three "
            "arm-specific checkpoint arguments."
        )
    if args.calib_sample_size <= 0:
        raise ValueError("--calib_sample_size must be positive.")
    if args.prefix_calib_sample_size <= 0:
        raise ValueError("--prefix_calib_sample_size must be positive.")
    if args.train_sample_size <= 0 or args.heldout_sample_size <= 0:
        raise ValueError("--train_sample_size and --heldout_sample_size must be positive.")
    if args.prefix_calib_sample_size > args.train_sample_size:
        raise ValueError("--prefix_calib_sample_size cannot exceed --train_sample_size.")
    if args.train_sample_size + args.heldout_sample_size != args.calib_sample_size:
        raise ValueError(
            "The controlled split requires train_sample_size + heldout_sample_size "
            "to equal calib_sample_size."
        )
    if (
        not math.isfinite(args.expected_boundary_lfq_loss_weight)
        or args.expected_boundary_lfq_loss_weight < 0.0
    ):
        raise ValueError("--expected_boundary_lfq_loss_weight must be finite and non-negative.")
    if args.expected_boundary_loss_weight is not None and (
        not math.isfinite(args.expected_boundary_loss_weight)
        or args.expected_boundary_loss_weight <= 0.0
    ):
        raise ValueError(
            "--expected_boundary_loss_weight must be finite and positive."
        )
    if args.topk <= 0 or args.negative_count <= 0:
        raise ValueError("--topk and --negative_count must be positive.")
    if not math.isfinite(args.tie_threshold) or args.tie_threshold < 0.0:
        raise ValueError("--tie_threshold must be finite and non-negative.")
    if not math.isfinite(args.gap_scale) or args.gap_scale <= 0.0:
        raise ValueError("--gap_scale must be finite and positive.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive.")


def compute_slot_diagnostics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    topk: int,
    negative_count: int,
    tie_threshold: float,
    gap_scale: float,
) -> dict[str, float]:
    """Compute distribution and teacher-boundary metrics for one SID slot."""

    teacher = teacher_logits.detach().float().flatten()
    student = student_logits.detach().float().flatten()
    if teacher.shape != student.shape:
        raise ValueError(
            f"Teacher/student shapes differ: {tuple(teacher.shape)} vs {tuple(student.shape)}."
        )
    candidate_count = topk + negative_count
    if teacher.numel() < candidate_count:
        raise ValueError(
            f"Need at least {candidate_count} slot logits, got {teacher.numel()}."
        )

    teacher_log_prob = torch.log_softmax(teacher, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = torch.log_softmax(student, dim=-1)
    cross_entropy = -(teacher_prob * student_log_prob).sum()
    teacher_entropy = -(teacher_prob * teacher_log_prob).sum()
    kl = cross_entropy - teacher_entropy

    teacher_top1 = int(torch.argmax(teacher).item())
    student_top1 = int(torch.argmax(student).item())
    teacher_top1_student_rank = 1 + int(
        (student > student[teacher_top1]).sum().item()
    )
    student_top1_teacher_rank = 1 + int(
        (teacher > teacher[student_top1]).sum().item()
    )

    def top_set(k: int, logits: torch.Tensor) -> torch.Tensor:
        return torch.topk(
            logits,
            k=min(k, logits.numel()),
            dim=-1,
            sorted=False,
        ).indices

    def set_retention(k: int) -> float:
        teacher_head = top_set(k, teacher)
        student_head = top_set(k, student)
        retained_head = student_head.unsqueeze(-1).eq(
            teacher_head.unsqueeze(0)
        ).any(dim=-1).sum()
        return float(retained_head) / float(teacher_head.numel())

    student_top5 = top_set(5, student)
    student_top10 = top_set(10, student)

    teacher_ranked = torch.topk(
        teacher,
        k=candidate_count,
        dim=-1,
        sorted=True,
    )
    teacher_positive = teacher_ranked.indices[:topk]
    teacher_negative = teacher_ranked.indices[topk:]
    teacher_positive_scores = teacher_ranked.values[:topk]
    teacher_negative_scores = teacher_ranked.values[topk:]

    student_topk = torch.topk(student, k=topk, dim=-1, sorted=True).indices
    in_teacher_positive = student_topk.unsqueeze(-1).eq(
        teacher_positive.unsqueeze(0)
    ).any(dim=-1)
    in_teacher_negative = student_topk.unsqueeze(-1).eq(
        teacher_negative.unsqueeze(0)
    ).any(dim=-1)
    retained = int(in_teacher_positive.sum().item())
    intruder_near = int((~in_teacher_positive & in_teacher_negative).sum().item())
    intruder_far = int((~in_teacher_positive & ~in_teacher_negative).sum().item())

    teacher_gaps = (
        teacher_positive_scores.unsqueeze(-1)
        - teacher_negative_scores.unsqueeze(-2)
    )
    student_positive_scores = student.index_select(0, teacher_positive)
    student_negative_scores = student.index_select(0, teacher_negative)
    student_gaps = (
        student_positive_scores.unsqueeze(-1)
        - student_negative_scores.unsqueeze(-2)
    )
    eligible = teacher_gaps > tie_threshold
    pair_weights = torch.where(
        eligible,
        torch.clamp(teacher_gaps / gap_scale, max=1.0),
        torch.zeros_like(teacher_gaps),
    )
    violations = student_gaps <= 0.0
    absolute_gap_error = (student_gaps - teacher_gaps).abs()
    eligible_count = int(eligible.sum().item())
    if eligible_count:
        violation_rate = violations[eligible].float().mean()
        gap_mae = absolute_gap_error[eligible].mean()
    else:
        violation_rate = teacher.new_zeros(())
        gap_mae = teacher.new_zeros(())
    weight_sum = pair_weights.sum()
    if float(weight_sum) > 0.0:
        weighted_violation_rate = (
            pair_weights * violations.float()
        ).sum() / weight_sum
        weighted_gap_mae = (
            pair_weights * absolute_gap_error
        ).sum() / weight_sum
    else:
        weighted_violation_rate = teacher.new_zeros(())
        weighted_gap_mae = teacher.new_zeros(())

    teacher_cutoff_gap = (
        teacher_positive_scores[-1] - teacher_negative_scores[0]
    )
    student_teacher_cutoff_gap = (
        student[teacher_positive[-1]] - student[teacher_negative[0]]
    )
    union_size = 2 * topk - retained

    return {
        "cross_entropy": float(cross_entropy),
        "teacher_entropy": float(teacher_entropy),
        "kl": float(kl),
        "top1_agreement_rate": float(teacher_top1 == student_top1),
        "teacher_top1_in_student_top5_rate": float(
            student_top5.eq(teacher_top1).any()
        ),
        "teacher_top1_in_student_top10_rate": float(
            student_top10.eq(teacher_top1).any()
        ),
        "teacher_top1_in_student_topk_rate": float(
            student_topk.eq(teacher_top1).any()
        ),
        "teacher_top1_student_rank": float(teacher_top1_student_rank),
        "student_top1_teacher_rank": float(student_top1_teacher_rank),
        "top5_retention": set_retention(5),
        "top10_retention": set_retention(10),
        "topk_retention": retained / float(topk),
        "topk_jaccard": retained / float(union_size),
        "intruder_rank_k1_to_kplusn_rate": intruder_near / float(topk),
        "intruder_below_rank_kplusn_rate": intruder_far / float(topk),
        "eligible_boundary_pair_fraction": eligible_count
        / float(topk * negative_count),
        "boundary_pair_violation_rate": float(violation_rate),
        "boundary_pair_weighted_violation_rate": float(
            weighted_violation_rate
        ),
        "boundary_gap_mae": float(gap_mae),
        "boundary_gap_weighted_mae": float(weighted_gap_mae),
        "teacher_cutoff_gap": float(teacher_cutoff_gap),
        "student_teacher_cutoff_gap": float(student_teacher_cutoff_gap),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return float(sum(values) / len(values))


def _macro_metrics(slot_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    metric_names = tuple(next(iter(slot_metrics.values())).keys())
    return {
        metric: _mean([float(slot_metrics[slot][metric]) for slot in SID_SLOT_NAMES])
        for metric in metric_names
    }


def aggregate_method_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot aggregate an empty method result.")
    slot_aggregates: dict[str, dict[str, float]] = {}
    for slot in SID_SLOT_NAMES:
        metric_names = tuple(samples[0]["slots"][slot].keys())
        slot_aggregates[slot] = {
            metric: _mean(
                [float(sample["slots"][slot][metric]) for sample in samples]
            )
            for metric in metric_names
        }
    macro_names = tuple(samples[0]["macro"].keys())
    macro = {
        metric: _mean([float(sample["macro"][metric]) for sample in samples])
        for metric in macro_names
    }
    return {
        "num_samples": len(samples),
        "slots": slot_aggregates,
        "macro": macro,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of no values.")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap_difference(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    """Return candidate-reference paired mean difference and bootstrap CI."""

    if len(reference) != len(candidate) or not reference:
        raise ValueError("Paired bootstrap inputs must have the same positive length.")
    differences = [
        float(candidate_value) - float(reference_value)
        for reference_value, candidate_value in zip(reference, candidate)
    ]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        draws.append(
            _mean([differences[rng.randrange(len(differences))] for _ in differences])
        )
    draws.sort()
    return {
        "mean_difference": _mean(differences),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _checkpoint_config(
    checkpoint_dir: Path,
    *,
    final_layer_idx: int,
) -> OmniQuantConfig:
    state = _load_omniquant_checkpoint(
        checkpoint_dir / f"layer_{final_layer_idx:02d}.pt",
        layer_idx=final_layer_idx,
    )
    saved = state.get("config")
    if not isinstance(saved, Mapping):
        raise TypeError(f"Missing checkpoint config in {checkpoint_dir}.")
    config = OmniQuantConfig(
        weight_quant_format=str(saved["weight_quant_format"]),
        activation_quant_format=str(saved["activation_quant_format"]),
        weight_quant_scheme=str(saved["weight_quant_scheme"]),
        weight_group_size=int(saved.get("weight_group_size", 0)),
        use_lwc=bool(saved["use_lwc"]),
        use_let=bool(saved["use_let"]),
        learn_let=bool(saved["learn_let"]),
        let_init=str(saved.get("let_init", "smoothquant")),
        smoothquant_alpha=float(saved.get("smoothquant_alpha", 0.4)),
        final_objective=str(saved.get("final_objective", "mse")),
        lfq_token_scope=str(saved.get("lfq_token_scope", "sid_slots")),
        lfq_vocab_scope=str(saved.get("lfq_vocab_scope", "s_abc")),
        lfq_slot_weights=tuple(saved.get("lfq_slot_weights", (1.0, 1.0, 1.0))),
        lfq_loss_weight=float(saved.get("lfq_loss_weight", 1.0)),
        lfq_boundary_loss_weight=float(
            saved.get("lfq_boundary_loss_weight", 0.0)
        ),
        lfq_boundary_topk=int(saved.get("lfq_boundary_topk", 32)),
        lfq_boundary_negative_count=int(
            saved.get("lfq_boundary_negative_count", 32)
        ),
        lfq_boundary_tie_threshold=float(
            saved.get("lfq_boundary_tie_threshold", 1e-2)
        ),
        lfq_boundary_gap_scale=float(
            saved.get("lfq_boundary_gap_scale", 1.0)
        ),
    )
    config.validate()
    return config


def verify_single_checkpoint_set(
    checkpoint_dir: Path,
    *,
    layer_count: int,
    expected_lfq_loss_weight: float,
    expected_boundary_loss_weight: float | None,
) -> tuple[OmniQuantConfig, Path]:
    """Validate one complete LFQ arm and recover its shared-prefix directory."""

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )
    prefix_dirs: set[Path] = set()
    for layer_idx in range(layer_count):
        checkpoint_path = checkpoint_dir / f"layer_{layer_idx:02d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if layer_idx == layer_count - 1:
            continue
        state = _load_omniquant_checkpoint(
            checkpoint_path,
            layer_idx=layer_idx,
        )
        source = state.get("source_checkpoint")
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"Layer {layer_idx} does not record a shared-prefix checkpoint."
            )
        source_path = Path(source).resolve()
        if source_path.name != f"layer_{layer_idx:02d}.pt":
            raise ValueError(
                f"Layer {layer_idx} references an unexpected prefix file: "
                f"{source_path}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Shared-prefix checkpoint not found: {source_path}"
            )
        prefix_dirs.add(source_path.parent)
    if len(prefix_dirs) != 1:
        raise ValueError(
            f"The single arm does not reference one shared prefix: {prefix_dirs}"
        )

    config = _checkpoint_config(
        checkpoint_dir,
        final_layer_idx=layer_count - 1,
    )
    if config.final_objective != "lfq_ce":
        raise ValueError(
            f"Single checkpoint final objective is {config.final_objective!r}, "
            "expected 'lfq_ce'."
        )
    if not math.isclose(
        config.lfq_loss_weight,
        expected_lfq_loss_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Single checkpoint LFQ loss weight mismatch: "
            f"{config.lfq_loss_weight} != {expected_lfq_loss_weight}."
        )
    if config.lfq_boundary_loss_weight <= 0.0:
        raise ValueError("Single checkpoint must have a positive boundary loss weight.")
    if expected_boundary_loss_weight is not None and not math.isclose(
        config.lfq_boundary_loss_weight,
        expected_boundary_loss_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Single checkpoint boundary loss weight mismatch: "
            f"{config.lfq_boundary_loss_weight} != "
            f"{expected_boundary_loss_weight}."
        )
    return config, next(iter(prefix_dirs))


def verify_single_run_protocol(
    checkpoint_dir: Path,
    *,
    prefix_checkpoint_dir: Path,
    checkpoint_config: OmniQuantConfig,
    layer_count: int,
    task: str,
    data_dir: str,
    calib_split: str,
    calib_sample_size: int,
    prefix_calib_sample_size: int,
    train_sample_size: int,
    heldout_sample_size: int,
    seed: int,
    topk: int,
    negative_count: int,
    tie_threshold: float,
    gap_scale: float,
    expected_lfq_loss_weight: float,
    expected_boundary_loss_weight: float | None,
    expected_omni_let_mode: str,
    expected_omni_let_init: str,
) -> dict[str, Any]:
    """Verify the saved runner metadata for one held-out LFQ evaluation."""

    config_path = checkpoint_dir.parent / "omniquant_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing single-arm run config: {config_path}")
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(run_config, Mapping):
        raise TypeError(f"Invalid single-arm run config: {config_path}")

    expected = {
        "task": task,
        "calib_split": calib_split,
        "calib_offset": 0,
        "calibration_only": True,
        "dtype": "bfloat16",
        "seed": seed,
        "layers": list(range(layer_count)),
        "weight_quant_format": "fp4_e2m1",
        "activation_quant_format": "fp8_e4m3fn",
        "weight_quant_scheme": "symmetric",
        "act_quant_mode": "shared_input",
        "omni_lwc": True,
        "omni_let_mode": expected_omni_let_mode,
        "omni_let_init": expected_omni_let_init,
        "omni_final_objective": "lfq_ce",
        "omni_train_sample_size": train_sample_size,
        "omni_validation_sample_size": heldout_sample_size,
        "omni_lfq_boundary_topk": topk,
        "omni_lfq_boundary_negative_count": negative_count,
        "omni_lfq_boundary_tie_threshold": tie_threshold,
        "omni_lfq_boundary_gap_scale": gap_scale,
    }
    for field, expected_value in expected.items():
        if run_config.get(field) != expected_value:
            raise ValueError(
                f"Single-arm protocol requires {field}={expected_value!r}, "
                f"got {run_config.get(field)!r}."
            )
    if int(run_config.get("calib_sample_size", -1)) != calib_sample_size:
        raise ValueError("Single-arm calibration sample count does not match diagnostics.")
    if Path(str(run_config.get("data_dir", ""))).resolve() != Path(data_dir).resolve():
        raise ValueError("Single-arm data_dir does not match diagnostics.")
    weight_group_size = run_config.get("weight_group_size")
    if (
        isinstance(weight_group_size, bool)
        or not isinstance(weight_group_size, int)
        or weight_group_size < 0
        or weight_group_size != checkpoint_config.weight_group_size
    ):
        raise ValueError(
            f"Invalid or inconsistent single-arm weight_group_size: {weight_group_size!r}."
        )
    if not math.isclose(
        float(run_config.get("omni_lfq_loss_weight", float("nan"))),
        expected_lfq_loss_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Single-arm LFQ loss weight does not match diagnostics.")
    recorded_boundary_weight = float(
        run_config.get("omni_lfq_boundary_loss_weight", float("nan"))
    )
    if not math.isclose(
        recorded_boundary_weight,
        checkpoint_config.lfq_boundary_loss_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or (
        expected_boundary_loss_weight is not None
        and not math.isclose(
            recorded_boundary_weight,
            expected_boundary_loss_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Single-arm boundary loss weight does not match diagnostics.")

    recorded_prefix = Path(
        str(run_config.get("omni_prefix_checkpoint_dir", ""))
    ).resolve()
    if recorded_prefix != prefix_checkpoint_dir.resolve():
        raise ValueError(
            f"Single arm records a different shared prefix: {recorded_prefix}."
        )
    prefix_config_path = prefix_checkpoint_dir.parent / "omniquant_config.json"
    if not prefix_config_path.is_file():
        raise FileNotFoundError(f"Missing shared-prefix run config: {prefix_config_path}")
    prefix_config = json.loads(prefix_config_path.read_text(encoding="utf-8"))
    if not isinstance(prefix_config, Mapping):
        raise TypeError(f"Invalid shared-prefix run config: {prefix_config_path}")
    shared_fields = (
        "task", "calib_split", "calib_offset", "dtype", "seed",
        "weight_quant_format", "activation_quant_format", "weight_quant_scheme",
        "weight_group_size", "act_quant_mode", "omni_lwc", "omni_let_mode",
        "omni_let_init", "omni_lwc_lr", "omni_let_lr", "omni_weight_decay",
        "omni_init_lwc_logit", "omni_max_grad_norm",
    )
    for field in shared_fields:
        if prefix_config.get(field) != run_config.get(field):
            raise ValueError(
                f"Shared prefix does not match single arm for {field}: "
                f"{prefix_config.get(field)!r} != {run_config.get(field)!r}."
            )
    expected_prefix = {
        "layers": list(range(layer_count - 1)),
        "calib_sample_size": prefix_calib_sample_size,
        "omni_train_sample_size": 0,
        "omni_validation_sample_size": 0,
        "omni_final_objective": "mse",
    }
    for field, expected_value in expected_prefix.items():
        actual = prefix_config.get(field)
        if field == "calib_sample_size":
            actual = int(actual)
        if actual != expected_value:
            raise ValueError(
                f"Shared prefix requires {field}={expected_value!r}, got {actual!r}."
            )
    return {
        "common": {field: run_config.get(field) for field in shared_fields},
        "shared_prefix_checkpoint_dir": str(prefix_checkpoint_dir.resolve()),
        "single_method_split": {
            "calib_sample_size": calib_sample_size,
            "train_sample_size": train_sample_size,
            "heldout_sample_size": heldout_sample_size,
            "final_objective": "lfq_ce",
            "lfq_loss_weight": expected_lfq_loss_weight,
            "boundary_loss_weight": recorded_boundary_weight,
        },
    }


def _assert_same_tree(left: Any, right: Any, *, label: str) -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"Shared-prefix tensor mismatch at {label}.")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(f"Shared-prefix mapping keys differ at {label}.")
        for key in left:
            _assert_same_tree(left[key], right[key], label=f"{label}.{key}")
        return
    if left != right:
        raise ValueError(f"Shared-prefix value mismatch at {label}: {left!r} != {right!r}.")


def verify_checkpoint_sets(
    checkpoint_dirs: Mapping[str, Path],
    *,
    layer_count: int,
) -> dict[str, OmniQuantConfig]:
    for method, checkpoint_dir in checkpoint_dirs.items():
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"{method} checkpoint directory not found: {checkpoint_dir}")
        for layer_idx in range(layer_count):
            path = checkpoint_dir / f"layer_{layer_idx:02d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {method} checkpoint: {path}")

    for layer_idx in range(layer_count - 1):
        states = {
            method: _load_omniquant_checkpoint(
                checkpoint_dir / f"layer_{layer_idx:02d}.pt",
                layer_idx=layer_idx,
            )
            for method, checkpoint_dir in checkpoint_dirs.items()
        }
        source_paths = {
            method: str(state.get("source_checkpoint", ""))
            for method, state in states.items()
        }
        if not all(source_paths.values()) or len(set(source_paths.values())) != 1:
            raise ValueError(
                f"Layer {layer_idx} does not reference one shared prefix checkpoint: "
                f"{source_paths}"
            )
        reference = states[METHOD_NAMES[0]]
        for method in METHOD_NAMES[1:]:
            _assert_same_tree(
                reference["lwc_parameters"],
                states[method]["lwc_parameters"],
                label=f"layer_{layer_idx:02d}.lwc.{method}",
            )
            _assert_same_tree(
                reference["let_log_scales"],
                states[method]["let_log_scales"],
                label=f"layer_{layer_idx:02d}.let.{method}",
            )

    configs = {
        method: _checkpoint_config(
            checkpoint_dir,
            final_layer_idx=layer_count - 1,
        )
        for method, checkpoint_dir in checkpoint_dirs.items()
    }
    expected_objectives = {"mse": "mse", "abc": "lfq_ce", "boundary": "lfq_ce"}
    for method, expected in expected_objectives.items():
        if configs[method].final_objective != expected:
            raise ValueError(
                f"{method} final objective is {configs[method].final_objective!r}, "
                f"expected {expected!r}."
            )
    if configs["abc"].lfq_boundary_loss_weight != 0.0:
        raise ValueError("ABC control must have zero boundary loss weight.")
    if configs["boundary"].lfq_boundary_loss_weight <= 0.0:
        raise ValueError("Boundary arm must have a positive boundary loss weight.")

    common_fields = (
        "weight_quant_format",
        "activation_quant_format",
        "weight_quant_scheme",
        "use_lwc",
        "use_let",
        "learn_let",
        "let_init",
        "smoothquant_alpha",
    )
    for field in common_fields:
        values = {method: getattr(config, field) for method, config in configs.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"Final-layer config mismatch for {field}: {values}")
    return configs


def verify_run_protocols(
    checkpoint_dirs: Mapping[str, Path],
    *,
    layer_count: int,
    task: str,
    data_dir: str,
    calib_split: str,
    calib_sample_size: int,
    prefix_calib_sample_size: int,
    train_sample_size: int,
    heldout_sample_size: int,
    seed: int,
    topk: int,
    negative_count: int,
    tie_threshold: float,
    gap_scale: float,
    expected_boundary_lfq_loss_weight: float = 1.0,
    expected_omni_let_mode: str = "none",
    expected_omni_let_init: str = "smoothquant",
) -> dict[str, Any]:
    """Verify that saved runner metadata implements the controlled comparison."""

    run_configs: dict[str, Mapping[str, Any]] = {}
    for method, checkpoint_dir in checkpoint_dirs.items():
        config_path = checkpoint_dir.parent / "omniquant_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing {method} run config: {config_path}")
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Invalid {method} run config: {config_path}")
        run_configs[method] = loaded

    common_fields = (
        "task",
        "split",
        "calib_split",
        "calib_offset",
        "calibration_only",
        "dtype",
        "seed",
        "layers",
        "weight_quant_format",
        "activation_quant_format",
        "weight_quant_scheme",
        "weight_group_size",
        "act_quant_mode",
        "omni_lwc",
        "omni_let_mode",
        "omni_let_init",
        "omni_epochs",
        "omni_epoch_eval_interval",
        "omni_lwc_lr",
        "omni_let_lr",
        "omni_weight_decay",
        "omni_init_lwc_logit",
        "omni_max_grad_norm",
    )
    for field in common_fields:
        values = {
            method: json.dumps(config.get(field), sort_keys=True)
            for method, config in run_configs.items()
        }
        if len(set(values.values())) != 1:
            raise ValueError(f"Run protocol mismatch for {field}: {values}")

    reference = run_configs["mse"]
    expected_common = {
        "task": task,
        "calib_split": calib_split,
        "calib_offset": 0,
        "calibration_only": True,
        "dtype": "bfloat16",
        "seed": seed,
        "layers": list(range(layer_count)),
        "weight_quant_format": "fp4_e2m1",
        "activation_quant_format": "fp8_e4m3fn",
        "weight_quant_scheme": "symmetric",
        "act_quant_mode": "shared_input",
        "omni_lwc": True,
        "omni_let_mode": expected_omni_let_mode,
        "omni_let_init": expected_omni_let_init,
        "omni_epoch_eval_interval": 0,
    }
    for field, expected in expected_common.items():
        if reference.get(field) != expected:
            raise ValueError(
                f"Controlled protocol requires {field}={expected!r}, "
                f"got {reference.get(field)!r}."
            )
    weight_group_size = reference.get("weight_group_size")
    if isinstance(weight_group_size, bool) or not isinstance(weight_group_size, int) or weight_group_size < 0:
        raise ValueError(
            f"Invalid controlled-protocol weight_group_size: {weight_group_size!r}."
        )
    expected_data_dir = Path(data_dir).resolve()
    for method, config in run_configs.items():
        if Path(str(config.get("data_dir", ""))).resolve() != expected_data_dir:
            raise ValueError(f"{method} data_dir does not match diagnostics: {config.get('data_dir')!r}")

    prefix_paths = {
        method: str(config.get("omni_prefix_checkpoint_dir", ""))
        for method, config in run_configs.items()
    }
    if not all(prefix_paths.values()):
        raise ValueError(f"Every arm must record a prefix checkpoint: {prefix_paths}")
    resolved_prefixes = {str(Path(path).resolve()) for path in prefix_paths.values()}
    if len(resolved_prefixes) != 1:
        raise ValueError(f"Arms do not share one prefix checkpoint: {prefix_paths}")

    prefix_checkpoint_dir = Path(next(iter(resolved_prefixes)))
    prefix_config_path = prefix_checkpoint_dir.parent / "omniquant_config.json"
    if not prefix_config_path.is_file():
        raise FileNotFoundError(f"Missing shared-prefix run config: {prefix_config_path}")
    prefix_config = json.loads(prefix_config_path.read_text(encoding="utf-8"))
    if not isinstance(prefix_config, Mapping):
        raise TypeError(f"Invalid shared-prefix run config: {prefix_config_path}")
    for field in common_fields:
        if field == "layers":
            continue
        if prefix_config.get(field) != reference.get(field):
            raise ValueError(
                f"Shared prefix does not match final arms for {field}: "
                f"{prefix_config.get(field)!r} != {reference.get(field)!r}."
            )
    expected_prefix = {
        "layers": list(range(layer_count - 1)),
        "calib_sample_size": prefix_calib_sample_size,
        "omni_train_sample_size": 0,
        "omni_validation_sample_size": 0,
        "omni_final_objective": "mse",
    }
    for field, expected in expected_prefix.items():
        actual = prefix_config.get(field)
        if field == "calib_sample_size":
            actual = int(actual)
        if actual != expected:
            raise ValueError(
                f"Shared prefix requires {field}={expected!r}, got {actual!r}."
            )

    expected_splits = {
        "mse": (train_sample_size, 0, 0, "mse"),
        "abc": (
            calib_sample_size,
            train_sample_size,
            heldout_sample_size,
            "lfq_ce",
        ),
        "boundary": (
            calib_sample_size,
            train_sample_size,
            heldout_sample_size,
            "lfq_ce",
        ),
    }
    for method, expected in expected_splits.items():
        config = run_configs[method]
        actual = (
            int(config.get("calib_sample_size", -1)),
            int(config.get("omni_train_sample_size", -1)),
            int(config.get("omni_validation_sample_size", -1)),
            config.get("omni_final_objective"),
        )
        if actual != expected:
            raise ValueError(f"{method} train/held-out protocol is {actual}, expected {expected}.")

    expected_boundary = {
        "omni_lfq_boundary_topk": topk,
        "omni_lfq_boundary_negative_count": negative_count,
        "omni_lfq_boundary_tie_threshold": tie_threshold,
        "omni_lfq_boundary_gap_scale": gap_scale,
    }
    for method in ("abc", "boundary"):
        config = run_configs[method]
        for field, expected in expected_boundary.items():
            if config.get(field) != expected:
                raise ValueError(
                    f"{method} {field}={config.get(field)!r}, expected {expected!r}."
                )
    if run_configs["abc"].get("omni_lfq_slot_weights") != run_configs[
        "boundary"
    ].get("omni_lfq_slot_weights"):
        raise ValueError("ABC and boundary arms use different SID slot weights.")
    abc_lfq_loss_weight = run_configs["abc"].get("omni_lfq_loss_weight")
    if abc_lfq_loss_weight != 1.0:
        raise ValueError(
            f"ABC control must use omni_lfq_loss_weight=1.0, got {abc_lfq_loss_weight!r}."
        )
    boundary_lfq_loss_weight = run_configs["boundary"].get("omni_lfq_loss_weight")
    if boundary_lfq_loss_weight != expected_boundary_lfq_loss_weight:
        raise ValueError(
            "Boundary arm LFQ loss weight mismatch: "
            f"{boundary_lfq_loss_weight!r} != {expected_boundary_lfq_loss_weight!r}."
        )

    return {
        "common": {field: reference.get(field) for field in common_fields},
        "shared_prefix_checkpoint_dir": next(iter(resolved_prefixes)),
        "per_method_split": {
            method: {
                "calib_sample_size": expected[0],
                "train_sample_size": expected[1],
                "heldout_sample_size": expected[2],
                "final_objective": expected[3],
                "lfq_loss_weight": run_configs[method].get("omni_lfq_loss_weight"),
            }
            for method, expected in expected_splits.items()
        },
    }


def _load_model(model_path: str, *, dtype: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(dtype),
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model


def _slot_logits(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    slot_ids: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_inputs = _move_tree_to_device(batch, device)
    with torch.inference_mode():
        output = model(**model_inputs, use_cache=False)
    logits = output.logits
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 3:
        raise ValueError(f"Unexpected model logits shape: {tuple(logits.shape)}")
    selected = logits[0, -3:, :].float()
    return {
        slot: selected[slot_idx].index_select(0, slot_ids[slot]).cpu()
        for slot_idx, slot in enumerate(SID_SLOT_NAMES)
    }


def _release_model(model: torch.nn.Module) -> None:
    model.to(torch.device("cpu"))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_teacher_logits(
    *,
    model_path: str,
    dtype: str,
    device: torch.device,
    batches: Sequence[Mapping[str, torch.Tensor]],
    slot_token_ids: Mapping[str, Sequence[int]],
) -> list[dict[str, torch.Tensor]]:
    model = _load_model(model_path, dtype=dtype, device=device)
    ids = {
        slot: torch.as_tensor(token_ids, dtype=torch.long, device=device)
        for slot, token_ids in slot_token_ids.items()
    }
    results = [
        _slot_logits(model, batch, slot_ids=ids, device=device)
        for batch in tqdm(batches, desc="BF16 teacher held-out logits")
    ]
    _release_model(model)
    return results


def evaluate_checkpoint_method(
    *,
    method: str,
    checkpoint_dir: Path,
    config: OmniQuantConfig,
    model_path: str,
    dtype: str,
    device: torch.device,
    batches: Sequence[Mapping[str, torch.Tensor]],
    sample_ids: Sequence[str],
    teacher_logits: Sequence[Mapping[str, torch.Tensor]],
    slot_token_ids: Mapping[str, Sequence[int]],
    topk: int,
    negative_count: int,
    tie_threshold: float,
    gap_scale: float,
) -> dict[str, Any]:
    model = _load_model(model_path, dtype=dtype, device=device)
    layers = get_transformer_layers(model)
    restore_omniquant_layers_from_checkpoints(
        model=model,
        layer_indices=list(range(len(layers))),
        config=config,
        checkpoint_dir=checkpoint_dir,
        act_quant_mode="shared_input",
    )
    ids = {
        slot: torch.as_tensor(token_ids, dtype=torch.long, device=device)
        for slot, token_ids in slot_token_ids.items()
    }
    sample_results: list[dict[str, Any]] = []
    iterator = zip(sample_ids, batches, teacher_logits)
    for sample_id, batch, teacher in tqdm(
        iterator,
        total=len(sample_ids),
        desc=f"{method} held-out diagnostics",
    ):
        student = _slot_logits(model, batch, slot_ids=ids, device=device)
        slots = {
            slot: compute_slot_diagnostics(
                teacher[slot],
                student[slot],
                topk=topk,
                negative_count=negative_count,
                tie_threshold=tie_threshold,
                gap_scale=gap_scale,
            )
            for slot in SID_SLOT_NAMES
        }
        sample_results.append(
            {
                "sample_id": sample_id,
                "slots": slots,
                "macro": _macro_metrics(slots),
            }
        )
    _release_model(model)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_final_objective": config.final_objective,
        "checkpoint_lfq_loss_weight": config.lfq_loss_weight,
        "checkpoint_boundary_loss_weight": config.lfq_boundary_loss_weight,
        "aggregate": aggregate_method_samples(sample_results),
        "samples": sample_results,
    }


def print_method_summary(
    method: str,
    result: Mapping[str, Any],
    *,
    topk: int,
) -> None:
    aggregate = result["aggregate"]
    print(
        f"[lfq held-out summary] method={method} "
        f"samples={aggregate['num_samples']} topk={topk}"
    )
    for slot in SID_SLOT_NAMES:
        metrics = aggregate["slots"][slot]
        print(
            f"[held-out slot {slot.upper()}] "
            f"ce={metrics['cross_entropy']:.8e} "
            f"kl={metrics['kl']:.8e} "
            f"top1={metrics['top1_agreement_rate']:.6f} "
            f"top5={metrics['top5_retention']:.6f} "
            f"top10={metrics['top10_retention']:.6f} "
            f"top{topk}={metrics['topk_retention']:.6f} "
            f"boundary_violation={metrics['boundary_pair_violation_rate']:.6f} "
            f"near_intruder={metrics['intruder_rank_k1_to_kplusn_rate']:.6f} "
            f"far_intruder={metrics['intruder_below_rank_kplusn_rate']:.6f}"
        )
    macro = aggregate["macro"]
    print(
        f"[held-out macro] ce={macro['cross_entropy']:.8e} "
        f"kl={macro['kl']:.8e} "
        f"top1={macro['top1_agreement_rate']:.6f} "
        f"top{topk}={macro['topk_retention']:.6f} "
        f"boundary_violation={macro['boundary_pair_violation_rate']:.6f}"
    )


def _paired_comparisons(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    comparison_pairs = (
        ("abc_minus_mse", "mse", "abc"),
        ("boundary_minus_abc", "abc", "boundary"),
        ("boundary_minus_mse", "mse", "boundary"),
    )
    output: dict[str, Any] = {}
    for comparison_name, reference_name, candidate_name in comparison_pairs:
        reference_samples = methods[reference_name]["samples"]
        candidate_samples = methods[candidate_name]["samples"]
        if [sample["sample_id"] for sample in reference_samples] != [
            sample["sample_id"] for sample in candidate_samples
        ]:
            raise ValueError(f"Sample IDs differ for {comparison_name}.")
        output[comparison_name] = {
            metric: {
                **paired_bootstrap_difference(
                    [sample["macro"][metric] for sample in reference_samples],
                    [sample["macro"][metric] for sample in candidate_samples],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + metric_idx,
                ),
                "candidate_minus_reference": True,
                "preferred_direction": METRIC_DIRECTIONS[metric],
            }
            for metric_idx, metric in enumerate(PRIMARY_METRICS)
        }
    return output


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)

    output_path = resolve_repo_path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")

    model_path = str(resolve_repo_path(args.model_path))
    data_dir = str(resolve_repo_path(args.data_dir))
    single_mode = args.single_checkpoint_dir is not None
    if single_mode:
        method_names = (args.single_method_name.strip(),)
        checkpoint_dirs = {
            method_names[0]: resolve_repo_path(args.single_checkpoint_dir)
        }
    else:
        method_names = METHOD_NAMES
        checkpoint_dirs = {
            "mse": resolve_repo_path(args.mse_checkpoint_dir),
            "abc": resolve_repo_path(args.abc_checkpoint_dir),
            "boundary": resolve_repo_path(args.boundary_checkpoint_dir),
        }
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
        sample_size=args.calib_sample_size,
        require_answer=False,
        drop_final_assistant=True,
    )
    items = list(calibration_data.items())
    if len(items) != args.calib_sample_size:
        raise ValueError(
            f"Loaded {len(items)} calibration samples, expected {args.calib_sample_size}."
        )
    heldout_items = items[
        args.train_sample_size : args.train_sample_size + args.heldout_sample_size
    ]
    sample_ids = [sample_id for sample_id, _sample in heldout_items]
    heldout_samples = [sample for _sample_id, sample in heldout_items]
    batches = build_lfq_sid_slot_batches(
        tokenizer=tokenizer,
        samples=heldout_samples,
        prompt_token=prompt_token,
        device=torch.device("cpu"),
    )
    slot_ids = {
        slot: sid_slot_token_ids(tokenizer, slot)
        for slot in SID_SLOT_NAMES
    }

    device = torch.device(args.device)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    layer_count = int(model_config.num_hidden_layers)
    if single_mode:
        method = method_names[0]
        config, prefix_checkpoint_dir = verify_single_checkpoint_set(
            checkpoint_dirs[method],
            layer_count=layer_count,
            expected_lfq_loss_weight=args.expected_boundary_lfq_loss_weight,
            expected_boundary_loss_weight=args.expected_boundary_loss_weight,
        )
        configs = {method: config}
        verified_controls = verify_single_run_protocol(
            checkpoint_dirs[method],
            prefix_checkpoint_dir=prefix_checkpoint_dir,
            checkpoint_config=config,
            layer_count=layer_count,
            task=args.task,
            data_dir=data_dir,
            calib_split=calib_split,
            calib_sample_size=args.calib_sample_size,
            prefix_calib_sample_size=args.prefix_calib_sample_size,
            train_sample_size=args.train_sample_size,
            heldout_sample_size=args.heldout_sample_size,
            seed=args.seed,
            topk=args.topk,
            negative_count=args.negative_count,
            tie_threshold=args.tie_threshold,
            gap_scale=args.gap_scale,
            expected_lfq_loss_weight=args.expected_boundary_lfq_loss_weight,
            expected_boundary_loss_weight=args.expected_boundary_loss_weight,
            expected_omni_let_mode=args.expected_omni_let_mode,
            expected_omni_let_init=args.expected_omni_let_init,
        )
    else:
        configs = verify_checkpoint_sets(checkpoint_dirs, layer_count=layer_count)
        verified_controls = verify_run_protocols(
            checkpoint_dirs,
            layer_count=layer_count,
            task=args.task,
            data_dir=data_dir,
            calib_split=calib_split,
            calib_sample_size=args.calib_sample_size,
            prefix_calib_sample_size=args.prefix_calib_sample_size,
            train_sample_size=args.train_sample_size,
            heldout_sample_size=args.heldout_sample_size,
            seed=args.seed,
            topk=args.topk,
            negative_count=args.negative_count,
            tie_threshold=args.tie_threshold,
            gap_scale=args.gap_scale,
            expected_boundary_lfq_loss_weight=args.expected_boundary_lfq_loss_weight,
            expected_omni_let_mode=args.expected_omni_let_mode,
            expected_omni_let_init=args.expected_omni_let_init,
        )

    teacher_logits = collect_teacher_logits(
        model_path=model_path,
        dtype=args.dtype,
        device=device,
        batches=batches,
        slot_token_ids=slot_ids,
    )
    methods: dict[str, Any] = {}
    for method in method_names:
        methods[method] = evaluate_checkpoint_method(
            method=method,
            checkpoint_dir=checkpoint_dirs[method],
            config=configs[method],
            model_path=model_path,
            dtype=args.dtype,
            device=device,
            batches=batches,
            sample_ids=sample_ids,
            teacher_logits=teacher_logits,
            slot_token_ids=slot_ids,
            topk=args.topk,
            negative_count=args.negative_count,
            tie_threshold=args.tie_threshold,
            gap_scale=args.gap_scale,
        )

        print_method_summary(method, methods[method], topk=args.topk)

    payload = {
        "protocol": {
            "diagnostic_mode": "single" if single_mode else "three_arm_comparison",
            "model_path": model_path,
            "data_dir": data_dir,
            "task": args.task,
            "calib_split": calib_split,
            "calib_sample_size": args.calib_sample_size,
            "prefix_calib_sample_size": args.prefix_calib_sample_size,
            "train_sample_size": args.train_sample_size,
            "heldout_sample_size": args.heldout_sample_size,
            "heldout_offset": args.train_sample_size,
            "topk": args.topk,
            "negative_count": args.negative_count,
            "tie_threshold": args.tie_threshold,
            "gap_scale": args.gap_scale,
            "expected_boundary_lfq_loss_weight": args.expected_boundary_lfq_loss_weight,
            "expected_omni_let_mode": args.expected_omni_let_mode,
            "expected_boundary_loss_weight": args.expected_boundary_loss_weight,
            "expected_omni_let_init": args.expected_omni_let_init,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": args.device,
            "gt_prefix_conditioned": True,
            "free_running_beam_metric": False,
        },
        "verified_controls": verified_controls,
        "metric_directions": METRIC_DIRECTIONS,
        "head_metric_definitions": HEAD_METRIC_DEFINITIONS,
        "methods": methods,
        "paired_comparisons": (
            {}
            if single_mode
            else _paired_comparisons(
                methods,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(f"[lfq held-out diagnostics] output={output_path}")


if __name__ == "__main__":
    main()
