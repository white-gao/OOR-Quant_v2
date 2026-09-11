#!/usr/bin/env python3
"""Held-out ABC/boundary diagnostics for matched FlatQuant fine-tune arms."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .evaluate_lfq_boundary_diagnostics import (
    HEAD_METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    _load_model,
    _macro_metrics,
    _release_model,
    _slot_logits,
    aggregate_method_samples,
    collect_teacher_logits,
    compute_slot_diagnostics,
    paired_bootstrap_difference,
    print_method_summary,
)
from .flatquant import FlatQuantCoreConfig, restore_flatquant_core_layers_from_checkpoints
from .flatquant.runtime import _load_flatquant_checkpoint
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    SID_SLOT_NAMES,
    build_lfq_sid_slot_batches,
    default_calib_split,
    get_task_config,
    get_transformer_layers,
    load_task_data,
    resolve_repo_path,
    set_seed,
    sid_slot_token_ids,
)


METHOD_NAMES = (
    "mse_continuation",
    "abc_boundary_frozen_transform",
    "abc_boundary_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three final-block FlatQuant continuations initialized from "
            "one complete MSE checkpoint on a fixed held-out calibration tail."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=("ad", "product", "video"))
    parser.add_argument("--calib_split", default="auto")
    parser.add_argument("--calib_sample_size", type=int, default=1024)
    parser.add_argument("--train_sample_size", type=int, default=512)
    parser.add_argument("--heldout_sample_size", type=int, default=512)
    parser.add_argument("--base_checkpoint_dir", required=True)
    parser.add_argument("--mse_checkpoint_dir", required=True)
    parser.add_argument("--frozen_checkpoint_dir", required=True)
    parser.add_argument("--joint_checkpoint_dir", required=True)
    parser.add_argument("--expected_epochs", type=int, default=15)
    parser.add_argument("--expected_boundary_weight", type=float, default=0.3)
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
    if args.calib_sample_size <= 0:
        raise ValueError("--calib_sample_size must be positive.")
    if args.train_sample_size <= 0 or args.heldout_sample_size <= 0:
        raise ValueError("Train and held-out sample counts must be positive.")
    if args.train_sample_size + args.heldout_sample_size != args.calib_sample_size:
        raise ValueError("Train plus held-out samples must equal calibration samples.")
    if args.expected_epochs <= 0:
        raise ValueError("--expected_epochs must be positive.")
    if not math.isfinite(args.expected_boundary_weight) or args.expected_boundary_weight <= 0:
        raise ValueError("--expected_boundary_weight must be finite and positive.")
    if args.topk <= 0 or args.negative_count <= 0:
        raise ValueError("--topk and --negative_count must be positive.")
    if not math.isfinite(args.tie_threshold) or args.tie_threshold < 0:
        raise ValueError("--tie_threshold must be finite and non-negative.")
    if not math.isfinite(args.gap_scale) or args.gap_scale <= 0:
        raise ValueError("--gap_scale must be finite and positive.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive.")


def _checkpoint_config(checkpoint_dir: Path, final_layer_idx: int) -> FlatQuantCoreConfig:
    state = _load_flatquant_checkpoint(
        checkpoint_dir / f"layer_{final_layer_idx:02d}.pt",
        layer_idx=final_layer_idx,
    )
    saved = state.get("config")
    if not isinstance(saved, Mapping):
        raise TypeError(f"Missing FlatQuant config in {checkpoint_dir}.")
    allowed = {field.name for field in fields(FlatQuantCoreConfig)}
    values = {key: value for key, value in saved.items() if key in allowed}
    if "lfq_slot_weights" in values:
        values["lfq_slot_weights"] = tuple(values["lfq_slot_weights"])
    config = FlatQuantCoreConfig(**values)
    config.validate()
    return config


def _load_run_config(checkpoint_dir: Path) -> Mapping[str, Any]:
    path = checkpoint_dir.parent / "flatquant_core_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FlatQuant run config: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Invalid FlatQuant run config: {path}")
    return loaded


def verify_controlled_protocol(
    checkpoint_dirs: Mapping[str, Path],
    *,
    base_checkpoint_dir: Path,
    layer_count: int,
    data_dir: str,
    task: str,
    calib_split: str,
    calib_sample_size: int,
    train_sample_size: int,
    heldout_sample_size: int,
    expected_epochs: int,
    expected_boundary_weight: float,
    seed: int,
) -> tuple[dict[str, FlatQuantCoreConfig], dict[str, Any]]:
    base = base_checkpoint_dir.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Base FlatQuant checkpoint directory not found: {base}")
    for layer_idx in range(layer_count):
        path = base / f"layer_{layer_idx:02d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing base checkpoint: {path}")

    for method, checkpoint_dir in checkpoint_dirs.items():
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"{method} checkpoint directory not found: {checkpoint_dir}")
        for layer_idx in range(layer_count):
            path = checkpoint_dir / f"layer_{layer_idx:02d}.pt"
            state = _load_flatquant_checkpoint(path, layer_idx=layer_idx)
            if layer_idx < layer_count - 1:
                expected_source = (base / f"layer_{layer_idx:02d}.pt").resolve()
                source = state.get("source_checkpoint")
                if not isinstance(source, str) or Path(source).resolve() != expected_source:
                    raise ValueError(
                        f"{method} layer {layer_idx} was not inherited from {expected_source}."
                    )
            else:
                expected_initialization = (base / f"layer_{layer_idx:02d}.pt").resolve()
                initialization = state.get("initialization_checkpoint")
                if (
                    not isinstance(initialization, str)
                    or Path(initialization).resolve() != expected_initialization
                ):
                    raise ValueError(
                        f"{method} final layer was not initialized from {expected_initialization}."
                    )

    configs = {
        method: _checkpoint_config(checkpoint_dir, layer_count - 1)
        for method, checkpoint_dir in checkpoint_dirs.items()
    }
    expected_objectives = {
        "mse_continuation": ("mse", True, 0.0),
        "abc_boundary_frozen_transform": ("lfq_ce", False, expected_boundary_weight),
        "abc_boundary_joint": ("lfq_ce", True, expected_boundary_weight),
    }
    for method, (objective, learn_transform, boundary_weight) in expected_objectives.items():
        config = configs[method]
        if config.final_objective != objective:
            raise ValueError(f"{method} objective is {config.final_objective!r}, expected {objective!r}.")
        if config.learn_transform is not learn_transform:
            raise ValueError(f"{method} learn_transform={config.learn_transform}, expected {learn_transform}.")
        if not math.isclose(config.lfq_boundary_loss_weight, boundary_weight, abs_tol=1e-12):
            raise ValueError(f"{method} boundary weight does not match the controlled protocol.")
        if not config.use_lwc or not config.use_lac:
            raise ValueError(f"{method} must train/use both LWC and LAC.")
        if config.train_sample_size != train_sample_size or config.validation_sample_size != heldout_sample_size:
            raise ValueError(f"{method} train/held-out split does not match diagnostics.")
        if config.epochs != expected_epochs or config.epoch_eval_interval != 0:
            raise ValueError(f"{method} epoch protocol does not match diagnostics.")

    common_config_fields = (
        "weight_quant_format",
        "activation_quant_format",
        "weight_quant_scheme",
        "weight_group_size",
        "use_lwc",
        "use_lac",
        "transform_kind",
        "transform_init",
        "diag_alpha",
        "epochs",
        "train_sample_size",
        "validation_sample_size",
        "epoch_eval_interval",
        "transform_lr",
        "lwc_lr",
        "lac_lr",
        "weight_decay",
        "init_lwc_logit",
        "init_lac_logit",
        "max_grad_norm",
    )
    for field_name in common_config_fields:
        values = {method: getattr(config, field_name) for method, config in configs.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            raise ValueError(f"Checkpoint config mismatch for {field_name}: {values}")

    run_configs = {
        method: _load_run_config(checkpoint_dir)
        for method, checkpoint_dir in checkpoint_dirs.items()
    }
    common_run_fields = (
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
        "flat_lac",
        "flat_epochs",
        "flat_train_sample_size",
        "flat_validation_sample_size",
        "flat_epoch_eval_interval",
        "flat_transform_lr",
        "flat_lwc_lr",
        "flat_lac_lr",
        "flat_weight_decay",
        "flat_diag_alpha",
        "flat_init_lwc_logit",
        "flat_init_lac_logit",
        "flat_max_grad_norm",
    )
    for field_name in common_run_fields:
        values = {
            method: json.dumps(run_config.get(field_name), sort_keys=True)
            for method, run_config in run_configs.items()
        }
        if len(set(values.values())) != 1:
            raise ValueError(f"Run protocol mismatch for {field_name}: {values}")

    expected_run = {
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
        "weight_group_size": 0,
        "act_quant_mode": "shared_input",
        "calib_sample_size": str(calib_sample_size),
        "flat_epochs": expected_epochs,
        "flat_train_sample_size": train_sample_size,
        "flat_validation_sample_size": heldout_sample_size,
        "flat_epoch_eval_interval": 0,
    }
    reference = run_configs["mse_continuation"]
    for field_name, expected in expected_run.items():
        if reference.get(field_name) != expected:
            raise ValueError(
                f"Controlled protocol requires {field_name}={expected!r}, "
                f"got {reference.get(field_name)!r}."
            )
    if Path(str(reference.get("data_dir", ""))).resolve() != Path(data_dir).resolve():
        raise ValueError("Run data_dir does not match diagnostic data_dir.")
    for method, run_config in run_configs.items():
        recorded_base = Path(str(run_config.get("flat_finetune_checkpoint_dir", ""))).resolve()
        if recorded_base != base:
            raise ValueError(f"{method} records a different fine-tune initialization: {recorded_base}")

    for method in ("abc_boundary_frozen_transform", "abc_boundary_joint"):
        run_config = run_configs[method]
        if float(run_config.get("flat_lfq_loss_weight", float("nan"))) != 1.0:
            raise ValueError(f"{method} must use ABC CE weight 1.0.")
        if not math.isclose(
            float(run_config.get("flat_lfq_boundary_loss_weight", float("nan"))),
            expected_boundary_weight,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{method} boundary weight mismatch.")

    return configs, {
        "base_checkpoint_dir": str(base),
        "common_checkpoint_fields": {
            field_name: getattr(configs["mse_continuation"], field_name)
            for field_name in common_config_fields
        },
        "per_method": {
            method: {
                "checkpoint_dir": str(checkpoint_dirs[method].resolve()),
                "objective": config.final_objective,
                "learn_transform": config.learn_transform,
                "lfq_loss_weight": config.lfq_loss_weight,
                "boundary_loss_weight": config.lfq_boundary_loss_weight,
            }
            for method, config in configs.items()
        },
    }


def evaluate_method(
    *,
    method: str,
    checkpoint_dir: Path,
    config: FlatQuantCoreConfig,
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
    restore_flatquant_core_layers_from_checkpoints(
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
            {"sample_id": sample_id, "slots": slots, "macro": _macro_metrics(slots)}
        )
    _release_model(model)
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_final_objective": config.final_objective,
        "learn_transform": config.learn_transform,
        "checkpoint_lfq_loss_weight": config.lfq_loss_weight,
        "checkpoint_boundary_loss_weight": config.lfq_boundary_loss_weight,
        "aggregate": aggregate_method_samples(sample_results),
        "samples": sample_results,
    }


def paired_comparisons(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    pairs = (
        ("frozen_minus_mse", "mse_continuation", "abc_boundary_frozen_transform"),
        ("joint_minus_mse", "mse_continuation", "abc_boundary_joint"),
        ("joint_minus_frozen", "abc_boundary_frozen_transform", "abc_boundary_joint"),
    )
    output: dict[str, Any] = {}
    for comparison, reference_name, candidate_name in pairs:
        reference = methods[reference_name]["samples"]
        candidate = methods[candidate_name]["samples"]
        if [item["sample_id"] for item in reference] != [item["sample_id"] for item in candidate]:
            raise ValueError(f"Held-out sample IDs differ for {comparison}.")
        output[comparison] = {
            metric: {
                **paired_bootstrap_difference(
                    [item["macro"][metric] for item in reference],
                    [item["macro"][metric] for item in candidate],
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
    checkpoint_dirs = {
        "mse_continuation": resolve_repo_path(args.mse_checkpoint_dir),
        "abc_boundary_frozen_transform": resolve_repo_path(args.frozen_checkpoint_dir),
        "abc_boundary_joint": resolve_repo_path(args.joint_checkpoint_dir),
    }
    base_checkpoint_dir = resolve_repo_path(args.base_checkpoint_dir)

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
        raise ValueError(f"Loaded {len(items)} calibration samples, expected {args.calib_sample_size}.")
    heldout_items = items[args.train_sample_size : args.train_sample_size + args.heldout_sample_size]
    sample_ids = [sample_id for sample_id, _sample in heldout_items]
    heldout_samples = [sample for _sample_id, sample in heldout_items]
    batches = build_lfq_sid_slot_batches(
        tokenizer=tokenizer,
        samples=heldout_samples,
        prompt_token=prompt_token,
        device=torch.device("cpu"),
    )
    slot_ids = {slot: sid_slot_token_ids(tokenizer, slot) for slot in SID_SLOT_NAMES}

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    layer_count = int(model_config.num_hidden_layers)
    configs, verified_controls = verify_controlled_protocol(
        checkpoint_dirs,
        base_checkpoint_dir=base_checkpoint_dir,
        layer_count=layer_count,
        data_dir=data_dir,
        task=args.task,
        calib_split=calib_split,
        calib_sample_size=args.calib_sample_size,
        train_sample_size=args.train_sample_size,
        heldout_sample_size=args.heldout_sample_size,
        expected_epochs=args.expected_epochs,
        expected_boundary_weight=args.expected_boundary_weight,
        seed=args.seed,
    )

    device = torch.device(args.device)
    teacher_logits = collect_teacher_logits(
        model_path=model_path,
        dtype=args.dtype,
        device=device,
        batches=batches,
        slot_token_ids=slot_ids,
    )
    methods: dict[str, Any] = {}
    for method in METHOD_NAMES:
        methods[method] = evaluate_method(
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
            "model_path": model_path,
            "data_dir": data_dir,
            "task": args.task,
            "calib_split": calib_split,
            "calib_sample_size": args.calib_sample_size,
            "train_sample_size": args.train_sample_size,
            "heldout_sample_size": args.heldout_sample_size,
            "heldout_offset": args.train_sample_size,
            "topk": args.topk,
            "negative_count": args.negative_count,
            "tie_threshold": args.tie_threshold,
            "gap_scale": args.gap_scale,
            "expected_boundary_weight": args.expected_boundary_weight,
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
        "paired_comparisons": paired_comparisons(
            methods,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output_path)
    print(f"[flatquant held-out diagnostics] output={output_path}")


if __name__ == "__main__":
    main()
