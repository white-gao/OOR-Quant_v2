"""Matched AD loss-ablation launcher and result collector (no CUDA on import).

Only the two missing objective variants are trained. Original FlatQuant-MSE
is an external baseline, NOT a matched extra-calibration-budget control.
The full-loss reference is the joint-transform RecQuant run, not the frozen
transform arm or the later MSE-1024 continuation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from shared.paths import artifacts_root, benchmark_data_root, model_root, repo_root


METRICS = ("pass@16", "pass@32", "recall@16", "recall@32")
FORMATS = {"w8a8": "fp8_e4m3fn", "w4a8": "fp4_e2m1"}
WEIGHTS = {"recquant": (1.0, 0.3), "wo_boundary": (1.0, 0.0),
           "wo_distribution": (0.0, 0.3)}
ROW_ORDER = (*WEIGHTS, "mse_baseline")
NOTE = ("MSE baseline reuses original FlatQuant (prefix128, final512, heldout512); "
        "it is not a matched 1024-sample joint MSE continuation. Alignment arms "
        "share that MSE initialization and train only block 27 on 1024 samples "
        "for 15 epochs, seed42, final epoch, joint matrix/diagonal/LWC/LAC.")
COMMON = {
    "method": "flatquant_core", "task": "ad", "layers": list(range(28)),
    "weight_group_size": 0, "weight_quant_scheme": "symmetric",
    "activation_quant_format": "fp8_e4m3fn", "act_quant": "per_token",
    "act_quant_mode": "shared_input", "dtype": "bfloat16", "seed": 42,
    "fake_quant_forward_mode": "deployment_matched", "omni_lwc": True,
    "flat_lac": True, "flat_transform_kind": "kronecker",
    "flat_transform_init": "random_orthogonal", "flat_diag_alpha": 0.5,
    "num_beams": 32, "num_return_sequences": 32, "max_new_tokens": 3,
}
ALIGNMENT = {
    "flat_final_objective": "lfq_ce", "flat_learn_transform": True,
    "calib_sample_size": "1024", "calib_offset": 0, "calib_split": "calib",
    "flat_train_sample_size": 1024, "flat_validation_sample_size": 0,
    "flat_epochs": 15, "flat_epoch_eval_interval": 0,
    "flat_transform_lr": 0.005, "flat_lwc_lr": 0.05, "flat_lac_lr": 0.05,
    "flat_weight_decay": 0.01, "flat_max_grad_norm": None,
    "flat_init_lwc_logit": 4.0, "flat_init_lac_logit": 4.0,
    "flat_normalize_mse_gradient": True, "calibration_only": True,
    "flat_lfq_token_scope": "sid_slots", "flat_lfq_vocab_scope": "s_abc",
    "flat_lfq_slot_weights": [1.0, 1.0, 1.0],
    "flat_lfq_boundary_topk": 32, "flat_lfq_boundary_negative_count": 32,
    "flat_lfq_boundary_tie_threshold": 0.01, "flat_lfq_boundary_gap_scale": 1.0,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expect(config: dict, wanted: dict, path: Path) -> None:
    for key, value in wanted.items():
        if key not in config or config[key] != value:
            raise ValueError(f"{path}: {key}={config.get(key)!r}, expected {value!r}")


def same_path(value: str | None, expected: Path, label: str) -> None:
    if value is None or Path(value).resolve() != expected.resolve():
        raise ValueError(f"{label}: {value!r}, expected {expected}")


def boolean(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def selection(name: str, default: str, allowed: tuple) -> tuple[str, ...]:
    values = tuple(os.environ.get(name, default).split())
    if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
        raise ValueError(f"Invalid {name}: {values}; choose from {allowed}")
    return values


@dataclass
class Plan:
    artifacts: Path
    output: Path
    model: Path
    data: Path
    gpus: str
    precisions: tuple[str, ...]
    variants: tuple[str, ...]
    dry_run: bool
    run_eval: bool
    overwrite: bool
    eval_overwrite: bool

    @classmethod
    def from_env(cls) -> "Plan":
        artifacts = artifacts_root().resolve()
        gpus = os.environ.get("GPUS", "2,3").replace(" ", "")
        if not re.fullmatch(r"\d+(,\d+)+", gpus) or len(set(gpus.split(","))) != len(gpus.split(",")):
            raise ValueError("GPUS requires at least two distinct comma-separated GPU IDs")
        overwrite = boolean("OVERWRITE")
        eval_overwrite = boolean("EVAL_OVERWRITE", str(int(overwrite)))
        if overwrite and not eval_overwrite:
            raise ValueError("OVERWRITE=1 requires EVAL_OVERWRITE=1 to avoid stale metrics")
        run_eval = boolean("RUN_EVAL", "1")
        if overwrite and not run_eval:
            raise ValueError("Retraining with OVERWRITE=1 requires RUN_EVAL=1; use a new RUN_ROOT for training-only work")
        return cls(
            artifacts=artifacts,
            output=Path(os.environ.get("RUN_ROOT", str(artifacts / "results/flat_quant/loss_ablation/1p7b_ad_pc_train1024_epoch15"))).resolve(),
            model=Path(os.environ.get("MODEL_PATH", str(model_root() / "1.7B"))).resolve(),
            data=Path(os.environ.get("DATA_DIR", str(benchmark_data_root()))).resolve(),
            gpus=gpus,
            precisions=selection("PRECISIONS", "w8a8 w4a8", tuple(FORMATS)),
            variants=selection("VARIANTS", "wo_boundary wo_distribution", tuple(WEIGHTS)[1:]),
            dry_run=boolean("DRY_RUN"), run_eval=run_eval,
            overwrite=overwrite, eval_overwrite=eval_overwrite,
        )

    def base(self, precision: str) -> Path:
        return self.artifacts / "results/flat_quant/official_fake_quant" / f"1p7b_ad_{precision}_pc_prefix128_final512_heldout512_epoch15"

    def source(self, precision: str) -> Path:
        return self.base(precision) / "final_train512_heldout512/1.7B/ad/flatquant_calibration"

    def training(self, precision: str, variant: str) -> Path:
        if variant == "recquant":
            return self.artifacts / "results/flat_quant/task_alignment" / f"1p7b_ad_{precision}_pc_from_mse_train1024_epoch15/exp3_abc_boundary0.3_joint"
        if variant == "mse_baseline":
            return self.base(precision) / "final_train512_heldout512"
        return self.output / precision / variant

    def evaluation(self, precision: str, variant: str) -> Path:
        if variant == "mse_baseline":
            return self.base(precision) / "ad_full_eval"
        training = self.training(precision, variant)
        return training.with_name(training.name + "_ad_full_eval")

    def checkpoint(self, precision: str, variant: str) -> Path:
        return self.training(precision, variant) / "1.7B/ad/flatquant_calibration"


def config_path(output: Path) -> Path:
    return output / "1.7B/ad/flatquant_core_config.json"


def validate_training(plan: Plan, precision: str, variant: str) -> dict:
    path = config_path(plan.training(precision, variant))
    cfg = read_json(path)
    expect(cfg, {**COMMON, "weight_quant_format": FORMATS[precision]}, path)
    same_path(cfg.get("calib_data_dir"), plan.data, str(path))
    if variant == "mse_baseline":
        # Formal baselines were recorded before flat_final_objective existed.
        expect(cfg, {"omni_final_objective": "mse",
                     "flatquant_variant": "official_svd_cayley_diag_lwc_lac_mse",
                     "flat_train_sample_size": 512,
                     "flat_validation_sample_size": 512, "flat_epochs": 15}, path)
        if "flat_final_objective" in cfg:
            expect(cfg, {"flat_final_objective": "mse"}, path)
    else:
        ce, boundary = WEIGHTS[variant]
        expect(cfg, {**ALIGNMENT, "flat_lfq_loss_weight": ce,
                     "flat_lfq_boundary_loss_weight": boundary}, path)
        same_path(cfg.get("flat_finetune_checkpoint_dir"), plan.source(precision), str(path))
        # Old metadata predates flat_learn_lac; trainable counts verify its state.
        final = cfg["baseline_summaries"]["27"]
        if cfg.get("flat_learn_lac", True) is not True or final.get("best_epoch") is not None:
            raise ValueError(f"{path}: LAC must be trainable and final epoch must be retained")
        for key in ("trainable_transform_parameters", "trainable_lwc_parameters", "trainable_lac_parameters"):
            if final.get(key, 0) <= 0:
                raise ValueError(f"{path}: {key} must be positive")
        if variant != "recquant":
            reference = read_json(config_path(plan.training(precision, "recquant")))["baseline_summaries"]["27"]
            expect(final, {k: reference[k] for k in (
                "trainable_transform_parameters", "trainable_lwc_parameters", "trainable_lac_parameters",
            )}, path)
    return cfg


def validate_evaluation(plan: Plan, precision: str, variant: str) -> dict:
    directory = plan.evaluation(precision, variant)
    path = config_path(directory)
    cfg = read_json(path)
    expect(cfg, {**COMMON, "weight_quant_format": FORMATS[precision],
                 "eval_sample_size": "full", "eval_offset": 0,
                 "split": "test", "eval_merged": True}, path)
    same_path(cfg.get("eval_data_dir"), plan.data, str(path))
    same_path(cfg.get("flat_load_checkpoint_dir"), plan.checkpoint(precision, variant), str(path))
    metrics = read_json(directory / "eval_results.json")["1.7B"]["ad"]["test"]
    if metrics.get("total_samples") != 27677:
        raise ValueError(f"{directory}: expected all 27677 AD test samples")
    for key in METRICS:
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{directory}: invalid {key}={value!r}")
    return metrics


def preflight(plan: Plan, *, need_checkpoints: bool = True) -> None:
    if need_checkpoints:
        model_config = read_json(plan.model / "config.json")
        expect(model_config, {"num_hidden_layers": 28, "hidden_size": 2048}, plan.model / "config.json")
        if plan.model.name != "1.7B" or not plan.data.is_dir():
            raise ValueError("Expected the 1.7B model directory and an existing benchmark data directory")
    for precision in plan.precisions:
        # Never allow new outputs to overwrite reused reference directories.
        for variant in plan.variants:
            for reference in ("recquant", "mse_baseline"):
                if plan.training(precision, variant) == plan.training(precision, reference):
                    raise ValueError("New arm output overlaps a reused reference")
        if need_checkpoints:
            for layer in range(28):
                checkpoint = plan.source(precision) / f"layer_{layer:02d}.pt"
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise ValueError(f"Incomplete source checkpoint: {checkpoint}")
        for variant in ("recquant", "mse_baseline"):
            validate_training(plan, precision, variant)
            validate_evaluation(plan, precision, variant)
        for variant in plan.variants:
            cp = plan.checkpoint(precision, variant)
            cfg_path = config_path(plan.training(precision, variant))
            if not plan.overwrite:
                if cfg_path.exists():
                    validate_training(plan, precision, variant)
                elif all((cp / f"layer_{i:02d}.pt").is_file() for i in range(28)):
                    raise ValueError(f"Complete checkpoints lack completion metadata: {cfg_path}")
            if not plan.eval_overwrite and (plan.evaluation(precision, variant) / "eval_results.json").exists():
                validate_training(plan, precision, variant)
                validate_evaluation(plan, precision, variant)
    print("[check] Reference protocols, results, source checkpoints, and cached outputs verified.", flush=True)


def arm_environment(plan: Plan, precision: str, variant: str) -> dict[str, str]:
    ce, boundary = WEIGHTS[variant]
    # Pin all protocol controls; do not inherit frozen-transform sweep settings.
    fixed = {
        "PYTHON_BIN": sys.executable, "MODEL_PATH": str(plan.model), "DATA_DIR": str(plan.data),
        "TASK": "ad", "QUANT_LABEL": precision, "WEIGHT_QUANT_FORMAT": FORMATS[precision],
        "ACTIVATION_QUANT_FORMAT": "fp8_e4m3fn", "GPUS": plan.gpus,
        "TRAIN_GPU": plan.gpus.split(",")[0], "NUM_LAYERS": "28",
        "CALIB_SAMPLES": "1024", "TRAIN_SAMPLES": "1024", "EPOCHS": "15",
        "LEARN_TRANSFORM": "1", "LEARN_LAC": "1", "LFQ_LOSS_WEIGHT": str(ce),
        "BOUNDARY_WEIGHT": str(boundary), "TRANSFORM_LR": "5e-3",
        "LWC_LR": "5e-2", "LAC_LR": "5e-2", "WEIGHT_DECAY": "0.01",
        "DIAG_ALPHA": "0.5", "INIT_LWC_LOGIT": "4.0", "INIT_LAC_LOGIT": "4.0",
        "BOUNDARY_TOPK": "32", "BOUNDARY_NEGATIVES": "32",
        "BOUNDARY_TIE_THRESHOLD": "0.01", "BOUNDARY_GAP_SCALE": "1.0",
        "SOURCE_CHECKPOINT_DIR": str(plan.source(precision)),
        "OUTPUT_DIR": str(plan.training(precision, variant)),
        "CHECKPOINT_DIR": str(plan.checkpoint(precision, variant)),
        "EVAL_OUTPUT_DIR": str(plan.evaluation(precision, variant)), "EVAL_SAMPLE_SIZE": "full",
        "DRY_RUN": str(int(plan.dry_run)), "RUN_EVAL": str(int(plan.run_eval)),
        "OVERWRITE": str(int(plan.overwrite)), "EVAL_OVERWRITE": str(int(plan.eval_overwrite)),
    }
    return {**os.environ, **fixed}


def collect(plan: Plan) -> list[dict]:
    rows = []
    for precision in plan.precisions:
        for variant in ROW_ORDER:
            result = plan.evaluation(precision, variant) / "eval_results.json"
            row = {"precision": precision, "variant": variant,
                   "status": "pending", **{key: None for key in METRICS},
                   "total_samples": None, "result_path": str(result),
                   "training_config": str(config_path(plan.training(precision, variant))),
                   "source_checkpoint": str(plan.source(precision))}
            if result.exists():
                validate_training(plan, precision, variant)
                metrics = validate_evaluation(plan, precision, variant)
                row.update(status="complete", total_samples=metrics["total_samples"])
                row.update({key: 100 * metrics[key] for key in METRICS})
            rows.append(row)
    if not plan.dry_run:
        plan.output.mkdir(parents=True, exist_ok=True)
        payload = {"metric_units": "percent", "protocol_note": NOTE, "rows": rows}
        (plan.output / "loss_ablation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (plan.output / "loss_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, **{k: "" if row[k] is None else f"{row[k]:.3f}" for k in METRICS}})
    print("precision  variant          P@16    P@32    R@16    R@32", flush=True)
    for row in rows:
        numbers = " ".join(f"{row[k]:7.3f}" if row[k] is not None else "     --" for k in METRICS)
        print(f"{row['precision']:10} {row['variant']:16} {numbers}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true", help="Read-only protocol/checkpoint validation")
    actions.add_argument("--collect-only", action="store_true", help="Refresh CSV/JSON, never launch training")
    args = parser.parse_args()
    plan = Plan.from_env()
    preflight(plan, need_checkpoints=not args.collect_only)
    print(f"[protocol] {NOTE}", flush=True)
    print(f"[output] {plan.output}", flush=True)
    if args.check_only:
        return
    if args.collect_only:
        collect(plan)
        return
    runner = repo_root() / "scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh"
    for precision in plan.precisions:
        for variant in plan.variants:
            print(f"\n[experiment] {precision} {variant}; CE/boundary={WEIGHTS[variant]}", flush=True)
            if (not plan.overwrite and not plan.eval_overwrite
                    and (plan.evaluation(precision, variant) / "eval_results.json").exists()):
                print("[skip] Validated completed evaluation; no training or generation needed.", flush=True)
                collect(plan)
                continue
            subprocess.run(["bash", str(runner)], cwd=repo_root(),
                           env=arm_environment(plan, precision, variant), check=True)
            if not plan.dry_run:
                validate_training(plan, precision, variant)
                if plan.run_eval:
                    validate_evaluation(plan, precision, variant)
                collect(plan)
    if plan.dry_run:
        print("[dry-run] No model loaded and no output files written.", flush=True)
    else:
        print(f"[done] {plan.output / 'loss_ablation.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
