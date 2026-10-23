"""Evaluate fixed RecQuant checkpoints across domains, without calibration.

W8A8 per-channel, OneRec-1.7B. Product/Video are evaluated serially; AD and
in-domain references are reused. No inference uses target calibration rows.
--complete-matrix adds Product->AD/Video and Video->AD/Product, reusing the
completed AD transfers and all three diagonal results without overwriting them.
Protocol constants are shared with the matched loss-ablation launcher.
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
from .loss_ablation import ALIGNMENT, COMMON, METRICS, boolean, expect, read_json, same_path, selection


DOMAINS = ("ad", "product", "video")
SAMPLE_COUNTS = {"ad": 27677, "product": 27910, "video": 38781}
MATRIX_PAIRS = ("product:ad", "product:video", "video:ad", "video:product")


@dataclass
class Plan:
    artifacts: Path
    output: Path
    model: Path
    data: Path
    gpus: str
    targets: tuple[str, ...]
    dry_run: bool
    overwrite: bool
    matrix: bool = False
    ad_output: Path | None = None
    pairs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_env(cls, *, matrix: bool = False) -> "Plan":
        artifacts = artifacts_root().resolve()
        cross_root = artifacts / "results/flat_quant/cross_domain"
        default_output = cross_root / ("1p7b_w8a8_pc_recquant_matrix" if matrix else "1p7b_w8a8_pc_ad_recquant")
        gpus = os.environ.get("GPUS", "2,3").replace(" ", "")
        if not re.fullmatch(r"\d+(,\d+)+", gpus) or len(set(gpus.split(","))) != len(gpus.split(",")):
            raise ValueError("GPUS must list at least two distinct GPU IDs")
        return cls(
            artifacts=artifacts,
            output=Path(os.environ.get("RUN_ROOT", str(default_output))).resolve(),
            model=Path(os.environ.get("MODEL_PATH", str(model_root() / "1.7B"))).resolve(),
            data=Path(os.environ.get("DATA_DIR", str(benchmark_data_root()))).resolve(),
            gpus=gpus, targets=() if matrix else selection("TARGETS", "product video", DOMAINS[1:]),
            dry_run=boolean("DRY_RUN"), overwrite=boolean("EVAL_OVERWRITE"),
            matrix=matrix,
            ad_output=Path(os.environ.get("AD_RESULTS_ROOT", str(cross_root / "1p7b_w8a8_pc_ad_recquant"))).resolve() if matrix else None,
            pairs=tuple(tuple(p.split(":")) for p in selection("PAIRS", " ".join(MATRIX_PAIRS), MATRIX_PAIRS)) if matrix else (),
        )

    def requested_pairs(self) -> tuple[tuple[str, str], ...]:
        return self.pairs if self.matrix else tuple(("ad", task) for task in self.targets)

    def reference_training(self, task: str) -> Path:
        source = "mse" if task == "ad" else "official_mse"
        return self.artifacts / "results/flat_quant/task_alignment" / f"1p7b_{task}_w8a8_pc_from_{source}_train1024_epoch15/exp3_abc_boundary0.3_joint"

    def reference_eval(self, task: str) -> Path:
        train = self.reference_training(task)
        return train.with_name(train.name + f"_{task}_full_eval")

    def checkpoint(self, task: str = "ad") -> Path:
        return self.reference_training(task) / f"1.7B/{task}/flatquant_calibration"

    def transferred_eval(self, task: str, source: str = "ad") -> Path:
        if source == task:
            return self.reference_eval(task)
        if self.matrix and source == "ad":
            if self.ad_output is None:
                raise ValueError("Matrix evaluation requires the completed AD_RESULTS_ROOT")
            return self.ad_output / f"ad_to_{task}_full_eval"
        return self.output / f"{source}_to_{task}_full_eval"


def config_path(directory: Path, task: str) -> Path:
    return directory / f"1.7B/{task}/flatquant_core_config.json"


def validate_reference_training(plan: Plan, task: str) -> None:
    path = config_path(plan.reference_training(task), task)
    cfg = read_json(path)
    expect(cfg, {**COMMON, **ALIGNMENT, "task": task,
                 "weight_quant_format": "fp8_e4m3fn",
                 "flat_lfq_loss_weight": 1.0, "flat_lfq_boundary_loss_weight": 0.3}, path)
    same_path(cfg.get("calib_data_dir"), plan.data, str(path))
    base = plan.artifacts / "results/flat_quant/official_fake_quant" / f"1p7b_{task}_w8a8_pc_prefix128_final512_heldout512_epoch15/final_train512_heldout512/1.7B/{task}/flatquant_calibration"
    same_path(cfg.get("flat_finetune_checkpoint_dir"), base, str(path))
    final = cfg["baseline_summaries"]["27"]
    if cfg.get("flat_learn_lac", True) is not True or final.get("best_epoch") is not None:
        raise ValueError(f"{path}: expected trainable LAC and final-epoch checkpoint")
    for key in ("trainable_transform_parameters", "trainable_lwc_parameters", "trainable_lac_parameters"):
        if final.get(key, 0) <= 0:
            raise ValueError(f"{path}: expected joint optimization, missing {key}")


def evaluation_metrics(plan: Plan, task: str, *, transferred: bool = False, source: str = "ad") -> dict:
    directory = plan.transferred_eval(task, source) if transferred else plan.reference_eval(task)
    path = config_path(directory, task)
    cfg = read_json(path)
    expect(cfg, {**COMMON, "task": task, "weight_quant_format": "fp8_e4m3fn",
                 "eval_sample_size": "full", "eval_offset": 0,
                 "split": "test", "eval_merged": True, "calibration_only": False}, path)
    same_path(cfg.get("eval_data_dir"), plan.data, str(path))
    same_path(cfg.get("flat_load_checkpoint_dir"), plan.checkpoint(source if transferred else task), str(path))
    if transferred and source != task:
        expect(cfg, {"calibration_data_loaded": False, "calib_split": None,
                     "flat_finetune_checkpoint_dir": None, "flat_prefix_checkpoint_dir": None}, path)
    metrics = read_json(directory / "eval_results.json")["1.7B"][task]["test"]
    if metrics.get("total_samples") != SAMPLE_COUNTS[task]:
        raise ValueError(f"{directory}: expected {SAMPLE_COUNTS[task]} test samples")
    for key in METRICS:
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{directory}: invalid {key}: {value!r}")
    return metrics


def preflight(plan: Plan, *, require_checkpoint: bool = True, allow_partial: bool = False) -> None:
    if plan.matrix:
        if plan.ad_output is None:
            raise ValueError("Matrix evaluation requires AD_RESULTS_ROOT")
        # Even explicit overwrite only affects the four new off-diagonal runs.
        protected = [plan.ad_output, plan.model, plan.data]
        protected.extend(plan.reference_training(t) for t in DOMAINS)
        protected.extend(plan.reference_eval(t) for t in DOMAINS)
        for path in protected:
            output, reference = plan.output.resolve(), path.resolve()
            if output == reference or output in reference.parents or reference in output.parents:
                raise ValueError(f"Matrix RUN_ROOT overlaps a protected reference: {reference}")
    if require_checkpoint:
        expect(read_json(plan.model / "config.json"),
               {"num_hidden_layers": 28, "hidden_size": 2048}, plan.model / "config.json")
        if plan.model.name != "1.7B" or not plan.data.is_dir():
            raise ValueError("Expected the 1.7B model and an existing benchmark data directory")
        for source in dict.fromkeys(s for s, _ in plan.requested_pairs()):
            for layer in range(28):
                path = plan.checkpoint(source) / f"layer_{layer:02d}.pt"
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Incomplete {source} checkpoint: {path}")
    for task in DOMAINS:
        validate_reference_training(plan, task)
        evaluation_metrics(plan, task)
    if plan.matrix:
        # These are mandatory read-only references, including under overwrite.
        for task in DOMAINS[1:]:
            evaluation_metrics(plan, task, transferred=True, source="ad")
        for pair in MATRIX_PAIRS:
            source, task = pair.split(":")
            if (source, task) not in plan.pairs and (plan.transferred_eval(task, source) / "eval_results.json").exists():
                evaluation_metrics(plan, task, transferred=True, source=source)
    for source, task in plan.requested_pairs():
        output = plan.transferred_eval(task, source)
        if not plan.overwrite:
            if (output / "eval_results.json").exists():
                evaluation_metrics(plan, task, transferred=True, source=source)
            elif not allow_partial and (output.exists() or output.with_name(output.name + ".shards").exists()):
                raise ValueError(f"Partial evaluation at {output}; use EVAL_OVERWRITE=1 to rerun it")
    print("[check] Source checkpoints and all three in-domain RecQuant references verified."
          + (" Both completed AD transfers are protected and reused." if plan.matrix else ""), flush=True)


def evaluation_command(plan: Plan, source: str = "ad") -> list[str]:
    return ["bash", str(repo_root() / "scripts/flat_quant/run_sharded_eval_cuda.sh"),
            "--mode", "flatquant_core", "--layers", "all",
            "--weight_quant_format", "fp8_e4m3fn", "--activation_quant_format", "fp8_e4m3fn",
            "--weight_quant_scheme", "symmetric", "--weight_group_size", "0",
            "--omni_lwc", "--flat_lac", "--flat_learn_transform", "--flat_learn_lac",
            "--flat_transform_kind", "kronecker", "--flat_transform_init", "random_orthogonal",
            "--flat_diag_alpha", "0.5", "--flat_init_lwc_logit", "4.0", "--flat_init_lac_logit", "4.0",
            "--flat_load_checkpoint_dir", str(plan.checkpoint(source)), "--eval_sample_size", "full"]


def collect_matrix(plan: Plan) -> list[dict]:
    """Rows are calibration domains; each target's diagonal is its denominator."""
    references = {task: evaluation_metrics(plan, task) for task in DOMAINS}
    rows = []
    for source in DOMAINS:
        for task in DOMAINS:
            output = plan.transferred_eval(task, source)
            ref = references[task]
            measured = evaluation_metrics(plan, task, transferred=True, source=source) if (output / "eval_results.json").exists() else None
            row = {"calibration_domain": source, "test_domain": task,
                   "status": "reused" if source == task or source == "ad" else ("complete" if measured is not None else "pending"),
                   "total_samples": ref["total_samples"],
                   "source_checkpoint": str(plan.checkpoint(source)),
                   "result_path": str(output / "eval_results.json"),
                   "in_domain_result": str(plan.reference_eval(task) / "eval_results.json")}
            if measured is None and row["status"] == "reused":
                raise ValueError(f"Missing reused matrix result: {output}")
            for metric in METRICS:
                row[metric] = None if measured is None else 100 * measured[metric]
                row[f"in_domain_{metric}"] = 100 * ref[metric]
                row[f"delta_{metric}_pp"] = None if measured is None else 100 * (measured[metric] - ref[metric])
                row[f"retention_{metric}_pct"] = None if measured is None or ref[metric] == 0 else 100 * measured[metric] / ref[metric]
            rows.append(row)
    if not plan.dry_run:
        plan.output.mkdir(parents=True, exist_ok=True)
        payload = {"method": "RecQuant", "model": "OneRec-1.7B", "precision": "W8A8 per-channel",
                   "metric_units": "percent", "row_axis": "calibration_domain", "column_axis": "test_domain",
                   "domain_order": list(DOMAINS),
                   "delta_definition": "Transferred minus target-domain-calibrated, percentage points",
                   "retention_definition": "100 * transferred metric / target-domain diagonal metric",
                   "note": "Three diagonal and two AD-transfer results are reused. Only four remaining transfers are new. No target calibration or parameter updates.",
                   "rows": rows,
                   "matrices": {metric: [[rows[3*i+j][metric] for j in range(3)] for i in range(3)] for metric in METRICS},
                   "retention_matrices_pct": {metric: [[rows[3*i+j][f"retention_{metric}_pct"] for j in range(3)] for i in range(3)] for metric in METRICS}}
        (plan.output / "cross_domain_matrix.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (plan.output / "cross_domain_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if val is None else f"{val:.6f}" if isinstance(val, float) else val
                                 for key, val in row.items()})
    print("[matrix] Recall@32 (%); rows=calibration, columns=test [ad, product, video]", flush=True)
    for i, source in enumerate(DOMAINS):
        values = [rows[3*i+j]["recall@32"] for j in range(3)]
        print(f"         {source:8} " + " ".join("      --" if v is None else f"{v:8.3f}" for v in values), flush=True)
    return rows


def collect(plan: Plan) -> list[dict]:
    if plan.matrix:
        return collect_matrix(plan)
    rows = []
    for task in DOMAINS:
        reference = evaluation_metrics(plan, task)
        output = plan.transferred_eval(task)
        cross = evaluation_metrics(plan, task, transferred=True) if (output / "eval_results.json").exists() else None
        row = {"test_domain": task, "source_domain": "ad",
               "status": "reused" if task == "ad" else ("complete" if cross is not None else "pending"),
               "total_samples": reference["total_samples"],
               "in_domain_result": str(plan.reference_eval(task) / "eval_results.json"),
               "ad_calibrated_result": str(output / "eval_results.json")}
        for metric in METRICS:
            row[f"in_domain_{metric}"] = 100 * reference[metric]
            row[f"ad_calibrated_{metric}"] = None if cross is None else 100 * cross[metric]
            row[f"delta_{metric}_pp"] = None if cross is None else 100 * (cross[metric] - reference[metric])
        rows.append(row)
    if not plan.dry_run:
        plan.output.mkdir(parents=True, exist_ok=True)
        payload = {"method": "RecQuant", "model": "OneRec-1.7B", "precision": "W8A8 per-channel",
                   "source_checkpoint": str(plan.checkpoint()), "metric_units": "percent",
                   "delta_definition": "AD-calibrated minus in-domain-calibrated, percentage points",
                   "note": "AD row is reused, not an independent transfer experiment. No target-domain calibration or parameter updates.",
                   "rows": rows}
        (plan.output / "cross_domain.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (plan.output / "cross_domain.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if val is None else f"{val:.3f}" if isinstance(val, float) else val
                                 for key, val in row.items()})
    print("[summary] domain    in-domain R@32    AD-calibrated R@32    delta (pp)", flush=True)
    for row in rows:
        value = row["ad_calibrated_recall@32"]
        delta = row["delta_recall@32_pp"]
        print(f"          {row['test_domain']:8} {row['in_domain_recall@32']:10.3f}"
              + (f" {value:20.3f} {delta:+13.3f}" if value is not None else "                   --            --"), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-matrix", action="store_true", help="Run the four non-AD transfers and collect all nine cells")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    plan = Plan.from_env(matrix=args.complete_matrix)
    preflight(plan, require_checkpoint=not args.collect_only, allow_partial=args.collect_only)
    pairs = plan.requested_pairs()
    for source in dict.fromkeys(s for s, _ in pairs):
        print(f"[protocol] {source} checkpoint={plan.checkpoint(source)}", flush=True)
    print(f"[protocol] evaluation only, serial pairs={','.join(s+'->'+t for s,t in pairs)}, GPUs={plan.gpus}", flush=True)
    if plan.matrix:
        print(f"[reuse] Three diagonal references and both AD transfers at {plan.ad_output}; never rerun or overwrite these.", flush=True)
    print(f"[output] {plan.output}", flush=True)
    if args.check_only:
        return
    if args.collect_only:
        collect(plan)
        return
    for index, (source, task) in enumerate(pairs, 1):
        output = plan.transferred_eval(task, source)
        if (output / "eval_results.json").exists() and not plan.overwrite:
            print(f"[skip] {source} -> {task}: validated complete result exists", flush=True)
        else:
            print(f"[evaluation {index}/{len(pairs)}] {source} -> {task}", flush=True)
            env = {**os.environ, "TASK": task, "GPUS": plan.gpus,
                   "MODEL_PATH": str(plan.model), "DATA_DIR": str(plan.data),
                   "OUTPUT_DIR": str(output), "PYTHON_BIN": sys.executable,
                   "OVERWRITE": str(int(plan.overwrite)), "DRY_RUN": str(int(plan.dry_run))}
            subprocess.run(evaluation_command(plan, source), cwd=repo_root(), env=env, check=True)
            if not plan.dry_run:
                evaluation_metrics(plan, task, transferred=True, source=source)
        if not plan.dry_run and not plan.overwrite:
            collect(plan)
    if plan.dry_run:
        print("[dry-run] No models loaded and no outputs written.", flush=True)
    else:
        if plan.overwrite:
            collect(plan)
        filename = "cross_domain_matrix.csv" if plan.matrix else "cross_domain.csv"
        print(f"[done] {plan.output / filename}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
