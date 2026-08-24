#!/usr/bin/env python3
"""Merge real-quant evaluation shards and compute benchmark metrics once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark  # noqa: E402
from real_quant.full_precision.latency import LatencyRecord, aggregate_latency  # noqa: E402
from real_quant.sharding import interleave_round_robin_shards  # noqa: E402


TASK_CHOICES = ("ad", "product", "video", "label_cond", "interactive")
DYNAMIC_CONFIG_KEYS = frozenset(
    {
        "eval_shard_id",
        "eval_shard_sample_count",
        "eval_merged",
        "eval_shard_total_times",
        "eval_parallel_wall_time",
        "eval_aggregate_gpu_time",
    }
)


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path).expanduser()
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge round-robin real-quant evaluation shards and evaluate once."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--task", choices=TASK_CHOICES, default="ad")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute benchmark metrics after merging (default: enabled).",
    )
    return parser.parse_args()


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read shard payload {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Shard payload must be a JSON object: {path}")
    if not isinstance(payload.get("samples"), dict):
        raise TypeError(f"Shard payload has no samples object: {path}")
    if not isinstance(payload.get("quant_config"), dict):
        raise TypeError(f"Shard payload has no quant_config object: {path}")
    return payload


def _find_shard_file(
    *,
    shard_root: Path,
    shard_id: int,
    num_shards: int,
    task: str,
    split: str,
) -> Path:
    shard_dir = shard_root / f"shard_{shard_id:03d}_of_{num_shards:03d}"
    candidates = sorted(shard_dir.glob(f"*/{task}/{split}_generated.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {split}_generated.json below {shard_dir} for "
            f"task={task}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in DYNAMIC_CONFIG_KEYS}


def _latency_record(sample_id: str, sample: Mapping[str, Any]) -> LatencyRecord:
    values = sample.get("latency")
    if not isinstance(values, Mapping):
        raise TypeError(f"Sample {sample_id!r} has no latency object")
    try:
        return LatencyRecord(
            sample_id=sample_id,
            prompt_tokens=int(values["prompt_tokens"]),
            generated_sequences=int(values["generated_sequences"]),
            generated_tokens=int(values["generated_tokens"]),
            tokenize_time=float(values["tokenize_time"]),
            generate_time=float(values["generate_time"]),
            decode_time=float(values["decode_time"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid latency object for sample {sample_id!r}") from exc


def merge_shards(
    *,
    output_dir: str,
    task: str,
    split: str,
    num_shards: int,
    overwrite: bool,
) -> tuple[Path, str]:
    if num_shards <= 1:
        raise ValueError(f"--num_shards must be greater than one, got {num_shards}")

    output_root = resolve_repo_path(output_dir)
    shard_root = Path(f"{output_root}.shards")
    shard_files = [
        _find_shard_file(
            shard_root=shard_root,
            shard_id=shard_id,
            num_shards=num_shards,
            task=task,
            split=split,
        )
        for shard_id in range(num_shards)
    ]
    payloads = [_read_payload(path) for path in shard_files]

    first = payloads[0]
    model_name = str(first.get("model_name", ""))
    if not model_name:
        raise ValueError(f"Missing model_name in {shard_files[0]}")
    reference_config = _normalized_config(first["quant_config"])
    expected_total = int(first["quant_config"].get("eval_unsharded_sample_count", -1))
    if expected_total <= 0:
        raise ValueError("Shard config is missing a positive eval_unsharded_sample_count.")

    shard_items: list[list[tuple[str, dict[str, Any]]]] = []
    shard_times: list[float] = []
    seen_sample_ids: set[str] = set()
    for shard_id, (path, payload) in enumerate(zip(shard_files, payloads)):
        config = payload["quant_config"]
        if payload.get("model_name") != model_name:
            raise ValueError(f"model_name mismatch in {path}")
        if payload.get("task_name") != task or payload.get("split") != split:
            raise ValueError(f"task/split mismatch in {path}")
        if int(config.get("eval_num_shards", -1)) != num_shards:
            raise ValueError(f"eval_num_shards mismatch in {path}")
        if int(config.get("eval_shard_id", -1)) != shard_id:
            raise ValueError(f"eval_shard_id mismatch in {path}")
        if config.get("eval_shard_strategy") != "round_robin":
            raise ValueError(f"Unsupported eval_shard_strategy in {path}")
        if int(config.get("eval_unsharded_sample_count", -1)) != expected_total:
            raise ValueError(f"eval_unsharded_sample_count mismatch in {path}")
        if _normalized_config(config) != reference_config:
            raise ValueError(f"Quantization/evaluation config mismatch in {path}")

        items = list(payload["samples"].items())
        declared_count = int(config.get("eval_shard_sample_count", -1))
        if declared_count != len(items):
            raise ValueError(
                f"eval_shard_sample_count={declared_count} but found {len(items)} samples in {path}"
            )
        for sample_id, _sample in items:
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id across shards: {sample_id}")
            seen_sample_ids.add(sample_id)
        shard_items.append(items)
        shard_times.append(float(payload.get("total_time", 0.0)))

    merged_samples = interleave_round_robin_shards(shard_items)
    if len(merged_samples) != expected_total:
        raise ValueError(
            f"Merged {len(merged_samples)} samples, expected {expected_total}; "
            "one or more shards are missing or contain the wrong selection."
        )

    latency_records = [
        _latency_record(sample_id, sample)
        for sample_id, sample in merged_samples.items()
    ]
    latency_summary = aggregate_latency(latency_records)
    parallel_wall_time = max(shard_times)
    aggregate_gpu_time = sum(shard_times)
    merged_config = dict(first["quant_config"])
    merged_config.update(
        {
            "eval_shard_id": None,
            "eval_shard_sample_count": len(merged_samples),
            "eval_merged": True,
            "eval_shard_total_times": shard_times,
            "eval_parallel_wall_time": parallel_wall_time,
            "eval_aggregate_gpu_time": aggregate_gpu_time,
        }
    )

    output_file = output_root / model_name / task / f"{split}_generated.json"
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Merged generation file exists: {output_file}. Use --overwrite.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_payload: dict[str, Any] = {
        "model_name": model_name,
        "task_name": task,
        "split": split,
        # Keep the single-GPU definition: total_time is summed per-sample
        # latency, while parallel_wall_time records multi-GPU elapsed work.
        "total_time": latency_summary["end_to_end_time_total"],
        "avg_time_per_sample": latency_summary["end_to_end_time_avg"],
        "parallel_wall_time": parallel_wall_time,
        "aggregate_gpu_time": aggregate_gpu_time,
        "shard_total_times": shard_times,
        "quant_config": merged_config,
        "latency": latency_summary,
        "samples": merged_samples,
    }
    if isinstance(first.get("hardware_info"), Mapping):
        merged_payload["hardware_info"] = dict(first["hardware_info"])
    if first.get("num_params") is not None:
        merged_payload["num_params"] = float(first["num_params"])

    temporary_file = output_file.with_name(f".{output_file.name}.tmp")
    temporary_file.write_text(
        json.dumps(merged_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_file, output_file)

    config_files = sorted(shard_files[0].parent.glob("*_config.json"))
    if len(config_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_config.json beside {shard_files[0]}, found {config_files}"
        )
    (output_file.parent / config_files[0].name).write_text(
        json.dumps(merged_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_file, model_name


def main() -> None:
    args = parse_args()
    output_file, _model_name = merge_shards(
        output_dir=args.output_dir,
        task=args.task,
        split=args.split,
        num_shards=args.num_shards,
        overwrite=args.overwrite,
    )
    print(f"[real shard merge] merged_file={output_file}")

    if args.evaluate:
        output_root = resolve_repo_path(args.output_dir)
        Benchmark.evaluate_dev(
            generation_results_dir=str(output_root),
            output_path=str(output_root / "eval_results.json"),
            data_dir=str(resolve_repo_path(args.data_dir)),
            overwrite=args.overwrite,
            task_types=[args.task],
        )
        print(f"[real shard merge] metrics={output_root / 'eval_results.json'}")


if __name__ == "__main__":
    main()
