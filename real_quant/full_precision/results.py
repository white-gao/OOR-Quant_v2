from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .latency import LatencyRecord, aggregate_latency


def build_generation_payload(
    *,
    model_name: str,
    task_name: str,
    split: str,
    samples: Mapping[str, Mapping[str, Any]],
    latency_records: list[LatencyRecord],
    config: Mapping[str, Any],
    hardware_info: Mapping[str, Any] | None = None,
    num_params: float | None = None,
) -> dict[str, Any]:
    latency_by_id = {record.sample_id: record for record in latency_records}
    output_samples: dict[str, dict[str, Any]] = {}
    for sample_id, sample in samples.items():
        item = {
            "prompt": sample.get("prompt", ""),
            "generations": list(sample.get("generations", [])),
            "ground_truth": sample.get("ground_truth", ""),
        }
        if "metadata" in sample:
            item["metadata"] = sample["metadata"]
        record = latency_by_id.get(sample_id)
        if record is not None:
            item.update(record.to_sample_fields())
            item["latency"] = record.to_latency_fields()
        output_samples[sample_id] = item

    latency_summary = aggregate_latency(latency_records)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "task_name": task_name,
        "split": split,
        "total_time": latency_summary["end_to_end_time_total"],
        "avg_time_per_sample": latency_summary["end_to_end_time_avg"],
        "quant_config": dict(config),
        "latency": latency_summary,
        "samples": output_samples,
    }
    if hardware_info:
        payload["hardware_info"] = dict(hardware_info)
    if num_params is not None:
        payload["num_params"] = float(num_params)
    return payload


def save_generation_payload(payload: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

