from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class LatencyRecord:
    sample_id: str
    prompt_tokens: int
    generated_sequences: int
    generated_tokens: int
    tokenize_time: float
    generate_time: float
    decode_time: float

    @property
    def end_to_end_time(self) -> float:
        return self.tokenize_time + self.generate_time + self.decode_time

    def to_sample_fields(self) -> dict[str, list[float] | list[int]]:
        return {
            "input_tokens": [int(self.prompt_tokens)],
            "output_tokens": [int(self.generated_tokens)],
            "times": [float(self.generate_time)],
        }

    def to_latency_fields(self) -> dict[str, float | int]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "generated_sequences": int(self.generated_sequences),
            "generated_tokens": int(self.generated_tokens),
            "tokenize_time": float(self.tokenize_time),
            "generate_time": float(self.generate_time),
            "decode_time": float(self.decode_time),
            "end_to_end_time": float(self.end_to_end_time),
        }


def cuda_synchronize_if_needed(device: torch.device | str | None) -> None:
    if device is None:
        return
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device_obj)


def _round(value: float) -> float:
    return round(float(value), 12)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return _round(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return _round(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize_values(values: Iterable[float], prefix: str) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            f"{prefix}_total": 0.0,
            f"{prefix}_avg": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p99": 0.0,
        }
    sorted_vals = sorted(vals)
    return {
        f"{prefix}_total": _round(sum(vals)),
        f"{prefix}_avg": _round(statistics.fmean(vals)),
        f"{prefix}_p50": _percentile(sorted_vals, 0.50),
        f"{prefix}_p90": _percentile(sorted_vals, 0.90),
        f"{prefix}_p99": _percentile(sorted_vals, 0.99),
    }


def aggregate_latency(records: Iterable[LatencyRecord]) -> dict[str, float | int]:
    recs = list(records)
    summary: dict[str, float | int] = {
        "num_samples": len(recs),
        "total_prompt_tokens": sum(r.prompt_tokens for r in recs),
        "total_generated_sequences": sum(r.generated_sequences for r in recs),
        "total_generated_tokens": sum(r.generated_tokens for r in recs),
    }
    summary.update(summarize_values((r.tokenize_time for r in recs), "tokenize_time"))
    summary.update(summarize_values((r.generate_time for r in recs), "generate_time"))
    summary.update(summarize_values((r.decode_time for r in recs), "decode_time"))
    summary.update(summarize_values((r.end_to_end_time for r in recs), "end_to_end_time"))
    generate_total = float(summary["generate_time_total"])
    end_to_end_total = float(summary["end_to_end_time_total"])
    summary["generated_tokens_per_generate_second"] = (
        _round(float(summary["total_generated_tokens"]) / generate_total) if generate_total > 0 else 0.0
    )
    summary["samples_per_end_to_end_second"] = (
        _round(float(summary["num_samples"]) / end_to_end_total) if end_to_end_total > 0 else 0.0
    )
    return summary

