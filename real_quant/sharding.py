from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def eval_run_output_dir(
    output_root: str | Path,
    *,
    num_shards: int,
    shard_id: int,
) -> Path:
    """Return the collision-free output directory for one evaluation process."""

    root = Path(output_root)
    if num_shards == 1:
        return root
    return Path(f"{root}.shards") / f"shard_{shard_id:03d}_of_{num_shards:03d}"


def select_round_robin_eval_shard(
    data: Mapping[str, Mapping[str, Any]],
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Mapping[str, Any]]:
    """Select one deterministic shard while preserving its source order."""

    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    if num_shards == 1:
        return dict(data)
    return dict(list(data.items())[shard_id::num_shards])


def interleave_round_robin_shards(
    shards: list[list[tuple[str, dict[str, Any]]]],
) -> dict[str, dict[str, Any]]:
    """Restore source order from deterministic round-robin shard items."""

    merged: dict[str, dict[str, Any]] = {}
    for local_index in range(max((len(items) for items in shards), default=0)):
        for items in shards:
            if local_index < len(items):
                sample_id, sample = items[local_index]
                merged[sample_id] = sample
    return merged
