from __future__ import annotations

import json

from flat_quant.merge_eval_shards import merge_shards


def test_merge_ignores_shard_local_flat_checkpoint_dir(tmp_path) -> None:
    output_root = tmp_path / "run"
    model_name = "1.7B"
    all_samples = {
        f"sample_{index}": {"generations": [f"generation {index}"]}
        for index in range(5)
    }
    load_checkpoint_dir = tmp_path / "trained" / "flatquant_calibration"

    for shard_id in range(2):
        samples = dict(list(all_samples.items())[shard_id::2])
        result_dir = (
            tmp_path
            / "run.shards"
            / f"shard_{shard_id:03d}_of_002"
            / model_name
            / "ad"
        )
        result_dir.mkdir(parents=True)
        config = {
            "mode": "flatquant_core",
            "eval_num_shards": 2,
            "eval_shard_id": shard_id,
            "eval_shard_strategy": "round_robin",
            "eval_unsharded_sample_count": len(all_samples),
            "eval_shard_sample_count": len(samples),
            "eval_merged": False,
            "flat_checkpoint_dir": str(result_dir / "flatquant_calibration"),
            "flat_load_checkpoint_dir": str(load_checkpoint_dir),
        }
        payload = {
            "model_name": model_name,
            "task_name": "ad",
            "split": "test",
            "total_time": float(shard_id + 1),
            "quant_config": config,
            "samples": samples,
        }
        (result_dir / "test_generated.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (result_dir / "flatquant_core_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

    output_file, merged_model_name, sid_metrics = merge_shards(
        output_dir=str(output_root),
        task="ad",
        split="test",
        num_shards=2,
        overwrite=False,
    )

    merged = json.loads(output_file.read_text(encoding="utf-8"))
    assert merged_model_name == model_name
    assert sid_metrics is None
    assert list(merged["samples"]) == list(all_samples)
    assert merged["quant_config"]["eval_merged"] is True
    assert merged["quant_config"]["flat_checkpoint_dir"] is None
    assert merged["quant_config"]["flat_load_checkpoint_dir"] == str(
        load_checkpoint_dir
    )
