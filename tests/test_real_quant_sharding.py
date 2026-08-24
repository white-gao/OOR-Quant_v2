from __future__ import annotations

import json

from real_quant.merge_eval_shards import merge_shards
from real_quant.sharding import (
    eval_run_output_dir,
    interleave_round_robin_shards,
    select_round_robin_eval_shard,
)


def test_select_and_interleave_round_robin_shards() -> None:
    data = {f"sample_{index}": {"index": index} for index in range(7)}
    shards = [
        select_round_robin_eval_shard(data, num_shards=3, shard_id=shard_id)
        for shard_id in range(3)
    ]

    assert [list(shard) for shard in shards] == [
        ["sample_0", "sample_3", "sample_6"],
        ["sample_1", "sample_4"],
        ["sample_2", "sample_5"],
    ]
    restored = interleave_round_robin_shards(
        [list(shard.items()) for shard in shards]
    )
    assert list(restored) == list(data)


def test_eval_run_output_dir_uses_collision_free_shard_directory() -> None:
    assert eval_run_output_dir("results/run", num_shards=1, shard_id=0).as_posix() == "results/run"
    assert (
        eval_run_output_dir("results/run", num_shards=2, shard_id=1).as_posix()
        == "results/run.shards/shard_001_of_002"
    )


def test_merge_real_quant_shards_restores_payload_and_latency(tmp_path) -> None:
    output_root = tmp_path / "run"
    model_name = "model-real-naive-w8a8"
    all_samples = {
        f"sample_{index}": {
            "prompt": f"prompt {index}",
            "generations": [f"generation {index}"],
            "ground_truth": f"target {index}",
            "latency": {
                "prompt_tokens": 10 + index,
                "generated_sequences": 32,
                "generated_tokens": 96,
                "tokenize_time": 0.01,
                "generate_time": 1.0 + index,
                "decode_time": 0.02,
                "end_to_end_time": 1.03 + index,
            },
        }
        for index in range(5)
    }
    for shard_id in range(2):
        samples = dict(list(all_samples.items())[shard_id::2])
        config = {
            "backend": "test-real-fp8",
            "eval_num_shards": 2,
            "eval_shard_id": shard_id,
            "eval_shard_strategy": "round_robin",
            "eval_unsharded_sample_count": len(all_samples),
            "eval_shard_sample_count": len(samples),
            "eval_merged": False,
        }
        result_dir = (
            tmp_path
            / "run.shards"
            / f"shard_{shard_id:03d}_of_002"
            / model_name
            / "ad"
        )
        result_dir.mkdir(parents=True)
        payload = {
            "model_name": model_name,
            "task_name": "ad",
            "split": "test",
            "total_time": sum(
                sample["latency"]["end_to_end_time"] for sample in samples.values()
            ),
            "quant_config": config,
            "samples": samples,
            "hardware_info": {"device": "test"},
            "num_params": 123.0,
        }
        (result_dir / "test_generated.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (result_dir / "hf_naive_w8a8_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

    output_file, merged_model_name = merge_shards(
        output_dir=str(output_root),
        task="ad",
        split="test",
        num_shards=2,
        overwrite=False,
    )

    merged = json.loads(output_file.read_text(encoding="utf-8"))
    assert merged_model_name == model_name
    assert list(merged["samples"]) == list(all_samples)
    assert merged["latency"]["num_samples"] == len(all_samples)
    assert merged["quant_config"]["eval_merged"] is True
    assert merged["quant_config"]["eval_shard_id"] is None
    assert merged["hardware_info"] == {"device": "test"}
    assert merged["num_params"] == 123.0
