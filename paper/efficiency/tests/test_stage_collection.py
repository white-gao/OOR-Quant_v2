"""CPU contract tests; this suite never measures GPU performance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import collect_stage_speedups as collect
import stage_benchmark as bench
import plot_stage_speedups as plot


def test_workloads_cover_twelve_bars_without_duplicate_generation_case():
    cases = collect.workloads([512, 1024, 2048], [1, 2, 4], 1024)
    assert len(cases) == 5
    assert sum(len(case["panels"]) for case in cases) * 2 == 12
    assert cases[1]["panels"] == ["prefill", "decode"]


def test_controlled_inputs_exact_length_suffix_and_no_padding():
    assert collect.controlled_ids([1, 2, 3], 8, [99]) == [3, 1, 2, 3, 1, 2, 3, 99]
    assert collect.controlled_ids([1, 2, 3, 4], 3, [99]) == [3, 4, 99]
    with pytest.raises(ValueError):
        collect.controlled_ids([], 3, [99])


def test_estimator_counts_shared_groups_once_and_keeps_negative_delta():
    key = collect.operator_key("qkv", [32, 1, 2048])
    ops = {key: [{"layer": 0, "RTN": [2, 2], "RecQuant": [1, 1]},
                 {"layer": 27, "RTN": [2, 2], "RecQuant": [3, 3]}]}
    trace = [{"group": "qkv", "shape": [32, 1, 2048], "calls": 28}]
    assert collect.project_recquant(100, ops, trace) == (100, 0)
    ops[key] = ops[key][:1]
    assert collect.project_recquant(100, ops, trace) == (72, -28)
    with pytest.raises(ValueError, match="Non-positive"):
        collect.project_recquant(10, ops, trace)


@pytest.fixture
def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    config = Qwen3Config(hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=16, vocab_size=64,
                        max_position_embeddings=128, pad_token_id=0,
                        eos_token_id=None, bos_token_id=1)
    return Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()


def test_real_hf_three_step_boundary_and_shared_shape_counts(tiny_model, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    inputs = {"input_ids": torch.tensor([[3, 4, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    tok = argparse.Namespace(pad_token_id=0, eos_token_id=None)
    kwargs = bench.generation_kwargs(tok, beam=2)
    result = bench.timed_generate(tiny_model, inputs, kwargs)
    assert result["forward_shapes"] == [[2, 4], [2, 1], [2, 1]]
    assert result["total_ms"] == pytest.approx(result["prefill_ms"] + result["decode_ms"])
    trace = bench.trace_shapes(tiny_model, inputs, kwargs)
    assert len(trace) == 8
    for item in trace:
        assert item["calls"] == (1 if item["stage"] == "prefill" else 2)
    assert not tiny_model._forward_pre_hooks
    assert not tiny_model.model.layers[0].self_attn.q_proj._forward_pre_hooks


def test_learned_weight_export_and_shared_group_quantization(tiny_model, tmp_path, monkeypatch):
    import copy
    import real_quant.naive_w8a8.modules as fp8
    from flat_quant.flatquant.runtime import FlatQuantCoreConfig, _TrainableFlatQuantBlock, _flat_wrappers
    config = FlatQuantCoreConfig(weight_quant_format="fp8_e4m3fn", transform_init="identity")
    block = tiny_model.model.layers[0]
    train = _TrainableFlatQuantBlock(copy.deepcopy(block), config=config, act_quant_mode="shared_input")
    with torch.no_grad():
        for wrapper in _flat_wrappers(train.block).values():
            wrapper.upper_clip_logits.fill_(.8)
            wrapper.lower_clip_logits.fill_(-.3)
    path = tmp_path / "layer_00.pt"
    torch.save({"method": "flatquant_core", "checkpoint_schema": 2, "layer_idx": 0,
                "config": asdict(config), "act_quant_mode": "shared_input",
                "transform_state_dict": train.transforms.state_dict(),
                "activation_quantizer_state_dict": train.activation_quantizers.state_dict(),
                "lwc_parameters": {name: wrapper.lwc_state() for name, wrapper in _flat_wrappers(train.block).items()}}, path)
    with torch.inference_mode():
        groups = bench.build_probes(block, path, 0)
    counts = []

    def cpu_quant(x):
        counts.append(1)
        scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

    monkeypatch.setattr(fp8, "_vllm_scaled_fp8_quant", cpu_quant)
    with torch.inference_mode():
        for name, (_, rec) in groups.items():
            before = len(counts)
            x = torch.randn(2, 3, rec.linears[0].in_features, dtype=torch.bfloat16)
            output = rec(x)
            assert len(counts) - before == 1
            assert len(output) == len(bench.GROUPS[name])
            assert all(torch.isfinite(value).all() for value in output)
    assert all(isinstance(module, fp8.RealFP8Linear) for pair in groups.values() for group in pair for module in group.linears)


def collected_fixture():
    cases = collect.workloads([512, 1024, 2048], [1, 2, 4], 1024)
    bf16, rtn, operators = {}, {}, {}
    for case in cases:
        shape = [32 * case["batch"], 1, 2048]
        key = collect.operator_key("qkv", shape)
        operators[key] = [{"layer": 0, "RTN": [.1, .1], "RecQuant": [.2, .2]}]
        trace = [{"stage": stage, "group": "qkv", "shape": shape, "calls": 28}
                 for stage in case["panels"]]
        for destination, scale in ((bf16, 1.0), (rtn, .8)):
            records = [{"sample": sample, "repeat": repeat, "prefill_ms": (20 + sample) * scale,
                        "decode_ms": (10 + sample) * scale, "total_ms": (30 + 2 * sample) * scale,
                        "forward_shapes": [[32, 512], [32, 1], [32, 1]]}
                       for sample in range(2) for repeat in range(2)]
            destination[case["id"]] = {"records": records, "trace": trace, "unsplit_total_ms": [30 * scale]}
    args = argparse.Namespace(seed=42, bootstrap=30, smoke=False, prefill_lengths=[512, 1024, 2048],
                              decode_batches=[1, 2, 4], decode_length=1024)
    return args, cases, bf16, rtn, operators


def test_summary_and_plot_keep_measured_vs_estimated_distinct(tmp_path):
    summary = collect.summarize(*collected_fixture())
    assert len(summary["rows"]) == 12
    for row in summary["rows"]:
        if row["method"] == "RTN":
            assert row["data_kind"] == "measured_stage"
            assert row["speedup"] == pytest.approx(1.25)
        else:
            assert row["data_kind"] == "operator_projected_stage"
            assert row["added_operator_ms"] == pytest.approx(2.8)
    source = tmp_path / "summary.json"
    collect.atomic_json(source, summary)
    plot.render(source)
    assert (tmp_path / "stage_speedups_estimated.pdf").is_file()
    assert "operator-based estimates" in (tmp_path / "stage_speedups_estimated.caption.txt").read_text()
    summary["rows"][0]["data_kind"] = "illustrative"
    with pytest.raises(ValueError, match="mislabeled"):
        plot.figure_data(summary)


def test_plot_rejects_mockup():
    with pytest.raises(ValueError, match="not illustrative"):
        plot.figure_data({"data_kind": "illustrative"})
