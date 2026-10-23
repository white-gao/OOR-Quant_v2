import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from metrics import compare_logits, concentration, path_overlap, bootstrap_mean
from beam_trace import trace_beams
import run_probe as probe


def tiny_model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=256, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()


def test_metrics_identity_ties_and_flips():
    t = torch.arange(128).float() / 10
    assert compare_logits(t, t)["kl"] == 0
    assert compare_logits(t, t)["top32_retention"] == 1
    flipped = compare_logits(t, -t)
    assert flipped["top32_retention"] == 0
    assert sum(b["flips"] for b in flipped["boundary_bins"]) == 32 * 32
    tied = compare_logits(torch.zeros(128), torch.zeros(128))
    assert tied["teacher_zero_gap_pairs"] == 1024
    assert sum(b["count"] for b in tied["boundary_bins"]) == 0
    assert tied["tie_aware_top32_retention"] == 1
    with pytest.raises(ValueError):
        concentration(torch.tensor([float("nan"), 1]))


def test_uniform_concentration_and_bootstrap():
    result = concentration(torch.zeros(128))
    assert result["mass"]["32"] == pytest.approx(.25)
    assert result["entropy_nats"] == pytest.approx(np.log(128))
    assert result["normalized_entropy"] == pytest.approx(1)
    assert bootstrap_mean([1, 1, 1], repeats=20)["low"] == 1
    assert bootstrap_mean([], repeats=20)["n"] == 0
    assert path_overlap([[1], [2]], [[2], [3]]) == .5
    assert path_overlap([], [[1]]) is None


@torch.inference_mode()
def test_beam_trace_leaves_generation_unchanged():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5]])
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3,
                  num_beams=32, num_return_sequences=32, do_sample=False)
    expected = model.generate(**kwargs)
    with trace_beams(model, 3) as trace:
        actual = model.generate(**kwargs)
    assert torch.equal(expected, actual)
    assert set(trace["live"]) == {"1", "2"}
    assert set(trace["expanded"]) == {"1", "2", "3"}
    assert len(trace["live"]["1"]["paths"]) == 32
    assert all(len(p) == 2 for p in trace["live"]["2"]["paths"])
    assert "_get_top_k_continuations" not in model.__dict__
    with pytest.raises(RuntimeError):
        with trace_beams(model, 3):
            raise RuntimeError("interrupt")
    assert "_get_running_beams_for_next_iteration" not in model.__dict__


@torch.inference_mode()
def test_selective_logits_match_full_forward():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    full = model(ids, use_cache=False).logits
    last = model(ids, use_cache=False, logits_to_keep=3).logits
    selected = model(ids, use_cache=False, logits_to_keep=torch.tensor([0, 3])).logits
    torch.testing.assert_close(last, full[:, -3:])
    torch.testing.assert_close(selected, full[:, [0, 3]])


def test_conditional_cache_roundtrip(tmp_path, monkeypatch):
    model = tiny_model()
    args = SimpleNamespace(device="cpu")
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    monkeypatch.setattr(probe, "build_lfq_sid_slot_batches", lambda **kw: [
        {"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
    sample = {"id": "test"}
    slots = {s: torch.arange(i*64, (i+1)*64) for i, s in enumerate(probe.SLOTS)}
    teacher = tmp_path / "teacher"
    t = probe.conditional_record(args, model, None, sample, slots, None, teacher)
    q = probe.conditional_record(args, model, None, sample, slots, teacher, tmp_path / "student")
    assert t["prompt_tokens"] == 3
    for s in probe.SLOTS:
        assert q["slots"][s]["kl"] == 0
        assert q["slots"][s]["final_block_nmse"] == 0
        assert q["slots"][s]["top32_retention"] == 1


def test_manifest_mismatch_and_output_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "HERE", tmp_path)
    path = tmp_path / "results/manifest.json"
    probe.validate_manifest(path, {"seed": 42})
    probe.validate_manifest(path, {"seed": 42})
    with pytest.raises(ValueError, match="Protocol/source files changed"):
        probe.validate_manifest(path, {"seed": 43})
    assert json.loads(path.read_text())["seed"] == 42
    with pytest.raises(ValueError):
        probe.confined(tmp_path.parent / "outside.json")


def test_text_selection_is_deterministic_and_document_disjoint(tmp_path, monkeypatch):
    path = tmp_path / "text.json"
    path.write_text(json.dumps({"data": [{"title": str(i // 2), "paragraphs": [
        {"context": f"passage-{i}"}]} for i in range(8)]}))
    args = SimpleNamespace(text_json=str(path), seed=42, samples=3)
    tokenizer = lambda *a, **kw: {"input_ids": list(range(200))}
    a, _ = probe.text_samples(args, tokenizer)
    b, _ = probe.text_samples(args, tokenizer)
    assert a == b
    assert len({r["document_sha256"] for r in a}) == 3
    assert all(len(r["positions"]) == 4 and min(r["positions"]) >= 127 for r in a)


def test_real_sample_selection_no_model_forward():
    """Optional local-data smoke test, no model load and no output/cache writes."""
    if not Path(probe.DATA).exists():
        pytest.skip("Local benchmark data not available")
    tokenizer = probe.AutoTokenizer.from_pretrained(probe.MODEL, local_files_only=True)
    args = SimpleNamespace(data_dir=probe.DATA, task="ad", seed=42, samples=2)
    samples, manifest = probe.recommendation_samples(args, tokenizer)
    batches = probe.build_lfq_sid_slot_batches(tokenizer=tokenizer, samples=samples,
                                             prompt_token="<|sid_begin|>", device=torch.device("cpu"))
    assert len(samples) == len(batches) == 2
    assert manifest["calib_excluded_users"] > 0
    assert len(set(r["uid_sha256"] for r in manifest["selection"])) == 2


@torch.inference_mode()
def test_beam_trace_with_early_eos():
    from transformers import LogitsProcessor, LogitsProcessorList

    class PreferEOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, 2] = 10
            return scores

    model = tiny_model()
    ids = torch.tensor([[3, 4, 5]])
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), eos_token_id=2,
                  max_new_tokens=3, num_beams=32, num_return_sequences=32, do_sample=False,
                  logits_processor=LogitsProcessorList([PreferEOS()]))
    expected = model.generate(**kwargs)
    with trace_beams(model, 3) as trace:
        actual = model.generate(**kwargs)
    assert torch.equal(expected, actual)
    assert trace["early_stopped_candidates"] > 0
    assert all(2 not in p for value in trace["live"].values() for p in value["paths"])


def test_main_resume_and_teacher_dependency(tmp_path, monkeypatch):
    args = SimpleNamespace(job="conditional", task="ad", samples=1, seed=42,
                           variant="bf16", model_path="unused", device="cpu",
                           run_root=str(tmp_path / "results/run"))
    monkeypatch.setattr(probe, "HERE", tmp_path)
    monkeypatch.setattr(probe, "parse_args", lambda: args)
    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(probe.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(probe, "recommendation_samples", lambda *a: ([{"id": "0"}], {}))
    monkeypatch.setattr(probe, "manifest_for", lambda *a: {"variant": args.variant, "checkpoints": []})
    monkeypatch.setattr(probe, "sid_slot_token_ids", lambda *a: [1, 2])
    calls = []
    monkeypatch.setattr(probe, "load_model", lambda *a: calls.append("load"))

    def fake_record(args, model, tokenizer, sample, slots, teacher, output):
        if teacher is None:
            (output / "cache").mkdir(exist_ok=True)
            torch.save({}, output / "cache/0.pt")
        return {"id": sample["id"], "fake_test_record": True}

    monkeypatch.setattr(probe, "conditional_record", fake_record)
    probe.main()
    probe.main()
    assert calls == ["load"]  # Completed task must not reload the model.
    args.variant = "w8a8"
    probe.main()
    assert calls == ["load", "load"]
    root = Path(args.run_root) / "ad/conditional"
    assert (root / "w8a8/COMPLETE.json").exists()
    # Incomplete teachers must not be treated as a valid completed dependency.
    (root / "bf16/COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        probe.main()


def test_plot_reader_rejects_partial_or_stale_results(tmp_path):
    from plot_motivation import read_records
    with pytest.raises(FileNotFoundError):
        read_records(tmp_path)
    manifest = {"selection": {"selection": [{"id": "0"}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"count": 1, "manifest_sha256": probe.digest(manifest)}))
    (tmp_path / "records").mkdir()
    with pytest.raises(ValueError, match="count"):
        read_records(tmp_path)
    (tmp_path / "records/0.json").write_text(json.dumps({"id": "0"}))
    assert read_records(tmp_path) == [{"id": "0"}]
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "seed": 7}))
    with pytest.raises(ValueError, match="manifest mismatch"):
        read_records(tmp_path)
