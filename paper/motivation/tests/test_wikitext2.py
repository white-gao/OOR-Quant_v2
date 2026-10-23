"""No-GPU tests for article sampling, causal positions, resume, and plot provenance."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import probe_wikitext2 as probe
from plot_motivation import export_text_controls, read_text_reference, summarize_text_positions


def test_article_boundaries_not_dataset_rows():
    rows = ["\n", " = Article One = \n", "", "First paragraph.\n", " = = Section = = \n",
            "More text.\n", "", " = Article Two = \n", "", "== Section ==\n", "Last paragraph.\n",
            " = 1 henry and capacitance C = \n", "1 farad.\n"]
    articles = probe.articles_from_lines(rows, expected=2)
    assert [a["title"] for a in articles] == ["Article One", "Article Two"]
    assert articles[0]["source_rows"] == [1, 7]
    assert " = = Section = = " in articles[0]["text"]
    assert " = 1 henry and capacitance C = " in articles[1]["text"]
    assert len({a["id"] for a in articles}) == 2
    with pytest.raises(ValueError, match="Expected 60"):
        probe.articles_from_lines(rows)
    with pytest.raises(ValueError, match="before the first"):
        probe.articles_from_lines(["orphan text\n"] + rows, expected=2)


def test_selection_is_uniform_deterministic_and_article_clustered():
    articles = probe.articles_from_lines([" = One = \n", "", "abc "*1000+"\n", "",
                                         " = Two = \n", "", "xyz "*1000+"\n"], expected=2)
    def tokenizer(text, **kwargs):
        return {"input_ids": [i % 200 for i in range(len(text))],
                "offset_mapping": [(i, i+1) for i in range(len(text))]}
    args = SimpleNamespace(seed=42, max_articles=0, min_context=128, positions_per_article=32, length_positions=8)
    selected, excluded = probe.select_articles(articles, tokenizer, args)
    assert not excluded
    assert selected == probe.select_articles(articles, tokenizer, args)[0]
    assert len(selected) == 2
    for sample in selected:
        assert len(sample["positions"]) == len(set(sample["positions"])) == 32
        assert all(127 <= p < len(sample["input_ids"])-1 for p in sample["positions"])
        assert set(sample["length_positions"]) <= set(sample["positions"])
        assert all(p >= 1023 for p in sample["length_positions"])
        assert len(sample["length_positions"]) <= 8
    args.max_articles = 1
    assert probe.select_articles(articles, tokenizer, args)[0][0] == selected[0]


def toy_sample():
    return {"id": "sample", "document_sha256": "sample", "title": "test", "source_rows": [0, 2],
            "input_ids": [3+i % 100 for i in range(1030)], "positions": [127, 1023],
            "length_positions": [1023], "target_char_spans": {"127": [128, 129], "1023": [1024, 1025]}}


def test_exact_next_token_index_and_causal_sanity():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()
    sample = toy_sample()
    result = probe.evaluate_position(model, sample, 127, 1024, "cpu", torch.tensor([200, 201]))
    assert result["next_token_id"] == sample["input_ids"][128]
    assert result["prefix_tokens"] == 128
    assert result["prefix_start"] == 0
    assert len(result["top128_token_ids"]) == 128
    assert sum(result["top128_probabilities"][:32]) == pytest.approx(result["full_vocab"]["mass"]["32"], abs=1e-6)
    assert probe.causal_sanity(model, sample, "cpu")["kl"] < 1e-6


def test_resume_inside_article_and_skip_completed_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.common, "HERE", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    sample = toy_sample()
    manifest = {"selection": {"selection": [sample]}}
    (output / "manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(device="cpu", max_context=1024)
    loads, calls = [], []
    monkeypatch.setattr(probe.common, "load_model", lambda args: loads.append("load") or object())
    monkeypatch.setattr(probe.common, "sid_slot_token_ids", lambda *args: (200, 201))
    monkeypatch.setattr(probe, "causal_sanity", lambda *args: {"kl": 0., "top32_mass_abs_delta": 0.})
    fail_once = [True]
    def fake_eval(model, sample, position, limit, *unused):
        calls.append((position, limit))
        if len(calls) == 2 and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("interrupted")
        return {"position": position, "prefix_tokens": min(position+1, limit),
                "next_token_id": sample["input_ids"][position+1],
                **probe.position_statistics(torch.arange(256).float()/20, 5, torch.tensor([200, 201]))}
    monkeypatch.setattr(probe, "evaluate_position", fake_eval)
    with pytest.raises(RuntimeError, match="interrupted"):
        probe.collect(args, [sample], manifest, output, None)
    assert not (output / "COMPLETE.json").exists()
    assert len(json.loads((output / "records/sample.json").read_text())["positions"]) == 1
    probe.collect(args, [sample], manifest, output, None)
    probe.collect(args, [sample], manifest, output, None)
    assert loads == ["load", "load"]
    assert calls.count((127, 1024)) == 1
    assert calls.count((1023, 1024)) == 2  # One failed attempt, one successful primary; control reused.
    record = json.loads((output / "records/sample.json").read_text())
    assert len(record["positions"]) == 2 and len(record["length_control"]) == 3
    assert record["length_control"][-1]["reused_primary"] is True
    bad = deepcopy(record)
    bad["positions"].append(bad["positions"][0])
    (output / "records/sample.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="duplicate"):
        probe.collect(args, [sample], manifest, output, None)


def test_macro_statistics_not_position_weighted():
    def position(mass, length=1024):
        return {"prefix_tokens": length, "full_vocab": {"mass": {str(k): mass for k in probe.concentration(torch.zeros(128))["mass"]},
                                                        "entropy_nats": 2.}}
    rows = [{"positions": [position(.2)]}, {"positions": [position(.8) for _ in range(9)]}]
    summary = summarize_text_positions(rows, 20)
    assert summary["mass"]["32"]["mean"] == pytest.approx(.5)  # Not .74.
    assert summary["mass"]["32"]["n"] == 2
    assert summary["positions"] == 10
    assert summary["target_nll"]["mean"] is None
    empty = summarize_text_positions(rows, 20, predicate=lambda p: p["prefix_tokens"] < 128)
    assert empty["documents"] == 0 and empty["mass"]["32"]["mean"] is None


def test_text_source_label_is_from_verified_manifest(tmp_path):
    meta = {"job": "text", "task": "wikitext2", "variant": "bf16", "model": ["model"],
            "selection": {"selection": [{"id": "article"}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"count": 1, "manifest_sha256": probe.common.digest(meta)}))
    (tmp_path / "records").mkdir()
    (tmp_path / "records/article.json").write_text(json.dumps({"id": "article", "document_sha256": "article", "positions": [{}]}))
    _, ref = read_text_reference(tmp_path, ["model"])
    assert ref["label"] == "WikiText-2"
    with pytest.raises(ValueError, match="same BF16"):
        read_text_reference(tmp_path, ["other-model"])


def test_launcher_isolated_outputs_and_no_legacy_probe_execution():
    import os
    result = subprocess.run(["bash", str(SCRIPTS / "run_wikitext2_reference.sh")],
                            env={**os.environ, "DRY_RUN": "1", "GPU": "5"}, capture_output=True, text=True, check=True)
    assert "probe_wikitext2.py preflight" in result.stdout
    assert "probe_wikitext2.py run" in result.stdout
    assert "run_probe.py" not in result.stdout
    assert "v2_wikitext2_seed42" in result.stdout
    assert "--extra-text-root" in result.stdout


def test_supplement_renders_all_sources_and_paired_lengths(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = probe.concentration(torch.arange(256).float()/20)
    positions = [{"prefix_tokens": n, "full_vocab": stats} for n in (256, 512, 1024)]
    rows = [{"positions": positions, "length_control": positions}]
    references = [(rows, {"task": "wikitext2", "label": "WikiText-2"}),
                  ([{"positions": positions}], {"task": "cmrc2018", "label": "CMRC2018 (Chinese)"})]
    teacher = [{"prompt_tokens": 1024, "slots": {s: {"full_vocab": stats} for s in "abc"}}]
    export_text_controls(references, teacher, tmp_path, "ad", 20, plt)
    summary = json.loads((tmp_path / "text_controls_ad.json").read_text())
    assert summary["references"]["wikitext2"]["paired_length_control"]["256"]["positions"] == 1
    assert summary["references"]["cmrc2018"]["paired_length_control"] == {}
    assert summary["sid_length_bins"]["a"]["512-1024"]["n"] == 1
    assert summary["sid_length_bins"]["b"][">1024"]["n"] == 1
    assert (tmp_path / "text_controls_ad.pdf").stat().st_size > 1000
    assert (tmp_path / "text_controls_ad.png").stat().st_size > 1000
    plt.close("all")
