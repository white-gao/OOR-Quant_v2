from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from real_quant.full_precision import run_recif_baseline as runner
from real_quant.full_precision.recif_tasks import (
    atomic_json, classification_token_ids, classify_batch, clean_caption, evaluate_local,
)
from real_quant.full_precision.recif_wip import score_wip
from benchmark.tasks.v1_0.registry import get_loader, get_task_config


SID = "<s_a_1><s_b_2><s_c_3>"


class Tokenizer:
    padding_side = "left"
    eos_token = "<|im_end|>"
    pad_token = "<|pad|>"
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return {"是": [1], "否": [2]}[text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return " ".join(m["content"] for m in messages)

    def __call__(self, prompts, **kwargs):
        # Deliberately different lengths; pad is on the left.
        tokens = [[3, 1] if "positive" in text else [2] for text in prompts]
        ids = torch.tensor([[0] * (2 - len(t)) + t for t in tokens])
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


class Model:
    def __call__(self, input_ids, attention_mask, **kwargs):
        assert kwargs == {"use_cache": False, "logits_to_keep": 1}
        assert attention_mask[:, -1].all()
        logits = torch.full((len(input_ids), 1, 4), -1000.0)
        logits[:, 0, 1] = torch.where(input_ids[:, -1] == 1, 3.0, -2.0)
        logits[:, 0, 2] = 0
        return SimpleNamespace(logits=logits)


def test_pair_normalized_classification_and_padding():
    tokenizer = Tokenizer()
    values = classify_batch(Model(), tokenizer, ["positive history", "negative"], "cpu",
                            classification_token_ids(tokenizer))
    probs = [json.loads(value[0]) for value in values]
    assert probs[0]["是"] == pytest.approx(torch.sigmoid(torch.tensor(3.)).item())
    assert probs[1]["是"] == pytest.approx(torch.sigmoid(torch.tensor(-2.)).item())
    tokenizer.padding_side = "right"
    with pytest.raises(ValueError, match="left padding"):
        classify_batch(Model(), tokenizer, ["positive"], "cpu", [1, 2])


def test_multi_token_labels_rejected():
    tokenizer = Tokenizer()
    tokenizer.encode = lambda *args, **kwargs: [1, 2]
    with pytest.raises(ValueError, match="single-token"):
        classification_token_ids(tokenizer)


def test_caption_cleanup_does_not_remove_sid():
    assert clean_caption("<think>x</think>说明<|im_end|><|pad|>", Tokenizer()) == "说明"
    assert clean_caption(SID, Tokenizer()) == SID


def test_official_auc_and_validation():
    samples = {str(i): {"ground_truth": truth, "generations": [json.dumps({"是": p, "否": 1-p})]}
               for i, (truth, p) in enumerate([("是", .9), ("否", .1), ("是", .8), ("否", .2)])}
    metrics, details = evaluate_local("label_pred", samples, "/unused")
    assert metrics["auc"] == 1
    assert len(details) == 4
    with pytest.raises(ValueError, match="one-class"):
        evaluate_local("label_pred", {"0": samples["0"]}, "/unused")
    samples["1"]["generations"] = ['{"是": -0.5, "否": 1.5}']
    with pytest.raises(ValueError, match="Invalid classification"):
        evaluate_local("label_pred", samples, "/unused")


def test_official_interactive_metrics_without_mutating_config():
    before = json.dumps(get_task_config("interactive"), sort_keys=True)
    samples = {"0": {"ground_truth": f"<|sid_begin|>{SID}<|sid_end|>", "generations": [SID] * 32}}
    metrics, details = evaluate_local("interactive", samples, "/unused")
    assert metrics["pass@1"] == metrics["recall@32"] == 1
    assert set(details) == {"0"}
    assert json.dumps(get_task_config("interactive"), sort_keys=True) == before


def make_data(root, task="label_pred", count=5):
    directory = root / task
    directory.mkdir(parents=True)
    rows = [{"messages": [{"role": "user", "content": [{"type": "text", "text":
             "positive" if i % 2 == 0 else "negative"}]}],
             "metadata": {"answer": "是" if i % 2 == 0 else "否"}} for i in range(count)]
    path = directory / f"{task}_test.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_chunk_indices_and_no_source_sampling_cache(tmp_path):
    path = make_data(tmp_path)
    chunks = list(runner.data_chunks(path, 5, 2))
    assert [start for start, _ in chunks] == [0, 2, 4]
    loader = get_loader("label_pred", str(tmp_path), Tokenizer(), enable_thinking=False)
    assert list(runner.format_chunk(loader, chunks[1][1])) == ["2", "3"]
    assert list(path.parent.iterdir()) == [path]


def test_reject_answer_leakage_and_skipped_rows(tmp_path):
    loader = get_loader("label_pred", str(tmp_path), Tokenizer(), enable_thinking=False)
    frame = pd.DataFrame([{"messages": [{"role": "assistant", "content": "答案"}],
                           "metadata": {"answer": "是"}}])
    with pytest.raises(ValueError, match="leakage"):
        runner.format_chunk(loader, frame)
    frame.at[0, "messages"] = [{"role": "user", "content": "question"}]
    frame.at[0, "metadata"] = {}
    with pytest.raises(ValueError, match="skipped"):
        runner.format_chunk(loader, frame)


@pytest.fixture
def baseline_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_data(data)
    model_path = tmp_path / "1.7B"
    model_path.mkdir()
    atomic_json(model_path / "config.json", {"model_type": "qwen3"})
    counter = {"loads": 0}

    def load_model(*args, **kwargs):
        counter["loads"] += 1
        return SimpleNamespace(model=Model(), tokenizer=Tokenizer(), device="cpu")

    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(runner.HFFullPrecisionGenerator, "from_pretrained", load_model)
    args = ["--task", "label_pred", "--model_path", str(model_path), "--data_dir", str(data),
            "--output_dir", str(tmp_path / "output"), "--device", "cpu", "--dtype", "float32",
            "--chunk_size", "2", "--batch_size", "2"]
    directory = tmp_path / "output/1.7B/label_pred"
    return args, directory, counter


def test_end_to_end_baseline_resume_and_score_only(baseline_env):
    args, directory, counter = baseline_env
    runner.main(args)
    metrics = json.loads((directory / "eval_results.json").read_text())
    assert metrics["metrics"]["auc"] == 1
    assert metrics["status"] == "complete"
    payload = json.loads((directory / "test_generated.json").read_text())
    assert len(payload["samples"]) == 5
    assert all("prompt" not in v for v in payload["samples"].values())
    runner.main(args)
    assert counter["loads"] == 1
    # Rebuild missing final JSON entirely from checkpoints, without model loading.
    (directory / "test_generated.json").unlink()
    runner.main(args)
    assert counter["loads"] == 1
    runner.main(args + ["--stage", "evaluate"])
    assert counter["loads"] == 1


def test_resume_protocol_change_refused(baseline_env):
    args, directory, counter = baseline_env
    runner.main(args + ["--stage", "generate"])
    with pytest.raises(ValueError, match="Configuration changed"):
        runner.main(args + ["--seed", "43"])
    assert counter["loads"] == 1
    assert not (directory / "eval_results.json").exists()


def test_preflight_does_not_write_or_load_weights(baseline_env):
    args, directory, counter = baseline_env
    runner.main(args + ["--check_only"])
    assert counter["loads"] == 0
    assert not directory.exists()


def test_invalid_part_ids_rejected(tmp_path):
    path = tmp_path / "part.json"
    atomic_json(path, {"run_id": "run", "samples": {"0": {}}})
    with pytest.raises(ValueError, match="mismatch"):
        runner.read_part(path, "run", ["0", "1"])


@pytest.mark.parametrize("task", ["interactive", "item_understand"])
def test_generation_branches_and_pending_judge(tmp_path, monkeypatch, task):
    data = tmp_path / "data"
    path = make_data(data, task=task, count=2)
    frame = pd.read_parquet(path)
    answer = f"<|sid_begin|>{SID}<|sid_end|>" if task == "interactive" else "a cat"
    frame["metadata"] = [{"answer": answer}] * 2
    frame.to_parquet(path, index=False)
    model_path = tmp_path / "1.7B"
    model_path.mkdir()
    atomic_json(model_path / "config.json", {"model_type": "qwen3"})

    class Generator:
        tokenizer = Tokenizer()
        device = "cpu"

        def generate_batch(self, batch, generation_kwargs):
            interactive = task == "interactive"
            assert generation_kwargs["num_beams"] == (32 if interactive else 1)
            assert generation_kwargs["max_new_tokens"] == (3 if interactive else 128)
            assert generation_kwargs["do_sample"] is False
            for _, prompt in batch:
                assert prompt.endswith("<|sid_begin|>") == interactive
            return {key: [SID] * 32 if interactive else ["a cat<|im_end|>"] for key, _ in batch}, {}

    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(runner.HFFullPrecisionGenerator, "from_pretrained", lambda *a, **kw: Generator())
    args = ["--task", task, "--model_path", str(model_path), "--data_dir", str(data),
            "--output_dir", str(tmp_path / "output"), "--device", "cpu", "--batch_size", "2"]
    runner.main(args)
    directory = tmp_path / "output/1.7B" / task
    assert (directory / "test_generated.json").is_file()
    if task == "interactive":
        assert json.loads((directory / "eval_results.json").read_text())["metrics"]["recall@32"] == 1
    else:
        assert not (directory / "eval_results.json").exists()
        assert json.loads((directory / "test_generated.json").read_text())["samples"]["0"]["generations"] == ["a cat"]
        with pytest.raises(ValueError, match="explicitly supply"):
            runner.main(args + ["--stage", "evaluate"])


WIP = {"info_point": "a cat", "importance_score": 5}
MATCH = {"matches": [{"gt_wip": WIP, "model_wip": WIP}],
         "unmatched_gt_wips": [], "unmatched_model_wips": []}


class Judge:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return next(self.responses)


def fake_bertscore(ids, matches, *args):
    for result in matches.values():
        for match in result["matches"]:
            match["match_quality"] = 1.0


def test_wip_official_metric_cache_and_empty_coverage(tmp_path):
    samples = {"0": {"ground_truth": "cat", "generations": ["a cat"]},
               "1": {"ground_truth": "empty", "generations": [""]}}
    judge = Judge([json.dumps({"wips": [WIP]}), json.dumps({"wips": [WIP]}),
                   json.dumps(MATCH), '{"wips": []}'])
    metrics, details = score_wip(samples, tmp_path, {}, "fake", client=judge, bertscore_fn=fake_bertscore)
    assert metrics["macro_wip_double_weighted_f1"] == 1
    assert metrics["wip_coverage"] == .5
    assert details["1"]["status"] == "excluded_empty_wips"
    assert judge.calls == 4
    metrics2, _ = score_wip(samples, tmp_path, {}, "fake", client=judge, bertscore_fn=fake_bertscore)
    assert metrics == metrics2
    assert judge.calls == 4


def test_wip_api_failure_is_not_a_zero_score(tmp_path):
    samples = {"0": {"ground_truth": "cat", "generations": ["cat"]}}
    with pytest.raises(RuntimeError, match="extraction failed"):
        score_wip(samples, tmp_path, {}, "fake", client=Judge(["invalid JSON"]), bertscore_fn=fake_bertscore)


def test_wip_reference_is_shared_across_model_outputs(tmp_path):
    references = tmp_path / "shared_references"
    samples = {"0": {"ground_truth": "cat", "generations": ["caption one"]}}
    first = Judge([json.dumps({"wips": [WIP]}), json.dumps({"wips": [WIP]}), json.dumps(MATCH)])
    score_wip(samples, tmp_path / "baseline", {}, "fake", client=first,
              bertscore_fn=fake_bertscore, reference_cache_dir=references)
    second = Judge([json.dumps({"wips": [WIP]}), json.dumps(MATCH)])
    samples["0"]["generations"] = ["caption two"]
    score_wip(samples, tmp_path / "comparison", {}, "fake", client=second,
              bertscore_fn=fake_bertscore, reference_cache_dir=references)
    assert first.calls == 3
    assert second.calls == 2


def test_wip_bertscore_failure_is_not_silently_skipped(tmp_path):
    samples = {"0": {"ground_truth": "cat", "generations": ["cat"]}}
    judge = Judge([json.dumps({"wips": [WIP]}), json.dumps({"wips": [WIP]}), json.dumps(MATCH)])
    with pytest.raises(RuntimeError, match="Missing/non-finite BERTScore"):
        score_wip(samples, tmp_path, {}, "fake", client=judge, bertscore_fn=lambda *a: None)


def test_qwen3_next_token_logits_matches_full_forward():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[0, 1, 3], [2, 4, 5]])
    mask = ids.ne(0).long()
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1]
        last = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits[:, -1]
    torch.testing.assert_close(last, full)


def test_real_hf_beam_generation_on_tiny_cpu_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    model = Qwen3ForCausalLM(config).eval()
    tokenizer = Tokenizer()
    tokenizer.decode = lambda ids, **kwargs: str(ids.tolist())
    generator = runner.HFFullPrecisionGenerator(model=model, tokenizer=tokenizer,
                                                model_name="tiny-qwen3", device="cpu")
    kwargs = runner.build_hf_generation_kwargs(max_new_tokens=3, num_beams=32, num_return_sequences=32,
                                               eos_token_id=0, pad_token_id=0)
    outputs, records = generator.generate_batch(batch=[("0", "positive"), ("1", "negative")],
                                                generation_kwargs=kwargs)
    assert len(outputs["0"]) == len(outputs["1"]) == 32
    assert records["0"].prompt_tokens == 2
    assert records["1"].prompt_tokens == 1
