"""Task adapters for the retained OpenOneRec evaluators (no quantization)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch

from shared.paths import repo_root

sys.path.insert(0, str(repo_root() / "benchmarks"))
from benchmark.tasks.v1_0.registry import get_evaluator, get_task_config  # noqa: E402

TASKS = ("interactive", "label_pred", "item_understand")
UPSTREAM_COMMIT = "a969edcadd579a06c1966ae1db5984e02f48beff"
DATASET_REVISION = "8f7cf2ee0b949e955a87a708d02024687be232c8"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    """A failed write must not masquerade as a complete result/checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)  # Only our own incomplete temporary file.


def classification_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in ("是", "否")]
    if any(len(tokens) != 1 for tokens in ids) or ids[0] == ids[1]:
        raise ValueError(f"label_pred requires distinct single-token 是/否 labels, got {ids}")
    return [tokens[0] for tokens in ids]


def classify_batch(model, tokenizer, prompts: list[str], device, token_ids: list[int]):
    """Official pair-normalized next-token probabilities, not sampled labels.

    logits_to_keep=1 is supported by the OneRec Qwen3 checkpoints; it avoids
    materializing [batch, prompt_length, vocabulary] logits for long histories.
    The final non-padding token is at -1 because we enforce left padding.
    """
    if tokenizer.padding_side != "left":
        raise ValueError("Classification requires left padding.")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        probs = logits[:, token_ids].float().softmax(dim=-1)
    if not torch.isfinite(probs).all():
        raise ValueError("Non-finite label probabilities.")
    return [[json.dumps({"是": yes, "否": no}, ensure_ascii=False)]
            for yes, no in probs.cpu().tolist()]


def clean_caption(text: str, tokenizer) -> str:
    # SID tokens must never be removed from recommendation outputs. Only the
    # caption adapter removes textual EOS/PAD and optional thinking wrappers.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for token in (tokenizer.eos_token, tokenizer.pad_token):
        if token:
            text = text.replace(token, "")
    return text.strip()


def evaluate_local(task: str, samples: dict, data_dir: str):
    """Use official metric implementations, but reject silently skipped rows."""
    if not samples:
        raise ValueError("Cannot score an empty evaluation set.")
    config = copy.deepcopy(get_task_config(task))
    if task == "label_pred":
        from benchmark.tasks.v1_0.label_pred.utils import extract_label_from_answer
        labels = set()
        for key, sample in samples.items():
            label = extract_label_from_answer(sample["ground_truth"])
            if label not in (0, 1):
                raise ValueError(f"Unrecognized classification label at sample {key}.")
            labels.add(label)
            generations = sample["generations"]
            if len(generations) != 1:
                raise ValueError(f"Expected one probability pair at sample {key}.")
            scores = json.loads(generations[0])
            values = [scores[name] for name in ("是", "否")]
            if (not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                        for v in values) or not math.isclose(sum(values), 1, abs_tol=1e-5)):
                raise ValueError(f"Invalid classification probabilities at sample {key}.")
        if len(labels) != 2:
            raise ValueError("AUC is undefined on a one-class subset; use more samples.")
    elif task == "interactive":
        from benchmark.tasks.v1_0.recommendation.utils import extract_ids_from_answer
        for key, sample in samples.items():
            if not extract_ids_from_answer(sample["ground_truth"]):
                raise ValueError(f"No valid ground-truth SIDs at sample {key}.")
            if len(sample["generations"]) != 32:
                raise ValueError(f"Expected 32 beam candidates at sample {key}.")
        config["evaluation_config"].update(
            k_values=[1, 4, 8, 16, 32], evaluation_mode="sid", select_k="first_k")
    else:
        raise ValueError(f"{task} needs the WIP judge adapter, not local scoring.")
    evaluator = get_evaluator(task)(samples=samples, task_name=task, task_config=config,
                                     data_dir=data_dir, overwrite=True)
    metrics, per_sample = evaluator.evaluate()
    if set(per_sample) != set(samples):
        raise RuntimeError("Official evaluator skipped samples; no complete result saved.")
    if not all(name in metrics and math.isfinite(metrics[name]) for name in evaluator.required_metrics):
        raise RuntimeError("Missing or non-finite required metrics.")
    return metrics, per_sample
