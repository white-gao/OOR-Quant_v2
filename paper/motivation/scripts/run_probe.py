#!/usr/bin/env python3
"""Read-only model probes. All new data, caches and outputs stay in motivation/."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import urllib.request

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from flat_quant.run_m1_onerec_ad import (
    SID_ITEM_RE, build_lfq_sid_slot_batches, get_loader, get_transformer_layers,
    set_seed, sid_slot_token_ids,
)
from flat_quant.evaluate_lfq_boundary_diagnostics import _load_model
from flat_quant.evaluate_flatquant_alignment_diagnostics import _checkpoint_config
from flat_quant.flatquant import restore_flatquant_core_layers_from_checkpoints
from metrics import compare_logits, concentration, path_overlap
from beam_trace import trace_beams

MODEL = "/root/dataDisk/guowei/models/1.7B"
DATA = "/root/dataDisk/guowei/data/onerec_data/benchmark_data"
TEXT_URL = "https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json"
SLOTS = ("a", "b", "c")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def confined(path):
    path = Path(path).resolve()
    if path == HERE or HERE not in path.parents:
        raise ValueError(f"Output must be a subdirectory/file of {HERE}: {path}")
    return path


def atomic_json(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def file_stamp(path):
    path = Path(path).resolve()
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def checkpoint_dir(args):
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).resolve()
    return (REPO / "artifacts/results/flat_quant/official_fake_quant" /
            f"1p7b_{args.task}_{args.variant}_pc_prefix128_final512_heldout512_epoch15" /
            f"final_train512_heldout512/1.7B/{args.task}/flatquant_calibration")


def model_stamp(args):
    root = Path(args.model_path)
    paths = sorted(root.glob("*.safetensors")) or sorted(root.glob("pytorch_model*.bin"))
    if not paths:
        raise FileNotFoundError(f"No local model weights: {root}")
    paths += sorted(root.glob("*.json"))
    return [file_stamp(p) for p in paths]


def checkpoint_stamp(args):
    if args.variant == "bf16":
        return []
    count = AutoConfig.from_pretrained(args.model_path, local_files_only=True).num_hidden_layers
    root = checkpoint_dir(args)
    paths = [root / f"layer_{i:02d}.pt" for i in range(count)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete checkpoints: " + ", ".join(missing))
    config = _checkpoint_config(root, count - 1)
    expected = "fp4_e2m1" if args.variant == "w4a8" else "fp8_e4m3fn"
    if config.weight_quant_format != expected or config.activation_quant_format != "fp8_e4m3fn":
        raise ValueError("Checkpoint precision does not match the requested variant")
    if config.weight_group_size != 0:
        raise ValueError("These motivation comparisons require per-channel checkpoints")
    return [file_stamp(p) for p in paths]


def load_model(args):
    print(f"[load] {args.model_path}; variant={args.variant}; device={args.device}", flush=True)
    model = _load_model(args.model_path, dtype="bfloat16", device=torch.device(args.device))
    if args.variant != "bf16":
        layers = get_transformer_layers(model)
        root = checkpoint_dir(args)
        config = _checkpoint_config(root, len(layers) - 1)
        restore_flatquant_core_layers_from_checkpoints(
            model=model, layer_indices=list(range(len(layers))), config=config,
            checkpoint_dir=root, act_quant_mode="shared_input")
    return model.eval()


def recommendation_samples(args, tokenizer):
    import pandas as pd
    root = Path(args.data_dir) / args.task
    test_path, calib_path = root / f"{args.task}_test.parquet", root / f"{args.task}_calib.parquet"
    df = pd.read_parquet(test_path)
    calib = pd.read_parquet(calib_path, columns=["metadata", "messages"])

    def meta(row):
        return json.loads(row) if isinstance(row, str) else row

    def message_key(messages):
        return digest(json.loads(messages) if isinstance(messages, str) else messages)

    excluded = {str(meta(m)["uid"]) for m in calib.metadata}
    excluded_prompts = {message_key(m) for m in calib.messages}
    # Retain one randomly selected query per user, preventing pseudo-independent CIs.
    eligible, seen = [], set()
    for idx in np.random.default_rng(args.seed).permutation(len(df)):
        uid = str(meta(df.iloc[idx].metadata)["uid"])
        if uid in excluded or uid in seen or message_key(df.iloc[idx].messages) in excluded_prompts:
            continue
        seen.add(uid)
        eligible.append(int(idx))
    if len(eligible) < args.samples:
        raise ValueError(f"Only {len(eligible)} eligible distinct users, requested {args.samples}")
    indices = eligible[:args.samples]
    selected = df.iloc[indices]
    loader = get_loader(args.task, args.data_dir, tokenizer=tokenizer, enable_thinking=False)
    loaded = loader._process_dataframe(selected)
    if len(loaded) != args.samples:
        raise ValueError("Some selected rows are malformed; refusing silent replacement")
    samples, manifest = [], []
    for idx in selected.index:
        sample = loaded[str(idx)]
        sample["id"] = str(idx)
        samples.append(sample)
        manifest.append({"id": str(idx), "uid_sha256": digest(sample["metadata"]["uid"]),
                         "prompt_sha256": digest(sample["prompt"]),
                         "target_sha256": digest(sample["ground_truth"])})
    return samples, {"selection": manifest, "test": file_stamp(test_path),
                     "calib": file_stamp(calib_path), "calib_excluded_users": len(excluded),
                     "eligible_distinct_users": len(eligible)}


def text_samples(args, tokenizer):
    path = Path(args.text_json).resolve() if args.text_json else HERE / "cache/cmrc2018_dev.json"
    if not path.is_file():
        if args.text_json:
            raise FileNotFoundError(path)
        path = confined(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {TEXT_URL}", flush=True)
        with urllib.request.urlopen(TEXT_URL, timeout=120) as response:
            raw = response.read()
        payload = json.loads(raw)
        atomic_json(path, payload)
    raw = path.read_bytes()
    payload = json.loads(raw)
    passages, seen, titles = [], set(), set()
    articles = payload["data"]
    rng = random.Random(args.seed)
    order = list(range(len(articles)))
    rng.shuffle(order)
    for i in order:
        article = articles[i]
        title = article.get("title", article.get("id", str(i)))
        if title in titles:
            continue
        paragraphs = list(article["paragraphs"])
        rng.shuffle(paragraphs)
        for paragraph in paragraphs:
            content = paragraph["context"]
            key = digest(content)
            if key in seen:
                continue
            tokens = tokenizer(content, add_special_tokens=False)["input_ids"][:1024]
            if len(tokens) < 132:
                continue
            # No next-token filtering: punctuation and easy/hard positions are retained.
            positions = sorted(rng.sample(range(127, len(tokens) - 1), 4))
            passages.append({"id": key, "input_ids": tokens, "positions": positions,
                             "document_sha256": digest(title)})
            seen.add(key)
            titles.add(title)
            break
        if len(passages) == args.samples:
            break
    if len(passages) != args.samples:
        raise ValueError(f"Only {len(passages)} eligible documents; requested {args.samples}")
    return passages, {"source": str(path), "source_url": TEXT_URL,
                      "sha256": hashlib.sha256(raw).hexdigest(), "selection": passages,
                      "sampling": "one unique passage/article; four uniform next-token positions; prefix>=128"}


def source_stamp():
    # Plot styling can be iterated without invalidating expensive forward passes.
    paths = [HERE / "scripts" / name for name in ("run_probe.py", "metrics.py", "beam_trace.py")]
    paths += sorted((REPO / "flat_quant").rglob("*.py"))
    paths += sorted((REPO / "benchmarks/benchmark/tasks/v1_0").rglob("*.py"))
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def manifest_for(args, selection):
    return {"schema": 1, "job": args.job, "task": args.task if args.job != "text" else "cmrc2018",
            "samples": args.samples, "seed": args.seed, "dtype": "bfloat16",
            "model": model_stamp(args), "selection": selection, "source": source_stamp(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
            "protocol": {"beam_width": 32, "max_new_tokens": 3, "sid_mask": False,
                         "prefix": "first ground-truth SID, shared by teacher/student",
                         "local_tie_break": "ascending token ID", "text_positions": 4},
            "variant": args.variant, "checkpoints": checkpoint_stamp(args)}


def validate_manifest(path, expected):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Protocol/source files changed: {path}. Use a new RUN_TAG; outputs were not overwritten.")
    else:
        atomic_json(path, expected)


def require_teacher(output, manifest):
    teacher = output.parent / "bf16"
    expected = {**manifest, "variant": "bf16", "checkpoints": []}
    path = teacher / "manifest.json"
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError(f"Run the matching BF16 probe first: {teacher}")
    if not (teacher / "COMPLETE.json").is_file():
        raise ValueError(f"Teacher probe is incomplete: {teacher}")
    return teacher


@torch.inference_mode()
def conditional_record(args, model, tokenizer, sample, slot_ids, teacher, output):
    batch = build_lfq_sid_slot_batches(tokenizer=tokenizer, samples=[sample],
                                      prompt_token="<|sid_begin|>", device=torch.device(args.device))[0]
    hidden = []

    def hook(_module, _inputs, result):
        h = result[0] if isinstance(result, tuple) else result
        hidden.append(h[0, -3:].detach().cpu())

    handle = get_transformer_layers(model)[-1].register_forward_hook(hook)
    try:
        logits = model(**batch, use_cache=False, logits_to_keep=3).logits[0].detach().float().cpu()
    finally:
        handle.remove()
    if len(hidden) != 1:
        raise RuntimeError("Final block hook did not fire exactly once")
    record = {"id": sample["id"], "prompt_tokens": batch["input_ids"].shape[1] - 2, "slots": {}}
    cached = None if teacher is None else torch.load(teacher / "cache" / f"{sample['id']}.pt", weights_only=True)
    legal_logits = {}
    for i, slot in enumerate(SLOTS):
        z = logits[i, slot_ids[slot]]
        legal_logits[slot] = z
        entry = {"full_vocab": concentration(logits[i]), "legal_vocab": concentration(z),
                 "legal_mass": float(logits[i].softmax(-1)[slot_ids[slot]].sum())}
        if cached is not None:
            entry.update(compare_logits(cached["logits"][slot], z))
            h, th = hidden[0][i].float(), cached["hidden"][i].float()
            entry["final_block_nmse"] = float((h-th).square().sum() / th.square().sum().clamp_min(1e-12))
        record["slots"][slot] = entry
    if teacher is None:
        path = output / "cache" / f"{sample['id']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"logits": legal_logits, "hidden": hidden[0]}, tmp)
        tmp.replace(path)
    return record


@torch.inference_mode()
def beam_record(args, model, tokenizer, sample, slot_ids, teacher):
    batch = build_lfq_sid_slot_batches(tokenizer=tokenizer, samples=[sample],
                                      prompt_token="<|sid_begin|>", device=torch.device(args.device))[0]
    batch = {key: value[:, :-2] for key, value in batch.items()}
    length = batch["input_ids"].shape[1]
    with trace_beams(model, length) as trace:
        generated = model.generate(**batch, max_new_tokens=3, num_beams=32, num_return_sequences=32,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   return_dict_in_generate=False, output_scores=False)
    returned = generated[:, length:].detach().cpu().tolist()
    paths = {str(d): trace["live"].get(str(d), {}).get("paths", []) for d in (1, 2)}
    paths["3"] = returned
    sets = {slot: set(ids.tolist()) for slot, ids in slot_ids.items()}
    valid = [p for p in returned if len(p) == 3 and all(p[i] in sets[s] for i, s in enumerate(SLOTS))]
    targets = set()
    for match in SID_ITEM_RE.finditer(sample["ground_truth"]):
        tokens = tuple(tokenizer(match.group("sid"), add_special_tokens=False)["input_ids"])
        if len(tokens) != 3:
            raise ValueError("Malformed target SID")
        targets.add(tokens)
    if not targets:
        raise ValueError("Missing target SID")
    record = {"id": sample["id"], "prompt_tokens": length, "paths": paths, "trace": trace,
              "invalid_returned_fraction": 1-len(valid)/32, "sid_recall": {}, "sid_pass": {}}
    for k in (1, 8, 32):
        hits = len(set(map(tuple, returned[:k])) & targets)
        record["sid_recall"][str(k)] = hits / len(targets)
        record["sid_pass"][str(k)] = int(hits > 0)
    if teacher is not None:
        t = json.loads((teacher / "records" / f"{sample['id']}.json").read_text())
        record["path_retention"] = {str(d): path_overlap(t["paths"][str(d)], paths[str(d)]) for d in (1, 2, 3)}
        record["delta_sid_recall32"] = record["sid_recall"]["32"] - t["sid_recall"]["32"]
        record["lost_teacher_hit32"] = int(t["sid_pass"]["32"] and not record["sid_pass"]["32"])
        record["gained_hit32"] = int(not t["sid_pass"]["32"] and record["sid_pass"]["32"])
    return record


@torch.inference_mode()
def text_record(args, model, tokenizer, sample):
    ids = torch.tensor([sample["input_ids"]], device=args.device)
    positions = torch.tensor(sample["positions"], device=args.device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   logits_to_keep=positions, use_cache=False).logits[0].float().cpu()
    records = []
    for pos, z in zip(sample["positions"], logits):
        token = sample["input_ids"][pos+1]
        records.append({"prefix_tokens": pos+1, "next_token_id": token,
                        "next_token_text": tokenizer.decode([token]),
                        "full_vocab": concentration(z)})
    return {"id": sample["id"], "document_sha256": sample["document_sha256"], "positions": records}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("conditional", "beam", "text", "preflight"))
    parser.add_argument("--task", choices=("ad", "product", "video"), default="ad")
    parser.add_argument("--variant", choices=("bf16", "w8a8", "w4a8"), default="bf16")
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--text-json", help="Offline CMRC2018 SQuAD-format dev JSON")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-root", default=str(HERE / "results/v1_seed42"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.job == "text" and args.variant != "bf16":
        raise ValueError("The text control uses the BF16 model, not recommendation-calibrated weights")
    torch.set_num_threads(4)
    if args.job == "preflight":
        model_stamp(args)
        checkpoint_stamp(args)
        for split in ("test", "calib"):
            path = Path(args.data_dir) / args.task / f"{args.task}_{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
        print(f"[preflight OK] {args.task} / {args.variant}; no GPU model loaded")
        return
    root = confined(args.run_root)
    output = (root / "text" / args.variant if args.job == "text"
              else root / args.task / args.job / args.variant)
    output.mkdir(parents=True, exist_ok=True)
    # flock is automatically released on process exit; a killed process leaves no stale lock.
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is already writing {output}") from exc
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        samples, selection = (text_samples(args, tokenizer) if args.job == "text"
                              else recommendation_samples(args, tokenizer))
        manifest = manifest_for(args, selection)
        validate_manifest(output / "manifest.json", manifest)
        teacher = require_teacher(output, manifest) if args.variant != "bf16" else None
        def completed(sample):
            record = output / "records" / f"{sample['id']}.json"
            cached = output / "cache" / f"{sample['id']}.pt"
            return record.is_file() and (args.job != "conditional" or args.variant != "bf16" or cached.is_file())
        pending = [s for s in samples if not completed(s)]
        print(f"[probe] {args.job}/{args.task}/{args.variant}: {len(samples)-len(pending)}/{len(samples)} already saved; {output}", flush=True)
        if pending:
            model = load_model(args)
            slot_ids = ({s: torch.tensor(sorted(sid_slot_token_ids(tokenizer, s))) for s in SLOTS}
                        if args.job != "text" else {})
            for sample in tqdm(pending, desc=f"{args.job} {args.task} {args.variant}", dynamic_ncols=True):
                if args.job == "conditional":
                    record = conditional_record(args, model, tokenizer, sample, slot_ids, teacher, output)
                elif args.job == "beam":
                    record = beam_record(args, model, tokenizer, sample, slot_ids, teacher)
                else:
                    record = text_record(args, model, tokenizer, sample)
                atomic_json(output / "records" / f"{sample['id']}.json", record)
        atomic_json(output / "COMPLETE.json", {"count": len(samples), "manifest_sha256": digest(manifest)})
        print(f"[complete] {output}", flush=True)


if __name__ == "__main__":
    main()
