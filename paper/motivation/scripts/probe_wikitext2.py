#!/usr/bin/env python3
"""Article-clustered WikiText-2 BF16 diagnostics, isolated from legacy SID probes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import urllib.request

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import run_probe as common
from metrics import concentration
from plot_motivation import read_records

HERE = common.HERE
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATA_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
HEADING = re.compile(r"^=\s+([^=].*?)\s+=$")
LENGTHS = (256, 512, 1024)


def dataset_path(args):
    """Download only the pinned test parquet; accept an explicitly supplied offline file."""
    if not re.fullmatch(r"[0-9a-f]{40}", args.dataset_revision):
        raise ValueError("Use an immutable 40-character dataset revision, not 'main'")
    url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{args.dataset_revision}/{DATA_FILE}"
    path = (Path(args.dataset_file).resolve() if args.dataset_file else
            HERE / "cache/wikitext2" / args.dataset_revision / "test.parquet")
    if not path.is_file():
        if args.dataset_file:
            raise FileNotFoundError(path)
        path = common.confined(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read()
        except OSError as exc:
            raise RuntimeError("WikiText-2 download failed. Retry or set WIKITEXT_FILE to the test parquet.") from exc
        import io
        import pyarrow.parquet as pq
        pq.read_schema(io.BytesIO(content))  # Do not cache an error page as parquet.
        temporary = path.with_suffix(f".download.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return path, {"repo": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                  "revision": args.dataset_revision, "path": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "source_url": None if args.dataset_file else url,
                  "origin": "user-supplied offline parquet" if args.dataset_file else "pinned official parquet"}


def articles_from_lines(lines, expected=60):
    """A standalone top-level '= title =' starts an article, not an inline equation."""
    articles, start, title = [], None, None
    for index, line in enumerate(lines):
        match = HEADING.fullmatch(line.strip())
        standalone = ((index == 0 or not lines[index-1].strip()) and
                      (index+1 == len(lines) or not lines[index+1].strip()))
        if match and standalone:
            if start is not None:
                articles.append((start, index, title))
            start, title = index, match.group(1).strip()
        elif start is None and line.strip():
            raise ValueError("Nonempty text before the first article heading")
    if start is not None:
        articles.append((start, len(lines), title))
    if len(articles) != expected:
        raise ValueError(f"Expected {expected} articles, parsed {len(articles)}; check the test split and parser")
    result, seen = [], set()
    for start, end, title in articles:
        # Preserve benchmark text; explicitly restore a line break only if absent.
        text = "".join(line if line.endswith("\n") else line + "\n" for line in lines[start:end])
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha in seen:
            raise ValueError("Duplicate article text; refusing to count it as independent")
        seen.add(sha)
        result.append({"id": sha, "document_sha256": sha, "title": title,
                       "source_rows": [start, end], "text": text})
    return result


def select_articles(articles, tokenizer, args):
    selected, excluded = [], []
    for article in articles[:args.max_articles or None]:
        encoded = tokenizer(article["text"], add_special_tokens=False, return_offsets_mapping=True)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        eligible = range(args.min_context - 1, len(tokens) - 1)
        if not len(eligible):
            excluded.append({"id": article["id"], "reason": "shorter than minimum prefix plus target",
                             "tokens": len(tokens)})
            continue
        rng = random.Random(f"{args.seed}:{article['id']}")
        positions = sorted(rng.sample(eligible, min(args.positions_per_article, len(eligible))))
        long_positions = [p for p in positions if p + 1 >= max(LENGTHS)]
        control = sorted(rng.sample(long_positions, min(args.length_positions, len(long_positions))))
        selected.append({k: v for k, v in article.items() if k != "text"} | {
            "input_ids": tokens, "positions": positions, "length_positions": control,
            "target_char_spans": {str(p): list(offsets[p+1]) for p in positions}})
    if not selected:
        raise ValueError("No eligible WikiText-2 articles")
    return selected, excluded


def make_manifest(args, source, samples, excluded):
    files = common.source_stamp()
    files[str(Path(__file__).resolve().relative_to(common.REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"schema": 2, "job": "text", "task": "wikitext2", "variant": "bf16", "dtype": "bfloat16",
            "model": common.model_stamp(args), "seed": args.seed, "source": files,
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "pyarrow")},
            "selection": {"dataset": source, "selection": samples, "excluded": excluded},
            "protocol": {"normalization": "complete model vocabulary; FP32 softmax; temperature=1",
                         "positions_per_article": args.positions_per_article, "min_context": args.min_context,
                         "max_context": args.max_context, "length_positions_per_article": args.length_positions,
                         "control_lengths": list(LENGTHS), "bootstrap_unit": "article",
                         "max_articles_smoke_only": args.max_articles}}


def validate_sid_source(args, model):
    root = Path(args.sid_run_root)
    for job in ("conditional", "beam"):
        for variant in ("bf16", "w8a8", "w4a8"):
            path = root / "ad" / job / variant
            read_records(path)
            if json.loads((path / "manifest.json").read_text())["model"] != model:
                raise ValueError(f"Model differs from the existing AD probe: {path}")
    cmrc = root / "text/bf16"
    read_records(cmrc)
    if json.loads((cmrc / "manifest.json").read_text())["model"] != model:
        raise ValueError("Existing CMRC reference uses a different model")


@torch.inference_mode()
def last_logits(model, tokens, device, keep=1):
    ids = torch.tensor([tokens], device=device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=keep)
    return result.logits[0].float().cpu()


def position_statistics(logits, target, sid_ids):
    z = logits.detach().float().cpu()
    metrics = concentration(z)
    logp = z.log_softmax(-1)
    values, ids = torch.topk(logp, 128)
    return {"full_vocab": metrics, "top128_token_ids": ids.tolist(),
            "top128_probabilities": values.exp().tolist(), "log_normalizer": float(z.logsumexp(-1)),
            "target_nll": float(-logp[target]), "top1_correct": int(z.argmax() == target),
            "sid_token_mass": float(logp[sid_ids].exp().sum())}


def evaluate_position(model, sample, position, limit, device, sid_ids):
    start = max(0, position + 1 - limit)
    ids = sample["input_ids"]
    logits = last_logits(model, ids[start:position+1], device)[-1]
    return {"position": position, "prefix_start": start, "prefix_tokens": position + 1 - start,
            "next_token_id": ids[position+1], "target_char_span": sample["target_char_spans"][str(position)],
            **position_statistics(logits, ids[position+1], sid_ids)}


def causal_sanity(model, sample, device):
    """Adding the known next token must not affect the preceding causal position."""
    ids = sample["input_ids"][:129]
    short = last_logits(model, ids[:-1], device)[-1]
    extended = last_logits(model, ids, device, keep=2)[0]
    p, q = short.log_softmax(-1), extended.log_softmax(-1)
    kl = max(0., float((p.exp() * (p-q)).sum()))
    mass_delta = abs(concentration(short)["mass"]["32"] - concentration(extended)["mass"]["32"])
    if not np.isfinite(kl) or kl > .01 or mass_delta > .01:
        raise ValueError(f"Causal position sanity failed: KL={kl}, Top32 mass delta={mass_delta}")
    return {"kl": kl, "top32_mass_abs_delta": mass_delta, "max_logit_abs_delta": float((short-extended).abs().max()),
            "tolerance_kl": .01, "tolerance_mass": .01}


def load_record(path, sample, fingerprint):
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("manifest_sha256") != fingerprint or record["id"] != sample["id"]:
            raise ValueError(f"Stale or mismatched partial record: {path}")
    else:
        record = {k: sample[k] for k in ("id", "document_sha256", "title", "source_rows")}
        record.update({"manifest_sha256": fingerprint, "positions": [], "length_control": []})
    for field, planned in (("positions", {(p, min(p+1, 1024)) for p in sample["positions"]}),
                           ("length_control", {(p, n) for p in sample["length_positions"] for n in LENGTHS})):
        actual = [(r["position"], r["prefix_tokens"]) for r in record[field]]
        if len(set(actual)) != len(actual) or not set(actual) <= planned:
            raise ValueError(f"Unexpected or duplicate prediction positions: {path}")
        for row in record[field]:
            if len(row["top128_probabilities"]) != 128 or not np.isfinite(row["target_nll"]):
                raise ValueError(f"Invalid prediction statistics: {path}")
    return record


def collect(args, samples, manifest, output, tokenizer):
    fingerprint = common.digest(manifest)
    pending, completed = [], 0
    for sample in samples:
        path = output / "records" / f"{sample['id']}.json"
        record = load_record(path, sample, fingerprint)
        expected = len(sample["positions"]) + len(sample["length_positions"])*len(LENGTHS)
        have = len(record["positions"]) + len(record["length_control"])
        if have != expected:
            pending.append((sample, path, record))
        else:
            completed += 1
    print(f"[resume] {completed}/{len(samples)} articles complete; pending={len(pending)}", flush=True)
    if pending:
        model = common.load_model(args)
        sid_ids = torch.tensor(sorted({i for slot in common.SLOTS for i in common.sid_slot_token_ids(tokenizer, slot)}))
        sanity = causal_sanity(model, pending[0][0], args.device)
        common.atomic_json(output / "position_sanity.json", {"manifest_sha256": fingerprint, **sanity})
        print(f"[causal sanity] KL={sanity['kl']:.3g}, mass delta={sanity['top32_mass_abs_delta']:.3g}", flush=True)
        for sample, path, record in tqdm(pending, desc="WikiText-2 articles", dynamic_ncols=True):
            main = {r["position"]: r for r in record["positions"]}
            controls = {(r["position"], r["prefix_tokens"]): r for r in record["length_control"]}
            todo = [("positions", p, args.max_context) for p in sample["positions"] if p not in main]
            todo += [("length_control", p, n) for p in sample["length_positions"] for n in LENGTHS if (p, n) not in controls]
            for field, position, limit in tqdm(todo, desc=f"positions {sample['title'][:28]}", leave=False, dynamic_ncols=True):
                if field == "length_control" and limit == args.max_context:
                    row = {**main[position], "reused_primary": True}
                else:
                    row = evaluate_position(model, sample, position, limit, args.device, sid_ids)
                record[field].append(row)
                if field == "positions":
                    main[position] = row
                record[field].sort(key=lambda r: (r["position"], r["prefix_tokens"]))
                common.atomic_json(path, record)  # Resume even inside an interrupted article.
        del model
    common.atomic_json(output / "COMPLETE.json", {"count": len(samples), "manifest_sha256": fingerprint})
    read_records(output)
    print(f"[complete] {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--model-path", default=common.MODEL)
    parser.add_argument("--sid-run-root", default=str(HERE / "results/v1_seed42"))
    parser.add_argument("--output-dir", default=str(HERE / "results/v2_text_controls_seed42/wikitext2/onerec_bf16"))
    parser.add_argument("--dataset-file", help="Offline official WikiText-2 raw test parquet")
    parser.add_argument("--dataset-revision", default=REVISION)
    parser.add_argument("--positions-per-article", type=int, default=32)
    parser.add_argument("--length-positions", type=int, default=8)
    parser.add_argument("--max-articles", type=int, default=0, help="0 means all 60; nonzero is for isolated smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(variant="bf16", min_context=128, max_context=1024)
    args = parser.parse_args()
    if args.positions_per_article < 1 or args.length_positions < 0 or not 0 <= args.max_articles <= 60:
        parser.error("Invalid sample counts")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(4)
    common.set_seed(args.seed)
    output = common.confined(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {output}") from exc
        model = common.model_stamp(args)
        validate_sid_source(args, model)  # Fail before expensive text inference if reused inputs are absent.
        path, source = dataset_path(args)
        import pyarrow.parquet as pq
        lines = pq.read_table(path, columns=["text"])["text"].to_pylist()
        articles = articles_from_lines(lines)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with character offsets is required")
        samples, excluded = select_articles(articles, tokenizer, args)
        manifest = make_manifest(args, source, samples, excluded)
        common.validate_manifest(output / "manifest.json", manifest)
        count = sum(len(s["positions"]) for s in samples)
        controls = sum(len(s["length_positions"])*len(LENGTHS) for s in samples)
        print(f"[protocol] WikiText-2 test: {len(articles)} articles, selected={len(samples)}, excluded={len(excluded)}", flush=True)
        print(f"[protocol] primary positions={count}; length-control records={controls}; BF16 only; no training", flush=True)
        print(f"[output] {output}", flush=True)
        if args.action == "preflight":
            print("[preflight OK] Data, sample manifest, local model, and reusable AD/CMRC probes checked; no model loaded.", flush=True)
            return
        collect(args, samples, manifest, output, tokenizer)


if __name__ == "__main__":
    main()
