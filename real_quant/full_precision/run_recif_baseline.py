"""Resumable BF16 evaluation of three auxiliary RecIF tasks.

Reuses OpenOneRec's prompts/metrics and this repository's HF generation backend.
No quantized checkpoint is loaded here. See README.md for protocol differences.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from shared.paths import artifacts_path, benchmark_data_root, model_root, repo_root
from .generator import HFFullPrecisionGenerator, append_prompt_token, build_hf_generation_kwargs
from .recif_tasks import (
    TASKS, UPSTREAM_COMMIT, atomic_json, classification_token_ids, classify_batch,
    clean_caption, digest, evaluate_local,
)
from benchmark.tasks.v1_0.registry import get_loader


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model_path", type=Path, default=model_root() / "1.7B")
    parser.add_argument("--data_dir", type=Path, default=benchmark_data_root())
    parser.add_argument("--output_dir", type=Path, default=artifacts_path("results", "recif", "bf16"))
    parser.add_argument("--stage", choices=("all", "generate", "evaluate"), default="all")
    parser.add_argument("--sample_size", default="full", help="First N rows, or full (all parquet rows).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=128, help="Atomic resume checkpoint interval.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--check_only", action="store_true", help="Read-only preflight; never load model weights.")
    parser.add_argument("--allow_judge", action="store_true", help="Consent to sending captions/references to a paid external judge.")
    parser.add_argument("--judge_config", type=Path, help="Gemini or DeepSeek JSON config; credentials stay outside results.")
    parser.add_argument("--bertscore_model", default="bert-base-chinese", help="HF ID or local directory.")
    parser.add_argument("--wip_reference_cache", type=Path,
                        default=artifacts_path("results", "recif", "wip_reference_cache"),
                        help="Share reference WIPs across model comparison arms; keyed by text/judge config.")
    return parser


def parse_args(argv=None, *, parser=None):
    parser = build_parser() if parser is None else parser
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.chunk_size < 1:
        parser.error("batch_size and chunk_size must be positive")
    if args.sample_size != "full":
        try:
            if int(args.sample_size) < 1:
                raise ValueError
        except ValueError:
            parser.error("sample_size must be full or a positive integer")
    return args


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_info(args):
    path = args.data_dir / args.task / f"{args.task}_test.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Use python -m real_quant.full_precision.download_recif_tasks --help")
    parquet = pq.ParquetFile(path)
    if not {"messages", "metadata"}.issubset(parquet.schema.names):
        # Nested Arrow schemas expose leaf names via .names.
        if not {"messages", "metadata"}.issubset(parquet.schema_arrow.names):
            raise ValueError("RecIF parquet requires messages and metadata columns.")
    rows = parquet.metadata.num_rows
    count = rows if args.sample_size == "full" else int(args.sample_size)
    if not 0 < count <= rows:
        raise ValueError(f"Requested {count} rows but dataset has {rows}.")
    return path, count, rows


def data_chunks(path: Path, count: int, chunk_size: int):
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size, columns=["messages", "metadata"]):
        if offset >= count:
            break
        rows = batch.slice(0, min(len(batch), count - offset)).to_pylist()
        # to_pylist keeps nested content as lists, unlike pandas' ndarray cells.
        frame = pd.DataFrame(rows, index=range(offset, offset + len(rows)))
        yield offset, frame
        offset += len(rows)


def format_chunk(loader, frame):
    for key, raw in frame["messages"].items():
        messages = json.loads(raw) if isinstance(raw, str) else raw
        if not messages or messages[-1].get("role") == "assistant":
            raise ValueError(f"Sample {key}: empty prompt or trailing assistant answer; refusing evaluation leakage.")
    data = loader._process_dataframe(frame, require_answer=True)
    if set(data) != {str(i) for i in frame.index}:
        raise ValueError("Loader skipped invalid input rows; refusing partial evaluation.")
    return data


def model_identity(path: Path):
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Expected local HF model directory: {path}")
    # Full weight hashing would reread gigabytes every restart. Record resolved
    # paths, size and nanosecond mtime; hash small config/tokenizer files.
    files = []
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix in (".json", ".safetensors", ".bin", ".jinja"):
            stat = file.stat()
            files.append([file.name, str(file.resolve()), stat.st_size, stat.st_mtime_ns,
                          sha256_file(file) if file.suffix in (".json", ".jinja") else None])
    return files


def protocol(args, path, count, total):
    template = repo_root() / "benchmarks/benchmark/tasks/v1_0/qwen3_soft_switch.jinja2"
    return {
        "schema_version": 1, "task": args.task, "split": "test", "backend": "hf_full_precision_baseline",
        "upstream_commit": UPSTREAM_COMMIT, "model_path": str(args.model_path.resolve()),
        "model_files": model_identity(args.model_path), "dtype": args.dtype,
        "data_sha256": sha256_file(path), "data_path": str(path.resolve()),
        "dataset_rows": total, "sample_count": count, "sample_policy": "first_n_rows",
        "enable_thinking": False, "chat_template_sha256": sha256_file(template),
        "batch_size": args.batch_size, "chunk_size": args.chunk_size, "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "generation": {"interactive": "beam32_return32_tokens3_no_sid_mask",
                       "label_pred": "next_token_pair_softmax_yes_no",
                       "item_understand": "greedy_tokens128"}[args.task],
    }


def part_path(directory, offset):
    return directory / "parts" / f"{offset:09d}.json"


def read_part(path, run_id, expected_ids):
    part = json.loads(path.read_text())
    if part.get("run_id") != run_id or set(part.get("samples", {})) != set(expected_ids):
        raise ValueError(f"Resume checkpoint mismatch: {path}. Use a new output_dir.")
    return part["samples"]


def collect_parts(directory, run_id, count, chunk_size):
    samples = {}
    for start in range(0, count, chunk_size):
        expected = [str(i) for i in range(start, min(start + chunk_size, count))]
        samples.update(read_part(part_path(directory, start), run_id, expected))
    return samples


def generate(args, directory, dataset_path, count, run_protocol, *,
             generator_factory=None, progress_label="BF16"):
    run_id = digest(run_protocol)
    manifest = directory / "run_config.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != run_protocol:
            raise ValueError(f"Configuration changed in {directory}; use a new output_dir, not overwritten results.")
    else:
        if any(directory.iterdir()):
            # The process lock is created by main, and is not experiment data.
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError(f"Unrecognized existing output in {directory}.")
        atomic_json(manifest, run_protocol)
    generator = None
    loader = None
    completed = 0
    set_seed(args.seed)
    progress = tqdm(total=count, desc=f"{progress_label} {args.task}", unit="sample", dynamic_ncols=True)
    try:
        for start, frame in data_chunks(dataset_path, count, args.chunk_size):
            checkpoint = part_path(directory, start)
            if checkpoint.exists():
                read_part(checkpoint, run_id, [str(i) for i in frame.index])
                completed += len(frame)
                progress.update(len(frame))
                continue
            if generator is None:
                if generator_factory is None:
                    generator = HFFullPrecisionGenerator.from_pretrained(
                        str(args.model_path), device=args.device, dtype=args.dtype,
                        attn_implementation=args.attn_implementation)
                else:
                    generator = generator_factory(args)
                loader = get_loader(args.task, str(args.data_dir), generator.tokenizer, enable_thinking=False)
            data = format_chunk(loader, frame)
            keys = list(data)
            samples = {}
            token_ids = classification_token_ids(generator.tokenizer) if args.task == "label_pred" else None
            generation_kwargs = build_hf_generation_kwargs(
                max_new_tokens=3 if args.task == "interactive" else 128,
                num_beams=32 if args.task == "interactive" else 1,
                num_return_sequences=32 if args.task == "interactive" else 1,
                pad_token_id=generator.tokenizer.pad_token_id, eos_token_id=generator.tokenizer.eos_token_id)
            for i in range(0, len(keys), args.batch_size):
                batch_keys = keys[i:i + args.batch_size]
                prompts = [append_prompt_token(data[key]["prompt"],
                           "<|sid_begin|>" if args.task == "interactive" else None) for key in batch_keys]
                if token_ids is not None:
                    outputs = dict(zip(batch_keys, classify_batch(generator.model, generator.tokenizer,
                                                                 prompts, generator.device, token_ids)))
                else:
                    outputs, _ = generator.generate_batch(batch=list(zip(batch_keys, prompts)),
                                                           generation_kwargs=generation_kwargs)
                for key in batch_keys:
                    generations = outputs[key]
                    if args.task == "item_understand":
                        generations = [clean_caption(text, generator.tokenizer) for text in generations]
                    samples[key] = {"generations": generations, "ground_truth": data[key]["ground_truth"]}
                progress.update(len(batch_keys))
            atomic_json(checkpoint, {"run_id": run_id, "samples": samples})
            completed += len(samples)
    finally:
        progress.close()
        del generator, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if completed != count:
        raise RuntimeError(f"Incomplete generation: {completed}/{count}")
    # Only short answers/candidates and labels are retained, not long prompts.
    payload = {"model_name": args.model_path.name, "task_name": args.task, "split": "test",
               "run_id": run_id, "status": "generation_complete", "protocol": run_protocol,
               "samples": collect_parts(directory, run_id, count, args.chunk_size)}
    atomic_json(directory / "test_generated.json", payload)


def score(args, directory):
    path = directory / "test_generated.json"
    payload = json.loads(path.read_text())
    run_protocol = json.loads((directory / "run_config.json").read_text())
    if (payload.get("run_id") != digest(run_protocol) or payload.get("task_name") != args.task
            or payload.get("status") != "generation_complete"
            or set(payload["samples"]) != {str(i) for i in range(run_protocol["sample_count"])}):
        raise ValueError("Incomplete or mismatched generation payload.")
    score_config = {"generation_sha256": sha256_file(path), "task": args.task,
                    "metric_protocol": "official_sid_k1_4_8_16_32_or_auc_or_macro_wip_v1"}
    if args.task == "item_understand":
        from .recif_wip import judge_preflight, public_judge_config, score_wip
        judge_config = judge_preflight(args)
        score_config.update(judge=public_judge_config(judge_config),
                            bertscore_model=args.bertscore_model, bertscore_layers=9)
    result_file = directory / "eval_results.json"
    if result_file.exists():
        old = json.loads(result_file.read_text())
        if old.get("score_config") != score_config or old.get("status") != "complete":
            raise ValueError(f"Scoring configuration changed: {result_file}; move the old metric file aside first.")
        print(f"[complete] Existing metrics: {result_file}", flush=True)
        return
    if args.task == "item_understand":
        metrics, per_sample = score_wip(payload["samples"], directory, judge_config, args.bertscore_model,
                                        reference_cache_dir=args.wip_reference_cache)
    else:
        metrics, per_sample = evaluate_local(args.task, payload["samples"], str(args.data_dir))
    atomic_json(directory / "per_sample_metrics.json", per_sample)
    atomic_json(result_file, {"status": "complete", "model_name": payload["model_name"],
                            "task_name": args.task, "run_id": payload["run_id"],
                            "score_config": score_config, "metrics": metrics})
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[complete] {result_file}", flush=True)


def run(args, *, generator_factory=None, protocol_extra=None, progress_label="BF16"):
    """Shared task protocol; optional load-only adapter changes model construction."""
    directory = args.output_dir / args.model_path.name / args.task
    print(f"[protocol] model={progress_label} task={args.task} stage={args.stage} dtype={args.dtype} thinking=False", flush=True)
    print(f"[output] {directory}", flush=True)
    if args.task == "item_understand" and args.stage != "generate" and args.allow_judge:
        from .recif_wip import judge_preflight
        judge_preflight(args)  # Dependency/config validation; no API request.
    if args.stage != "evaluate":
        dataset_path, count, total = dataset_info(args)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
        if args.task == "label_pred":
            classification_token_ids(tokenizer)
        loader = get_loader(args.task, str(args.data_dir), tokenizer, enable_thinking=False)
        _, frame = next(data_chunks(dataset_path, min(count, 2), 2))
        format_chunk(loader, frame)
        del loader, tokenizer
        print(f"[data] {count}/{total} rows, batch_size={args.batch_size}, checkpoint every {args.chunk_size}", flush=True)
        if args.check_only:
            print("[check] Generation prerequisites OK; model weights not loaded.", flush=True)
    elif not (directory / "test_generated.json").is_file():
        raise FileNotFoundError(f"No completed generation: {directory / 'test_generated.json'}")
    if args.check_only:
        if args.task == "item_understand" and not args.allow_judge:
            print("[pending] WIP requires --allow_judge --judge_config and optional judge dependencies.", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {directory}") from None
        if args.stage != "evaluate":
            run_protocol = protocol(args, dataset_path, count, total)
            run_protocol.update(protocol_extra or {})
            generate(args, directory, dataset_path, count, run_protocol,
                     generator_factory=generator_factory, progress_label=progress_label)
        if args.stage != "generate":
            if args.task == "item_understand" and not args.allow_judge:
                if args.stage == "evaluate":
                    raise ValueError("WIP scoring sends text externally; explicitly supply --allow_judge and --judge_config.")
                print("[pending] Caption generation saved; WIP not scored. Rerun --stage evaluate with judge consent/config.", flush=True)
            else:
                score(args, directory)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
