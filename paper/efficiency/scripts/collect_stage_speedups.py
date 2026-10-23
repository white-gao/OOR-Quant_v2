#!/usr/bin/env python3
"""Measure BF16/RTN stage latency and project RecQuant from paired FP8 probes.

No fake-quant full-model timing is used as a real-quant result. RecQuant is
ALWAYS identified as an operator-based estimate, in both prefill and decode.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.paths import model_root, benchmark_data_root, artifacts_path


def positive_list(value):
    values = [int(item) for item in value.split(",")]
    if len(values) != 3 or len(set(values)) != 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected three distinct positive integers, e.g. 1,2,4.")
    return values


def workloads(lengths, batches, decode_length):
    cases = {}
    for length in lengths:
        cases.setdefault((1, length), []).append("prefill")
    for batch in batches:
        cases.setdefault((batch, decode_length), []).append("decode")
    return [{"id": f"b{b}_l{length}", "batch": b, "length": length, "panels": panels}
            for (b, length), panels in cases.items()]


def project_recquant(rtn_ms, operators, trace):
    """Use measured RTN as anchor, replacing each shared group's latency."""
    delta = 0.0
    for item in trace:
        key = operator_key(item["group"], item["shape"])
        samples = operators[key]
        differences = [statistics.fmean(s["RecQuant"]) - statistics.fmean(s["RTN"])
                       for s in samples]
        delta += item["calls"] * statistics.fmean(differences)
    estimated = rtn_ms + delta
    if not math.isfinite(estimated) or estimated <= 0:
        raise ValueError("Non-positive RecQuant estimate; additive projection is invalid. Do not clip it.")
    return estimated, delta


def operator_key(group, shape):
    return group + ":" + "x".join(map(str, shape))


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(args, layers):
    paths = [args.model / "config.json"] + [args.checkpoint_dir / f"layer_{layer:02d}.pt" for layer in sorted(set(layers + [27]))]
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=model_root() / "1.7B")
    p.add_argument("--data", type=Path, default=benchmark_data_root() / "ad/ad_test.parquet")
    p.add_argument("--checkpoint-dir", type=Path, default=artifacts_path(
        "results/flat_quant/task_alignment/1p7b_ad_w8a8_pc_from_mse_train1024_epoch15/"
        "exp3_abc_boundary0.3_joint/1.7B/ad/flatquant_calibration"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "paper/efficiency/results" /
                   datetime.now(timezone.utc).strftime("stage_speedups_%Y%m%d_%H%M%S"))
    p.add_argument("--prefill-lengths", type=positive_list, default=[512, 1024, 2048])
    p.add_argument("--decode-batches", type=positive_list, default=[1, 2, 4])
    p.add_argument("--decode-length", type=int, default=1024)
    p.add_argument("--probe-layers", default="0,13,27")
    p.add_argument("--samples", type=int, default=8, help="Distinct request batches per workload.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--operator-warmup", type=int, default=10)
    p.add_argument("--operator-repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--check-only", action="store_true", help="CPU-only validation; no outputs or CUDA initialization.")
    p.add_argument("--smoke", action="store_true", help="Tiny 64/128/256-token protocol, NOT paper results.")
    p.add_argument("--resume", action="store_true", help="Reuse completed cases/probes in --output-dir, with identical settings.")
    p.add_argument("--summarize-only", action="store_true", help="CPU-only rebuild of summary/figures from completed raw files.")
    return p


def preflight(args):
    from stage_benchmark import load_config
    if args.smoke:
        args.prefill_lengths, args.decode_length = [64, 128, 256], 128
        args.samples, args.warmup, args.repeats = 1, 1, 2
        args.operator_warmup, args.operator_repeats = 1, 3
        args.probe_layers = "27"
    for name in ("samples", "warmup", "repeats", "operator_warmup", "operator_repeats",
                 "decode_length", "cpu_threads", "bootstrap"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.beam != 32:
        raise ValueError("This figure uses beam=32. Do not silently change the decoding workload.")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    cfg = json.loads((args.model / "config.json").read_text())
    if (cfg.get("hidden_size"), cfg.get("intermediate_size"), cfg.get("num_hidden_layers")) != (2048, 6144, 28):
        raise ValueError("Expected the OneRec-1.7B architecture (2048/6144/28).")
    layers = [int(value) for value in args.probe_layers.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < 28 for layer in layers):
        raise ValueError("Invalid representative probe layers.")
    for layer in layers:
        load_config(args.checkpoint_dir / f"layer_{layer:02d}.pt", layer)
    final_state, _ = load_config(args.checkpoint_dir / "layer_27.pt", 27)
    if final_state.get("objective") != "lfq_ce" or not math.isclose(
            final_state["config"].get("lfq_boundary_loss_weight", -1), .3):
        raise ValueError("Expected the joint RecQuant CE + 0.3 boundary checkpoint.")
    if os.environ.get("OOR_QUANT_ENABLE_FUSED_FP8_PROJECTIONS", "0") != "0":
        raise ValueError("Disable OOR_QUANT_ENABLE_FUSED_FP8_PROJECTIONS for matched unfused group probes.")
    return layers, workloads(args.prefill_lengths, args.decode_batches, args.decode_length)


def controlled_ids(ids, length, suffix):
    """Shape-only workload; repeat short AD token contexts, never hide padding."""
    if not ids or not suffix or length <= len(suffix):
        raise ValueError("Insufficient context or invalid SID suffix/length.")
    count = length - len(suffix)
    body = (ids * math.ceil(count / len(ids)))[-count:]
    return body + suffix


def input_manifest(args, cases, tokenizer):
    import pyarrow.parquet as pq
    # Efficiency inputs are controlled token streams, not quality-eval prompts.
    needed = args.samples * max(case["batch"] for case in cases)
    candidates = []
    for batch in pq.ParquetFile(args.data).iter_batches(batch_size=256, columns=["messages"]):
        for row in batch.to_pylist():
            messages = row["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            if not isinstance(messages, list):
                raise ValueError("AD messages must be a list.")
            # Remove a potential gold completion; do not use test answers.
            if messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]
            text = json.dumps(messages, ensure_ascii=False)
            candidates.append(text)
            if len(candidates) >= max(needed * 4, 256):
                break
        if len(candidates) >= max(needed * 4, 256):
            break
    if len(candidates) < needed:
        raise ValueError(f"Need {needed} distinct source records, found {len(candidates)}.")
    indices = random.Random(args.seed).sample(range(len(candidates)), needed)
    ids = [tokenizer.encode(candidates[i], add_special_tokens=False) for i in indices]
    suffix = tokenizer.encode("<|sid_begin|>", add_special_tokens=False)
    result = {}
    for case in cases:
        result[case["id"]] = [
            [controlled_ids(ids[n * case["batch"] + j], case["length"], suffix)
             for j in range(case["batch"])] for n in range(args.samples)]
    return {"input_kind": "controlled_ad_message_token_stream_suffix_crop_or_repeat",
            "not_quality_evaluation": True, "source_row_indices": indices,
            "source_token_lengths": [len(value) for value in ids], "batches": result}


def tensor_inputs(batch, device):
    import torch
    value = torch.tensor(batch, dtype=torch.long, device=device)
    return {"input_ids": value, "attention_mask": torch.ones_like(value)}


def measure_method(model, name, args, cases, inputs, kwargs, directory):
    from stage_benchmark import timed_generate, trace_shapes
    import torch
    from tqdm import tqdm
    result = {}
    for case in cases:
        saved_path = directory / f"{name}_{case['id']}.json"
        if args.resume and saved_path.is_file():
            saved = json.loads(saved_path.read_text())
            if saved.get("complete"):
                if saved.get("case") != case or len(saved["records"]) != args.samples * args.repeats:
                    raise ValueError(f"Invalid completed case in {saved_path}")
                result[case["id"]] = saved
                print(f"[resume] {name} {case['id']}: complete", flush=True)
                continue
        batches = inputs["batches"][case["id"]]
        first = tensor_inputs(batches[0], "cuda:0")
        for _ in range(args.warmup):
            timed_generate(model, first, kwargs)
        unsplit = [timed_generate(model, first, kwargs, split=False)["total_ms"]
                   for _ in range(3)]
        trace = trace_shapes(model, first, kwargs) if name == "BF16" else None
        records = []
        order = [(sample, repeat) for repeat in range(args.repeats) for sample in range(args.samples)]
        random.Random(args.seed).shuffle(order)
        for sample, repeat in tqdm(order, desc=f"{name} {case['id']}", unit="generate", dynamic_ncols=True):
            batch = tensor_inputs(batches[sample], "cuda:0")
            values = timed_generate(model, batch, kwargs)
            records.append({"sample": sample, "repeat": repeat, **values})
            atomic_json(directory / f"{name}_{case['id']}.json",
                        {"complete": False, "case": case, "records": records})
        result[case["id"]] = {"complete": True, "case": case, "records": records,
                                "trace": trace, "unsplit_total_ms": unsplit}
        atomic_json(directory / f"{name}_{case['id']}.json", result[case["id"]])
        del first
        torch.cuda.empty_cache()
    return result


def stage_sample_means(record, stage):
    grouped = {}
    for row in record["records"]:
        grouped.setdefault(row["sample"], []).append(row[stage + "_ms"])
    return [statistics.fmean(grouped[key]) for key in sorted(grouped)]


def summarize(args, cases, bf16, rtn, operators):
    import numpy as np
    rows = []
    rng = np.random.default_rng(args.seed)
    for case in cases:
        key = case["id"]
        if [row["forward_shapes"] for row in bf16[key]["records"]] != [
                row["forward_shapes"] for row in rtn[key]["records"]]:
            raise ValueError(f"BF16/RTN generation shapes differ for {key}.")
        for stage in case["panels"]:
            base = np.asarray(stage_sample_means(bf16[key], stage))
            quant = np.asarray(stage_sample_means(rtn[key], stage))
            trace = [entry for entry in bf16[key]["trace"] if entry["stage"] == stage]
            rec_ms, delta = project_recquant(float(quant.mean()), operators, trace)
            for method, elapsed, kind in (("RTN", float(quant.mean()), "measured_stage"),
                                         ("RecQuant", rec_ms, "operator_projected_stage")):
                shift = 0 if method == "RTN" else delta
                boot = []
                if len(base) > 1:
                    for _ in range(args.bootstrap):
                        selected = rng.integers(0, len(base), size=len(base))
                        denominator = float(quant[selected].mean()) + shift
                        if denominator <= 0:
                            raise ValueError("Bootstrap encountered invalid projected time.")
                        boot.append(float(base[selected].mean()) / denominator)
                rows.append({"stage": stage, "case": key, "batch": case["batch"],
                             "input_length": case["length"], "method": method, "data_kind": kind,
                             "bf16_ms": float(base.mean()), "method_ms": elapsed,
                             "added_operator_ms": shift, "speedup": float(base.mean()) / elapsed,
                             "ci95_low": float(np.quantile(boot, .025)) if boot else None,
                             "ci95_high": float(np.quantile(boot, .975)) if boot else None})
    audits = []
    for method, records in (("BF16", bf16), ("RTN", rtn)):
        for case in cases:
            record = records[case["id"]]
            # Compare the same first input batch; don't mix different samples.
            split_ms = statistics.fmean(row["total_ms"] for row in record["records"] if row["sample"] == 0)
            unsplit_ms = statistics.fmean(record["unsplit_total_ms"])
            audits.append({"method": method, "case": case["id"],
                           "split_over_unsplit": split_ms / unsplit_ms,
                           "warning": abs(split_ms / unsplit_ms - 1) > .10})
    return {"schema_version": 1, "data_kind": "measured_rtn_and_operator_estimated_recquant",
            "model": "OneRec-1.7B", "precision": "W8A8", "reference": "BF16",
            "smoke_test": args.smoke, "prefill_lengths": args.prefill_lengths,
            "decode_batches": args.decode_batches, "decode_length": args.decode_length,
            "recquant_estimator": "measured RTN stage + sum(calls * mean_layer(rec_group_ms-rtn_group_ms))",
            "ci_scope": "paired request-batch bootstrap; conditional on measured operator means; excludes projection model bias",
            "synchronization_audit": audits, "rows": rows}


def save_summary(args, cases, bf16, rtn, operators):
    result = summarize(args, cases, bf16, rtn, operators)
    atomic_json(args.output_dir / "summary.json", result)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    from plot_stage_speedups import render
    render(args.output_dir / "summary.json")
    if any(row["warning"] for row in result["synchronization_audit"]):
        print("[warning] Split synchronization differs from unsplit timing by >10% in some cases; inspect the audit before publication.", flush=True)
    print(f"[done] {args.output_dir / 'summary.csv'}", flush=True)
    print("[notice] RecQuant columns are operator-based estimates, NOT measured deployed latency.", flush=True)


def main():
    args = parser().parse_args()
    if args.summarize_only:
        stored = json.loads((args.output_dir / "manifest.json").read_text())["args"]
        stored["output_dir"] = args.output_dir
        args = argparse.Namespace(**stored)
        cases = workloads(args.prefill_lengths, args.decode_batches, args.decode_length)
        raw = {}
        for method in ("BF16", "RTN"):
            raw[method] = {case["id"]: json.loads((args.output_dir / "raw" /
                         f"{method}_{case['id']}.json").read_text()) for case in cases}
            if not all(record.get("complete") for record in raw[method].values()):
                raise ValueError(f"Incomplete {method} stage measurements.")
        operators = json.loads((args.output_dir / "operators.json").read_text())
        required_layers = set(map(int, args.probe_layers.split(",")))
        if any({item["layer"] for item in values} != required_layers for values in operators.values()):
            raise ValueError("Incomplete operator measurements.")
        save_summary(args, cases, raw["BF16"], raw["RTN"], operators)
        return
    import torch
    torch.set_num_threads(args.cpu_threads)
    layers, cases = preflight(args)
    print(f"[protocol] 1.7B W8A8 beam=32; BF16/RTN measured, RecQuant estimated", flush=True)
    print(f"[checkpoint] {args.checkpoint_dir}\n[output] {args.output_dir}", flush=True)
    print(f"[workloads] {cases}\n[probe layers] {layers}", flush=True)
    if args.check_only:
        print("[check] CPU preflight passed; no CUDA initialized, no files written.", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Output is nonempty; use a new --output-dir. Existing raw data are preserved.")
    arg_record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.resume:
        previous = json.loads((args.output_dir / "manifest.json").read_text())
        ignored = {"resume", "check_only", "summarize_only", "output_dir"}
        if {k: v for k, v in previous["args"].items() if k not in ignored} != {
                k: v for k, v in arg_record.items() if k not in ignored}:
            raise ValueError("Resume settings differ from manifest.json; use the original arguments.")
    source_hashes = provenance(args, layers)
    if args.resume and previous.get("source_sha256") != source_hashes:
        raise ValueError("Model config or checkpoint contents changed; refusing to mix timings.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU/fake-quant timing fallback is allowed.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES or the launcher GPU setting.")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from stage_benchmark import build_probes, benchmark_group, generation_kwargs
    from real_quant.naive_w8a8.apply import apply_naive_w8a8
    from real_quant.naive_w8a8.modules import RealFP8Linear, set_fp8_record_functions_enabled
    # Fail early on unavailable CUDA custom ops rather than after a long BF16 run.
    native_probe = RealFP8Linear.from_linear(torch.nn.Linear(128, 128, bias=False,
                                 device="cuda", dtype=torch.bfloat16))
    with torch.inference_mode():
        native_probe(torch.randn(32, 128, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
    del native_probe
    set_fp8_record_functions_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    inputs = input_manifest(args, cases, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if args.resume and previous["input_sha256"] != input_hash:
        raise ValueError("Input manifest changed; refusing to mix results.")
    atomic_json(args.output_dir / "inputs.json", inputs)
    manifest = {"args": arg_record, "source_sha256": source_hashes,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "input_sha256": input_hash,
                "method_order": ["BF16", "RTN"],
                "limitations": ["RecQuant is operator-projected, not deployed stage timing",
                                "homogeneous-layer shape model using representative learned layers",
                                "fixed BF16 then RTN order; hardware drift may affect ratios",
                                "microbench cache/launch conditions can differ from full generation",
                                "controlled token streams; not original full-prompt quality evaluation"]}
    if args.resume and (previous["gpu"], previous["torch"], previous["transformers"], previous["cuda_visible_devices"] ) != (
            manifest["gpu"], manifest["torch"], manifest["transformers"], manifest["cuda_visible_devices"]):
        raise ValueError("Resume device or software changed; do not merge incompatible timings.")
    atomic_json(args.output_dir / "manifest.json", manifest)
    print("[1/4] Loading BF16 model and measuring generation stages.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16,
              attn_implementation="sdpa", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    kwargs = generation_kwargs(tokenizer, args.beam)
    with torch.inference_mode():
        bf16 = measure_method(model, "BF16", args, cases, inputs, kwargs, args.output_dir / "raw")
        print("[2/4] Paired RTN/RecQuant shared-projection microbenchmarks.", flush=True)
        unique = {}
        for case in cases:
            for entry in bf16[case["id"]]["trace"]:
                if entry["stage"] in case["panels"]:
                    unique[operator_key(entry["group"], entry["shape"])] = entry
        operators = {key: [] for key in unique}
        operator_path = args.output_dir / "operators.json"
        if args.resume and operator_path.is_file():
            operators = json.loads(operator_path.read_text())
            if set(operators) != set(unique):
                raise ValueError("Operator shapes changed on resume.")
        for layer in layers:
            if all(any(item["layer"] == layer for item in operators[key]) for key in unique):
                print(f"[resume] operators layer {layer}: complete", flush=True)
                continue
            print(f"[operators] restoring learned layer {layer}", flush=True)
            probes = build_probes(model.model.layers[layer], args.checkpoint_dir / f"layer_{layer:02d}.pt", layer)
            from tqdm import tqdm
            for key, entry in tqdm(unique.items(), desc=f"operators layer {layer}", unit="group", dynamic_ncols=True):
                if any(item["layer"] == layer for item in operators[key]):
                    continue
                raw = benchmark_group(*probes[entry["group"]], entry["shape"],
                          warmup=args.operator_warmup, repeats=args.operator_repeats, seed=args.seed)
                operators[key].append({"layer": layer, **raw})
                atomic_json(args.output_dir / "operators.json", operators)
            del probes
            gc.collect()
            torch.cuda.empty_cache()
        print("[3/4] Converting the model to true RTN FP8 and measuring stages.", flush=True)
        summary = apply_naive_w8a8(model, output_dtype=torch.bfloat16,
                                  decode_a16_when_single_token=False, use_fast_accum=False)
        if summary.replaced_linears != 196:
            raise ValueError(f"Unexpected quantization coverage: {summary}")
        rtn = measure_method(model, "RTN", args, cases, inputs, kwargs, args.output_dir / "raw")
    print("[4/4] Computing measured/estimated speedups and paired intervals.", flush=True)
    save_summary(args, cases, bf16, rtn, operators)


if __name__ == "__main__":
    main()
