"""Compute paired token-level perplexity for a OneRec/Qwen checkpoint.

WikiText-2 is evaluated as its complete raw-test token stream. C4 uses a
locally cached, deterministic prefix of the English validation stream so that
the same fixed windows are reused by every full-precision, real-RTN, and fake-QDQ run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.paths import data_root, model_root, real_results_root


DATASETS = {
    "wikitext2": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "default_output": str(real_results_root() / "generic" / "wikitext2_1p7b_bf16"),
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "revision": "main",
        "default_output": str(real_results_root() / "generic" / "c4_validation_1p7b_bf16"),
    },
}
QUANTIZATION_MODES = ("full_precision", "rtn_w8a8", "fake_qdq")
C4_DEFAULT_NUM_SEQUENCES = 256
C4_CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=str(model_root() / "1.7B"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="wikitext2")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_MODES,
        default="full_precision",
        help="Use full precision, the existing real FP8 RTN W8A8 runtime, or composable fake QDQ.",
    )
    parser.add_argument(
        "--weight_quant_format",
        choices=("none", "fp8_e4m3fn", "int8", "int4"),
        default="fp8_e4m3fn",
        help="Weight format used only with --quantization fake_qdq.",
    )
    parser.add_argument(
        "--activation_quant_format",
        choices=("none", "fp8_e4m3fn", "int8", "int4"),
        default="fp8_e4m3fn",
        help="Activation format used only with --quantization fake_qdq.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=2048,
        help="Independent teacher-forcing window length; GPTQ-style PPL reporting commonly uses 2048.",
    )
    parser.add_argument(
        "--max_sequences",
        default="full",
        help='"full" or a positive integer; useful only for a smoke test.',
    )
    parser.add_argument(
        "--c4_num_sequences",
        type=int,
        default=C4_DEFAULT_NUM_SEQUENCES,
        help=(
            "Number of fixed C4 validation windows cached for --dataset c4. "
            "The default is 256 windows of --sequence_length tokens."
        ),
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Build or validate the local C4 token-window cache without loading a model or scoring PPL.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Discard previous per-window records.")
    return parser.parse_args()


def _parse_max_sequences(value: str) -> int | None:
    if value.lower() == "full":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"max_sequences must be positive or 'full', got {value!r}.")
    return parsed


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    if args.quantization == "full_precision":
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        print(f"[ppl] loading full-precision model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(args.device).eval()
        return model, tokenizer, {"mode": "full_precision"}

    if args.quantization == "fake_qdq":
        from fake_quant.runtime import load_fake_qdq_causal_lm

        print(
            "[ppl] loading fake-QDQ model from "
            f"{args.model_path} (weight={args.weight_quant_format}, "
            f"activation={args.activation_quant_format})"
        )
        return load_fake_qdq_causal_lm(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=True,
            weight_quant_format=args.weight_quant_format,
            activation_quant_format=args.activation_quant_format,
        )

    # Keep this exactly aligned with the real recommendation RTN W8A8
    # baseline: min-max FP8 weights, dynamic FP8 activations, with no protected
    # decode tokens or activation-tail exception.
    from real_quant.naive_w8a8.run_hf_naive_w8a8 import HFNaiveW8A8Generator

    print(f"[ppl] loading real RTN W8A8 model from {args.model_path}")
    generator = HFNaiveW8A8Generator.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=True,
        weight_quant_mode="minmax",
        activation_quant_mode="dynamic",
        decode_a16_when_single_token=False,
    )
    quant_summary = generator.quant_summary
    return generator.model, generator.tokenizer, {
        "mode": "rtn_w8a8",
        "weight_quant_mode": "minmax",
        "activation_quant_mode": "dynamic",
        "decode_a16_when_single_token": False,
        "replaced_linears": quant_summary.replaced_linears,
        "skipped_linears": quant_summary.skipped_linears,
        "shared_attention_modules": quant_summary.shared_attention_modules,
        "shared_mlp_modules": quant_summary.shared_mlp_modules,
    }


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    """Fingerprint the tokenizer vocabulary used by the reusable C4 cache."""
    vocab = tokenizer.get_vocab()
    encoded = json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_ids_fingerprint(token_ids: torch.Tensor) -> str:
    return hashlib.sha256(token_ids.cpu().contiguous().numpy().tobytes()).hexdigest()


def _c4_cache_paths(*, sequence_length: int, num_sequences: int) -> tuple[Path, Path]:
    cache_dir = data_root() / "ppl" / "c4_en_validation"
    stem = f"v{C4_CACHE_VERSION}_first_{num_sequences}x{sequence_length}"
    return cache_dir / f"{stem}.pt", cache_dir / f"{stem}.json"


def _load_or_prepare_c4_token_ids(
    tokenizer: Any,
    *,
    sequence_length: int,
    num_sequences: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a fixed C4-validation token prefix, creating it with streaming if needed."""
    if num_sequences <= 0:
        raise ValueError(f"c4_num_sequences must be positive, got {num_sequences}.")

    token_path, metadata_path = _c4_cache_paths(
        sequence_length=sequence_length,
        num_sequences=num_sequences,
    )
    tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
    expected_tokens = int(sequence_length) * int(num_sequences)
    if token_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            token_ids = torch.load(token_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[ppl] rebuilding invalid C4 cache at {token_path}: {exc}")
        else:
            if (
                isinstance(token_ids, torch.Tensor)
                and token_ids.dtype == torch.long
                and token_ids.numel() == expected_tokens
                and metadata.get("tokenizer_fingerprint") == tokenizer_fingerprint
                and metadata.get("sequence_length") == sequence_length
                and metadata.get("num_sequences") == num_sequences
            ):
                token_ids_fingerprint = _token_ids_fingerprint(token_ids)
                if metadata.get("token_ids_fingerprint") != token_ids_fingerprint:
                    metadata["token_ids_fingerprint"] = token_ids_fingerprint
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                metadata = dict(metadata)
                metadata["cache_hit"] = True
                return token_ids, metadata
            print(f"[ppl] rebuilding incompatible C4 cache at {token_path}")

    spec = DATASETS["c4"]
    print(
        "[ppl] streaming C4 English validation to prepare "
        f"{num_sequences} fixed windows of {sequence_length} tokens"
    )
    dataset = load_dataset(
        spec["path"],
        spec["name"],
        split=spec["split"],
        revision=spec["revision"],
        streaming=True,
    )
    separator_ids = tokenizer("\n\n", add_special_tokens=False, verbose=False).input_ids
    separator = torch.tensor(separator_ids, dtype=torch.long)
    pieces: list[torch.Tensor] = []
    collected_tokens = 0
    documents_read = 0
    for example in dataset:
        text = str(example.get("text", ""))
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
        if not ids:
            continue
        pieces.append(torch.tensor(ids, dtype=torch.long))
        if separator.numel():
            pieces.append(separator)
        collected_tokens += len(ids) + int(separator.numel())
        documents_read += 1
        if collected_tokens >= expected_tokens:
            break
    if collected_tokens < expected_tokens:
        raise RuntimeError(
            f"C4 validation stream ended after {collected_tokens} tokens; expected at least {expected_tokens}."
        )

    token_ids = torch.cat(pieces)[:expected_tokens].contiguous()
    metadata = {
        "cache_version": C4_CACHE_VERSION,
        "dataset": "c4",
        "dataset_source": spec["path"],
        "dataset_config": spec["name"],
        "dataset_split": spec["split"],
        "dataset_revision": spec["revision"],
        "selection": "first contiguous token windows from the deterministic validation stream",
        "document_separator": "\\n\\n tokenized without special tokens",
        "sequence_length": sequence_length,
        "num_sequences": num_sequences,
        "num_input_tokens": int(token_ids.numel()),
        "num_scored_tokens": num_sequences * (sequence_length - 1),
        "source_documents_read": documents_read,
        "source_tokens_collected_before_truncation": collected_tokens,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "token_ids_fingerprint": _token_ids_fingerprint(token_ids),
        "cache_hit": False,
    }
    token_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_ids, token_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[ppl] cached C4 windows at {token_path}")
    return token_ids, metadata


def _load_token_ids(
    tokenizer: Any,
    dataset_name: str,
    *,
    sequence_length: int,
    c4_num_sequences: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if dataset_name == "c4":
        return _load_or_prepare_c4_token_ids(
            tokenizer,
            sequence_length=sequence_length,
            num_sequences=c4_num_sequences,
        )

    spec = DATASETS[dataset_name]
    print(f"[ppl] loading {spec['path']} ({spec['name']}, {spec['split']})")
    dataset = load_dataset(spec["path"], spec["name"], split=spec["split"])
    text = "\n\n".join(str(value) for value in dataset["text"])
    # The corpus is deliberately tokenized as one stream and split below.  Do
    # not let the tokenizer mistake the temporary CPU tensor for a single model
    # input longer than the checkpoint context window.
    token_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    ).input_ids.squeeze(0).cpu()
    if token_ids.numel() < 2:
        raise RuntimeError(f"{dataset_name} produced fewer than two tokens.")
    return token_ids, {
        "selection": "complete split concatenated with double-newline separators",
        "num_input_tokens": int(token_ids.numel()),
    }


def _load_completed_indices(records_path: Path) -> set[int]:
    completed: set[int] = set()
    if not records_path.exists():
        return completed
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                completed.add(int(record["sequence_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def _records_by_index(records_path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not records_path.exists():
        return latest
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                latest[int(record["sequence_index"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return latest


def _window_ranges(total_tokens: int, sequence_length: int) -> list[tuple[int, int]]:
    """Return non-overlapping score windows, retaining the final short window."""
    windows = []
    for start in range(0, total_tokens, sequence_length):
        end = min(start + sequence_length, total_tokens)
        if end - start >= 2:
            windows.append((start, end))
    return windows


@torch.inference_mode()
def _score_window(model: Any, token_ids: torch.Tensor, device: str) -> tuple[float, int]:
    inputs = token_ids.unsqueeze(0).to(device)
    outputs = model(input_ids=inputs, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    targets = inputs[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
    return float(nll.item()), int(targets.numel())


def _write_summary(
    *,
    records_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    expected_indices: set[int],
) -> None:
    records = _records_by_index(records_path)
    selected = [records[index] for index in sorted(expected_indices) if index in records]
    token_count = sum(int(record["scored_tokens"]) for record in selected)
    nll = sum(float(record["negative_log_likelihood"]) for record in selected)
    summary = {
        "config": config,
        "num_sequences": len(selected),
        "num_scored_tokens": token_count,
        "total_negative_log_likelihood": nll,
        "mean_negative_log_likelihood": nll / token_count if token_count else None,
        "perplexity": math.exp(nll / token_count) if token_count else None,
        "mean_forward_seconds": (
            sum(float(record["forward_seconds"]) for record in selected) / len(selected) if selected else None
        ),
        "complete": len(selected) == len(expected_indices),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if args.c4_num_sequences <= 0:
        raise ValueError(f"c4_num_sequences must be positive, got {args.c4_num_sequences}.")

    if args.prepare_only:
        if args.dataset != "c4":
            raise ValueError("--prepare_only is only needed for the fixed C4 validation subset.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        _token_ids, metadata = _load_token_ids(
            tokenizer,
            args.dataset,
            sequence_length=args.sequence_length,
            c4_num_sequences=args.c4_num_sequences,
        )
        print(json.dumps(metadata, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires a CUDA device.")

    output_dir = Path(args.output_dir or DATASETS[args.dataset]["default_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "windows.jsonl"
    summary_path = output_dir / "summary.json"
    if args.overwrite and records_path.exists():
        records_path.unlink()

    model, tokenizer, quantization_config = _load_model_and_tokenizer(args)
    token_ids, data_preparation = _load_token_ids(
        tokenizer,
        args.dataset,
        sequence_length=args.sequence_length,
        c4_num_sequences=args.c4_num_sequences,
    )
    windows = _window_ranges(int(token_ids.numel()), args.sequence_length)
    max_sequences = _parse_max_sequences(args.max_sequences)
    if max_sequences is not None:
        windows = windows[:max_sequences]
    expected_indices = set(range(len(windows)))
    completed = _load_completed_indices(records_path)
    config = {
        "dataset": args.dataset,
        "dataset_source": DATASETS[args.dataset]["path"],
        "dataset_config": DATASETS[args.dataset]["name"],
        "dataset_split": DATASETS[args.dataset]["split"],
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "quantization": quantization_config,
        "sequence_length": args.sequence_length,
        "max_sequences": args.max_sequences,
        "c4_num_sequences": args.c4_num_sequences if args.dataset == "c4" else None,
        "data_preparation": data_preparation,
        "protocol": (
            "non-overlapping teacher-forcing token windows; first token of each window excluded from NLL"
        ),
    }
    print(f"[ppl] scoring {len(windows)} windows ({int(token_ids.numel())} total source tokens)")

    with records_path.open("a", encoding="utf-8") as handle:
        for index, (start, end) in enumerate(windows):
            if index in completed:
                continue
            torch.cuda.synchronize(args.device)
            begin = time.perf_counter()
            nll, scored_tokens = _score_window(model, token_ids[start:end], args.device)
            torch.cuda.synchronize(args.device)
            record = {
                "sequence_index": index,
                "start_token": start,
                "end_token": end,
                "input_tokens": end - start,
                "scored_tokens": scored_tokens,
                "negative_log_likelihood": nll,
                "forward_seconds": time.perf_counter() - begin,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"[ppl] {index + 1}/{len(windows)}: "
                f"nll/token={nll / scored_tokens:.6f}, {record['forward_seconds']:.3f}s",
                flush=True,
            )

    _write_summary(
        records_path=records_path,
        summary_path=summary_path,
        config=config,
        expected_indices=expected_indices,
    )


if __name__ == "__main__":
    main()
