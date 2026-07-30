"""Evaluate a full-precision, real RTN W8A8, or fake-QDQ OneRec/Qwen model on math tasks.

This is a zero-shot, deterministic generation evaluation.  It reports relative
quantization-retention experiments against the same BF16 OneRec checkpoint; it is not a
replacement for the official Qwen3 leaderboard prompting protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable

import torch
from datasets import load_dataset
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.paths import model_root, real_results_root


BENCHMARKS = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "default_output": str(real_results_root() / "generic" / "math500_8b_bf16"),
        "kind": "latex",
    },
    "aime2024": {
        "dataset": "HuggingFaceH4/aime_2024",
        "split": "train",
        "default_output": str(real_results_root() / "generic" / "aime2024_8b_bf16"),
        "kind": "latex",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "default_output": str(real_results_root() / "generic" / "gsm8k_1p7b_bf16_nonthinking_5shot"),
        "kind": "numeric",
    },
}

QUANTIZATION_MODES = ("full_precision", "rtn_w8a8", "fake_qdq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=str(model_root() / "8B"))
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="math500")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_MODES,
        default="full_precision",
        help=(
            "full_precision loads ordinary BF16/FP16 Linear layers; rtn_w8a8 reuses the real "
            "FP8 runtime; fake_qdq applies composable FP8/INT8/INT4 QDQ before BF16/FP16 Linear calls."
        ),
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
    parser.add_argument("--sample_size", default="full", help='"full" or a positive integer.')
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--gsm8k_num_fewshot",
        type=int,
        default=5,
        help="Fixed GSM8K training demonstrations in the user message; ignored by other benchmarks.",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass Qwen3's enable_thinking flag to the tokenizer chat template.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Discard any previous per-sample records.")
    return parser.parse_args()


def _parse_sample_size(value: str) -> int | None:
    if value.lower() == "full":
        return None
    size = int(value)
    if size <= 0:
        raise ValueError(f"sample_size must be positive or 'full', got {value!r}.")
    return size


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str):
                completed.add(sample_id)
    return completed


def _sample_id(example: dict[str, Any], index: int) -> str:
    value = example.get("unique_id")
    return str(value) if value is not None else str(index)


def _prompt(
    tokenizer: Any,
    problem: str,
    *,
    benchmark_kind: str,
    enable_thinking: bool,
    fewshot_examples: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    if benchmark_kind == "numeric":
        system_message = "You are a mathematical problem solver. Solve the problem carefully."
        demonstrations = []
        for example in fewshot_examples or []:
            demonstrations.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
        demonstrations.append(f"Question: {problem}\nAnswer:")
        user_message = "\n\n".join(demonstrations)
    else:
        system_message = (
            "You are a mathematical problem solver. Solve the problem carefully but concisely. "
            "Put the final result in a LaTeX \\boxed{...} expression at the end of your answer."
        )
        user_message = f"Problem:\n{problem}"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def _last_boxed_expression(text: str) -> str | None:
    """Return the contents of the last complete ``\\boxed{...}`` expression."""
    candidates: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        content_start = match.end()
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[content_start:index])
                    break
    return candidates[-1] if candidates else None


def _parse_answer(expression: str) -> Any:
    """Prefer full LaTeX parsing, which preserves tuples and other structures."""
    try:
        return latex2sympy(expression)
    except Exception:
        parsed = parse(
            expression,
            extraction_config=(LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()),
            fallback_mode="no_fallback",
        )
        if not parsed:
            raise ValueError(f"Unable to parse expression: {expression!r}")
        return parsed


def _is_latex_correct(answer: str, completion: str) -> tuple[bool, str | None]:
    boxed_prediction = _last_boxed_expression(completion)
    if boxed_prediction is None:
        return False, None
    try:
        gold = _parse_answer(answer)
        prediction = _parse_answer(boxed_prediction)
        return bool(verify(gold, prediction, timeout_seconds=5)), boxed_prediction
    except Exception:
        return False, boxed_prediction


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def _last_number(text: str) -> str | None:
    matches = _NUMBER_PATTERN.findall(text.replace("−", "-"))
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        # Canonicalize harmless formatting differences such as 42.0 vs 42.
        numeric = float(value)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return format(numeric, ".15g")


def _gsm_gold_answer(answer: str) -> str | None:
    # GSM8K uses `#### <answer>` in the reference solution.  Extracting the
    # final number also keeps the code robust to a few whitespace variants.
    return _last_number(answer.split("####")[-1])


def _gsm_prediction(completion: str) -> tuple[str | None, str | None]:
    tagged = re.findall(r"####\s*([^\n]+)", completion)
    if tagged:
        return _last_number(tagged[-1]), "hash_answer"
    boxed = _last_boxed_expression(completion)
    if boxed is not None:
        return _last_number(boxed), "boxed_fallback"
    return _last_number(completion), "last_number_fallback"


def _is_gsm_correct(answer: str, completion: str) -> tuple[bool, str | None, str | None]:
    gold = _gsm_gold_answer(answer)
    prediction, source = _gsm_prediction(completion)
    return gold is not None and prediction == gold, prediction, source


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Load the selected math-evaluation runtime without changing its generation protocol."""
    if args.quantization == "full_precision":
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        print(f"[math_benchmark] loading full-precision model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(args.device).eval()
        return model, tokenizer, {"mode": "full_precision"}

    if args.quantization == "fake_qdq":
        from fake_quant.runtime import load_fake_qdq_causal_lm

        print(
            "[math_benchmark] loading fake-QDQ model from "
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

    # This is the exact RTN/min-max path used by the real recommendation W8A8
    # baseline.  It deliberately has no calibration stage, decode-A16 exception,
    # or activation-tail protection.
    from real_quant.naive_w8a8.run_hf_naive_w8a8 import HFNaiveW8A8Generator

    print(f"[math_benchmark] loading real RTN W8A8 model from {args.model_path}")
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


def _summarize(records: Iterable[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    all_records = list(records)
    correct = sum(bool(record["correct"]) for record in all_records)
    by_subject: dict[str, list[bool]] = defaultdict(list)
    for record in all_records:
        by_subject[str(record.get("subject", "unknown"))].append(bool(record["correct"]))
    return {
        "config": config,
        "num_samples": len(all_records),
        "num_correct": correct,
        "accuracy": (correct / len(all_records)) if all_records else 0.0,
        "mean_generate_seconds": (
            sum(float(record["generate_seconds"]) for record in all_records) / len(all_records)
            if all_records
            else 0.0
        ),
        "by_subject": {
            subject: {
                "num_samples": len(values),
                "num_correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for subject, values in sorted(by_subject.items())
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires a CUDA device.")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if args.gsm8k_num_fewshot < 0:
        raise ValueError("gsm8k_num_fewshot must be non-negative.")

    benchmark = BENCHMARKS[args.benchmark]
    output_dir = Path(args.output_dir or benchmark["default_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"
    if args.overwrite and records_path.exists():
        records_path.unlink()
    completed_ids = _load_completed_ids(records_path)

    requested_size = _parse_sample_size(args.sample_size)
    dataset_config = benchmark.get("config")
    dataset = load_dataset(
        benchmark["dataset"], dataset_config, split=benchmark["split"]
    ) if dataset_config is not None else load_dataset(benchmark["dataset"], split=benchmark["split"])
    examples = list(dataset)
    if requested_size is not None:
        examples = examples[:requested_size]
    selected_ids = {_sample_id(example, index) for index, example in enumerate(examples)}
    fewshot_examples: list[dict[str, Any]] = []
    if benchmark["kind"] == "numeric" and args.gsm8k_num_fewshot:
        fewshot_examples = list(
            load_dataset(benchmark["dataset"], dataset_config, split="train").select(
                range(args.gsm8k_num_fewshot)
            )
        )

    model, tokenizer, quantization_summary = _load_model_and_tokenizer(args)

    config = {
        "benchmark": args.benchmark,
        "dataset": benchmark["dataset"],
        "dataset_config": benchmark.get("config"),
        "split": benchmark["split"],
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "quantization": quantization_summary,
        "sample_size": args.sample_size,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "gsm8k_num_fewshot": args.gsm8k_num_fewshot if benchmark["kind"] == "numeric" else None,
        "protocol": (
            "zero-shot deterministic chat-template generation; math_verify symbolic answer match"
            if benchmark["kind"] == "latex"
            else "fixed 5-shot (unless configured otherwise) deterministic chat-template generation; numeric final-answer match, preferring `#### <number>`"
        ),
    }
    print(f"[math_benchmark] benchmark={args.benchmark}, samples={len(examples)}, already_complete={len(completed_ids)}")

    with records_path.open("a", encoding="utf-8") as output_handle, torch.inference_mode():
        for index, example in enumerate(examples):
            sample_id = _sample_id(example, index)
            if sample_id in completed_ids:
                continue
            problem = str(example["problem"] if benchmark["kind"] == "latex" else example["question"])
            answer = str(example["answer"])
            input_ids = _prompt(
                tokenizer,
                problem,
                benchmark_kind=benchmark["kind"],
                enable_thinking=args.enable_thinking,
                fewshot_examples=fewshot_examples,
            ).to(args.device)
            attention_mask = torch.ones_like(input_ids, device=args.device)
            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            torch.cuda.synchronize(args.device)
            duration = time.perf_counter() - start
            completion = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
            if benchmark["kind"] == "latex":
                correct, prediction = _is_latex_correct(answer, completion)
                prediction_source = "boxed"
            else:
                correct, prediction, prediction_source = _is_gsm_correct(answer, completion)
            record = {
                "sample_id": sample_id,
                "subject": example.get("subject", args.benchmark),
                "level": example.get("level"),
                "gold_answer": answer,
                "completion": completion,
                "boxed_prediction": prediction,
                "prediction_source": prediction_source,
                "correct": correct,
                "generate_seconds": duration,
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            print(
                f"[{index + 1}/{len(examples)}] correct={record['correct']} "
                f"time={duration:.2f}s answer={answer!r}"
            )

    raw_records = []
    with records_path.open("r", encoding="utf-8") as input_handle:
        for line in input_handle:
            record = json.loads(line)
            if record.get("sample_id") in selected_ids:
                raw_records.append(record)

    # Interrupted jobs are resumable.  If a user accidentally starts two
    # resumptions at once, keep the first completed record for each example so
    # the aggregate metric remains one score per benchmark item.
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            records_by_id.setdefault(sample_id, record)
    summary = _summarize(list(records_by_id.values()), config)
    summary["num_raw_records"] = len(raw_records)
    summary["num_duplicate_records_discarded"] = len(raw_records) - len(records_by_id)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[math_benchmark] saved samples to {records_path}")


if __name__ == "__main__":
    main()
