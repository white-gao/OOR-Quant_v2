from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence, TypeVar

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark  # noqa: E402
from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402
from shared.paths import data_root, model_root, real_results_root  # noqa: E402

from .generator import HFFullPrecisionGenerator  # noqa: E402
from .results import build_generation_payload, save_generation_payload  # noqa: E402


T = TypeVar("T")
BatchSizeArg = int | Literal["auto"]
DEFAULT_MODEL_PATH = str(model_root() / "1.7B")
DEFAULT_DATA_DIR = str(data_root() / "onerec_data" / "benchmark_data")
DEFAULT_OUTPUT_DIR = str(real_results_root() / "recommender" / "bf16")
RECOMMENDATION_TASKS = ("ad", "product", "video", "label_cond", "interactive")


def batched_items(items: Sequence[T], *, batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def parse_sample_size(value: str | None) -> str | int | None:
    if value is None or value == "":
        return None
    if value == "full":
        return "full"
    return int(value)


def parse_batch_size_arg(value: str) -> BatchSizeArg:
    if value.lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"batch_size must be positive or 'auto', got {value!r}.")
    return parsed


def infer_model_size_billions(model_path: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\D|$)", model_path)
    if not match:
        return None
    return float(match.group(1))


def choose_auto_batch_size(
    *,
    total_memory_gb: float,
    model_size_billions: float | None,
    task: str,
) -> int:
    long_prompt_task = task in {"video", "interactive"}
    large_model = model_size_billions is not None and model_size_billions >= 6.0

    if total_memory_gb >= 120.0:
        if large_model:
            return 2 if long_prompt_task else 4
        return 4 if long_prompt_task else 8
    if total_memory_gb >= 80.0:
        if large_model:
            return 1 if long_prompt_task else 2
        return 2 if long_prompt_task else 4
    if total_memory_gb >= 48.0:
        if large_model:
            return 1
        return 1 if long_prompt_task else 2
    return 1


def get_device_total_memory_gb(device: str) -> float:
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_properties(index).total_memory / 1024**3


def resolve_batch_size(
    requested: BatchSizeArg,
    *,
    device: str,
    model_path: str,
    task: str,
) -> tuple[int, dict[str, Any]]:
    if requested != "auto":
        return int(requested), {"requested_batch_size": requested, "auto_batch_size": False}
    total_memory_gb = get_device_total_memory_gb(device)
    model_size_billions = infer_model_size_billions(model_path)
    batch_size = choose_auto_batch_size(
        total_memory_gb=total_memory_gb,
        model_size_billions=model_size_billions,
        task=task,
    )
    return batch_size, {
        "requested_batch_size": "auto",
        "auto_batch_size": True,
        "auto_batch_total_memory_gb": total_memory_gb,
        "auto_batch_model_size_billions": model_size_billions,
    }


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenOneRec-style full-precision HuggingFace baseline with latency stats."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_size", default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=parse_batch_size_arg, default=1)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--prompt_token", default=None)
    parser.add_argument("--output_scores", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def load_task_data(
    *,
    task_name: str,
    data_dir: str,
    tokenizer: Any,
    split: str,
    sample_size: str | int | None,
) -> dict[str, dict[str, Any]]:
    loader = get_loader(
        task_name=task_name,
        data_dir=data_dir,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    return loader.load_data(split=split, sample_size=sample_size)


def build_output_samples(
    *,
    test_data: dict[str, dict[str, Any]],
    generations: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for sample_id, sample in test_data.items():
        item = {
            "prompt": sample.get("prompt", ""),
            "generations": generations.get(sample_id, []),
            "ground_truth": sample.get("ground_truth", ""),
        }
        if "metadata" in sample:
            item["metadata"] = sample["metadata"]
        samples[sample_id] = item
    return samples


def result_path(output_dir: str, model_name: str, task_name: str, split: str) -> Path:
    return resolve_repo_path(output_dir) / model_name / task_name / f"{split}_generated.json"


def maybe_evaluate(output_dir: str, data_dir: str, overwrite: bool, *, task_name: str) -> None:
    output_root = resolve_repo_path(output_dir)
    data_root = resolve_repo_path(data_dir)
    Benchmark.evaluate_dev(
        generation_results_dir=str(output_root),
        output_path=str(output_root / "eval_results.json"),
        data_dir=str(data_root),
        overwrite=overwrite,
        task_types=[task_name],
    )


def main() -> None:
    args = parse_args()
    batch_size, batch_size_config = resolve_batch_size(
        args.batch_size,
        device=args.device,
        model_path=args.model_path,
        task=args.task,
    )
    if batch_size_config.get("auto_batch_size"):
        print(
            "[hf_full_precision] auto batch_size="
            f"{batch_size} (total_memory_gb={batch_size_config.get('auto_batch_total_memory_gb'):.2f}, "
            f"model_size_b={batch_size_config.get('auto_batch_model_size_billions')}, task={args.task})"
        )

    generator = HFFullPrecisionGenerator.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model_name = str(generator)
    output_file = result_path(args.output_dir, model_name, args.task, args.split)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    task_config = get_task_config(args.task)
    generation_config = dict(task_config.get("generation_config", {}))
    prompt_token = args.prompt_token
    if prompt_token is None:
        prompt_token = generation_config.get("prompt_token", "<|sid_begin|>")

    sample_size = parse_sample_size(args.sample_size)
    test_data = load_task_data(
        task_name=args.task,
        data_dir=str(resolve_repo_path(args.data_dir)),
        tokenizer=generator.tokenizer,
        split=args.split,
        sample_size=sample_size,
    )
    test_items = list(test_data.items())
    prompts = {sample_id: sample["prompt"] for sample_id, sample in test_items}
    generations, _ = generator.generate(
        prompts,
        prompt_token=prompt_token,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        output_scores=args.output_scores,
    )

    config = {
        "backend": "hf_full_precision",
        "reference": "OpenOneRec GitHub benchmark settings, vLLM/Ray replaced by HuggingFace generate",
        "task": args.task,
        "split": args.split,
        "data_dir": args.data_dir,
        "sample_size": args.sample_size,
        "dtype": args.dtype,
        "device": args.device,
        "batch_size": batch_size,
        **batch_size_config,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "prompt_token": prompt_token,
        "output_scores": args.output_scores,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    samples = build_output_samples(test_data=test_data, generations=generations)
    payload = build_generation_payload(
        model_name=model_name,
        task_name=args.task,
        split=args.split,
        samples=samples,
        latency_records=list(generator.latency_records.values()),
        config=config,
        hardware_info=generator.get_hardware_info(),
        num_params=generator.num_params,
    )
    save_generation_payload(payload, output_file)
    (output_file.parent / "hf_full_precision_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Generation results saved to: {output_file}")
    print(
        "Latency summary: "
        f"generate_total={payload['latency']['generate_time_total']:.3f}s, "
        f"end_to_end_total={payload['latency']['end_to_end_time_total']:.3f}s, "
        f"avg_generate={payload['latency']['generate_time_avg']:.6f}s/sample"
    )

    if args.evaluate:
        maybe_evaluate(args.output_dir, args.data_dir, args.overwrite, task_name=args.task)


if __name__ == "__main__":
    main()
