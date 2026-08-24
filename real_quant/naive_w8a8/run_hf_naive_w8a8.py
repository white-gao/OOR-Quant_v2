from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_quant.full_precision.generator import HFFullPrecisionGenerator, dtype_from_name
from real_quant.full_precision.results import build_generation_payload, save_generation_payload
from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    RECOMMENDATION_TASKS,
    BatchSizeArg,
    build_output_samples,
    load_task_data,
    maybe_evaluate,
    parse_batch_size_arg,
    parse_sample_size,
    resolve_batch_size,
    resolve_repo_path,
    result_path,
)
from shared.paths import real_results_root

from .apply import NaiveW8A8Summary, apply_naive_w8a8, iter_real_fp8_linears
from .gptq_runtime import (
    WEIGHT_QUANT_MODES,
    apply_gptaq_real_w8a8_layers,
    apply_gptq_real_w8a8_layers,
    build_model_batches,
    default_calib_split,
    format_prompt,
    get_transformer_layers,
    parse_layer_indices,
)
from .modules import FP8_MAX, ActivationQuantMode, require_fp8_runtime, set_fp8_record_functions_enabled
from real_quant.sharding import eval_run_output_dir, select_round_robin_eval_shard


from fake_quant.gptq import (
    DEFAULT_GPTQ_BLOCK_SIZE,
    DEFAULT_GPTQ_DAMP_PERCENT,
)

DEFAULT_OUTPUT_DIR = str(real_results_root() / "recommender" / "quantized")


class HFNaiveW8A8Generator(HFFullPrecisionGenerator):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        model_name: str,
        device: str | torch.device,
        num_params: float | None,
        quant_summary: NaiveW8A8Summary,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            num_params=num_params,
        )
        self.quant_summary = quant_summary

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        target_regex: str | None = None,
        skip_regex: str | None = None,
        use_fast_accum: bool = False,
        activation_quant_mode: ActivationQuantMode = "dynamic",
        decode_a16_when_single_token: bool = False,
        weight_quant_mode: str = "minmax",
        task: str = "ad",
        split: str = "test",
        data_dir: str = DEFAULT_DATA_DIR,
        prompt_token: str = "<|sid_begin|>",
        gptq_calib_split: str | None = None,
        gptq_calib_sample_size: str = "1024",
        gptq_layers: str = "all",
        gptq_damp_percent: float = DEFAULT_GPTQ_DAMP_PERCENT,
        gptq_block_size: int = DEFAULT_GPTQ_BLOCK_SIZE,
        gptaq_alpha: float = 1.0,
        gptaq_activation_aware: bool = True,
    ) -> "HFNaiveW8A8Generator":
        require_real_fp8_device(device)
        if weight_quant_mode not in WEIGHT_QUANT_MODES:
            raise ValueError(f"Unsupported weight_quant_mode {weight_quant_mode!r}; expected one of {WEIGHT_QUANT_MODES}.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype_from_name(dtype),
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        num_params = float(sum(p.numel() for p in model.parameters()))
        output_dtype = dtype_from_name(dtype)
        if output_dtype not in (torch.bfloat16, torch.float16):
            output_dtype = torch.bfloat16

        if weight_quant_mode == "minmax":
            quant_summary = apply_naive_w8a8(
                model,
                target_regex=target_regex,
                skip_regex=skip_regex,
                output_dtype=output_dtype,
                use_fast_accum=use_fast_accum,
                activation_quant_mode=activation_quant_mode,
                decode_a16_when_single_token=decode_a16_when_single_token,
            )
            model.to(device)
        else:
            model.to(device)
            layers = get_transformer_layers(model)
            layer_indices = parse_layer_indices(gptq_layers, num_layers=len(layers))
            calib_split = gptq_calib_split or default_calib_split(
                data_dir,
                split,
                task_name=task,
                resolve_path=resolve_repo_path,
            )
            calib_data = load_task_data(
                task_name=task,
                data_dir=str(resolve_repo_path(data_dir)),
                tokenizer=tokenizer,
                split=calib_split,
                sample_size=parse_sample_size(gptq_calib_sample_size),
            )
            calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
            input_device = torch.device(device)
            calib_batches = build_model_batches(
                tokenizer=tokenizer,
                prompts=calib_prompts,
                device=input_device,
            )
            print(
                "[hf_naive_w8a8] collecting GPTQ Hessians "
                f"mode={weight_quant_mode}, split={calib_split}, samples={len(calib_prompts)}, layers={layer_indices}"
            )
            if weight_quant_mode == "gptaq":
                quant_summary = apply_gptaq_real_w8a8_layers(
                    model=model,
                    model_batches=calib_batches,
                    layer_indices=layer_indices,
                    output_dtype=output_dtype,
                    target_regex=target_regex,
                    skip_regex=skip_regex,
                    use_fast_accum=use_fast_accum,
                    activation_quant_mode=activation_quant_mode,
                    decode_a16_when_single_token=decode_a16_when_single_token,
                    damp_percent=gptq_damp_percent,
                    block_size=gptq_block_size,
                    alpha=gptaq_alpha,
                    activation_aware=gptaq_activation_aware,
                )
            else:
                quant_summary = apply_gptq_real_w8a8_layers(
                    model=model,
                    model_batches=calib_batches,
                    layer_indices=layer_indices,
                    output_dtype=output_dtype,
                    target_regex=target_regex,
                    skip_regex=skip_regex,
                    use_fast_accum=use_fast_accum,
                    activation_quant_mode=activation_quant_mode,
                    decode_a16_when_single_token=decode_a16_when_single_token,
                    damp_percent=gptq_damp_percent,
                    block_size=gptq_block_size,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()
        suffix = "real-naive-w8a8" if weight_quant_mode == "minmax" else f"real-naive-w8a8-{weight_quant_mode}"
        model_name = f"{Path(model_path.rstrip('/')).name}-{suffix}"
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            num_params=num_params,
            quant_summary=quant_summary,
        )

    def collect_static_activation_scales(
        self,
        prompts: Mapping[str, str],
        *,
        sample_size: int,
        generation_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if sample_size <= 0:
            raise ValueError(f"static activation calibration sample_size must be positive, got {sample_size}.")
        modules = list(iter_real_fp8_linears(self.model))
        if not modules:
            raise RuntimeError("No RealFP8Linear modules found for static activation calibration.")

        selected_items = list(prompts.items())[: min(int(sample_size), len(prompts))]
        if not selected_items:
            raise ValueError("No prompts available for static activation calibration.")
        calib_prompts = dict(selected_items)

        previous_modes = [module.activation_quant_mode for module in modules]
        for module in modules:
            module.reset_activation_scale_observer()
            module.enable_activation_scale_observer(True)
            module.set_activation_quant_mode("dynamic")

        try:
            self.generate(calib_prompts, **dict(generation_kwargs))
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
        finally:
            for module in modules:
                module.enable_activation_scale_observer(False)

        scales = []
        for module in modules:
            scales.append(module.freeze_static_activation_scale_from_observer())
            module.set_activation_quant_mode("static")

        self.latency_records = {}
        self.mfu_stats = {}
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "num_calibration_prompts": len(calib_prompts),
            "num_modules": len(modules),
            "scale_min": min(scales),
            "scale_max": max(scales),
            "scale_mean": sum(scales) / float(len(scales)),
            "previous_modes": sorted(set(previous_modes)),
        }


def require_real_fp8_device(device: str | torch.device) -> None:
    require_fp8_runtime()
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real naive W8A8 requires a CUDA device with torch._scaled_mm FP8 support.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenOneRec-style real naive W8A8 HuggingFace baseline using torch._scaled_mm."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_size", default="full")
    parser.add_argument(
        "--eval_num_shards",
        type=int,
        default=1,
        help=(
            "Split selected evaluation samples into this many deterministic "
            "round-robin shards. Each shard must run in a separate process."
        ),
    )
    parser.add_argument(
        "--eval_shard_id",
        type=int,
        default=0,
        help="Zero-based evaluation shard index used with --eval_num_shards.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=parse_batch_size_arg, default=1)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--prompt_token", default=None)
    parser.add_argument("--output_scores", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--target_regex", default=None)
    parser.add_argument("--skip_regex", default=None)
    parser.add_argument("--use_fast_accum", action="store_true")
    parser.add_argument("--weight_quant_mode", choices=WEIGHT_QUANT_MODES, default="minmax")
    parser.add_argument("--gptq_calib_split", default=None)
    parser.add_argument("--gptq_calib_sample_size", default="1024")
    parser.add_argument("--gptq_layers", default="all", help='Layer spec for GPTQ: "all", "last:K", or "0,2-4".')
    parser.add_argument("--gptq_damp_percent", type=float, default=DEFAULT_GPTQ_DAMP_PERCENT)
    parser.add_argument("--gptq_block_size", type=int, default=DEFAULT_GPTQ_BLOCK_SIZE)
    parser.add_argument("--gptaq_alpha", type=float, default=1.0)
    parser.add_argument(
        "--gptaq_activation_aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use runtime FP8-QDQ activations as the GPTAQ X_hat target.",
    )
    parser.add_argument("--activation_quant_mode", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument(
        "--decode_a16_single_token",
        action="store_true",
        help="Use BF16 activations with quantized-dequantized weights for seq_len=1 decode Linear calls.",
    )
    parser.add_argument(
        "--static_activation_calib_samples",
        type=int,
        default=0,
        help="Number of prompts used to collect per-Linear static activation scales before timed generation.",
    )
    parser.add_argument(
        "--static_activation_calib_split",
        default=None,
        help="Optional split for static activation calibration. Defaults to the evaluation split prompts.",
    )
    parser.add_argument("--profile_fp8", action="store_true", help="Collect torch.profiler CUDA timing for FP8 scopes.")
    parser.add_argument("--profile_fp8_output", default=None, help="Optional JSON path for the FP8 profiler summary.")
    parser.add_argument(
        "--profile_fp8_trace_output",
        default=None,
        help="Optional Chrome trace JSON path exported with torch.profiler.export_chrome_trace.",
    )
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def _batch_size_value(value: int | BatchSizeArg) -> int | str:
    return int(value) if value != "auto" else "auto"


def _profile_time_us(event: Any, *attrs: str) -> float:
    for attr in attrs:
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def summarize_fp8_profile(profiler: Any, *, generate_time_total: float) -> dict[str, Any]:
    keys: dict[str, dict[str, float | int]] = {}
    for event in profiler.key_averages():
        key = str(event.key)
        if not key.startswith("real_fp8/"):
            continue
        cuda_total_us = _profile_time_us(event, "cuda_time_total", "device_time_total")
        self_cuda_us = _profile_time_us(event, "self_cuda_time_total", "self_device_time_total")
        cpu_total_us = _profile_time_us(event, "cpu_time_total")
        keys[key] = {
            "count": int(event.count),
            "cuda_time_ms": cuda_total_us / 1000.0,
            "self_cuda_time_ms": self_cuda_us / 1000.0,
            "cpu_time_ms": cpu_total_us / 1000.0,
        }

    def cuda_ms(key: str) -> float:
        return float(keys.get(key, {}).get("cuda_time_ms", 0.0))

    dynamic_scale_ms = cuda_ms("real_fp8/activation_dynamic_scale")
    static_scale_ms = cuda_ms("real_fp8/activation_static_scale")
    activation_quantize_ms = cuda_ms("real_fp8/activation_quantize")
    scaled_mm_ms = cuda_ms("real_fp8/scaled_mm")
    decode_w8a16_ms = cuda_ms("real_fp8/decode_w8a16_linear")
    bias_reshape_ms = cuda_ms("real_fp8/bias_reshape")
    activation_prepare_ms = dynamic_scale_ms + static_scale_ms + activation_quantize_ms
    measured_fp8_ms = activation_prepare_ms + scaled_mm_ms + decode_w8a16_ms + bias_reshape_ms
    generate_time_ms = float(generate_time_total) * 1000.0

    def share(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0.0 else 0.0

    return {
        "keys": keys,
        "activation_dynamic_scale_ms": dynamic_scale_ms,
        "activation_static_scale_ms": static_scale_ms,
        "activation_quantize_ms": activation_quantize_ms,
        "activation_prepare_ms": activation_prepare_ms,
        "scaled_mm_ms": scaled_mm_ms,
        "decode_w8a16_ms": decode_w8a16_ms,
        "bias_reshape_ms": bias_reshape_ms,
        "measured_fp8_ms": measured_fp8_ms,
        "generate_time_ms": generate_time_ms,
        "activation_prepare_share_of_measured_fp8": share(activation_prepare_ms, measured_fp8_ms),
        "activation_dynamic_scale_share_of_measured_fp8": share(dynamic_scale_ms, measured_fp8_ms),
        "activation_quantize_share_of_measured_fp8": share(activation_quantize_ms, measured_fp8_ms),
        "scaled_mm_share_of_measured_fp8": share(scaled_mm_ms, measured_fp8_ms),
        "decode_w8a16_share_of_measured_fp8": share(decode_w8a16_ms, measured_fp8_ms),
        "activation_prepare_share_of_generate_wall_time": share(activation_prepare_ms, generate_time_ms),
        "scaled_mm_share_of_generate_wall_time": share(scaled_mm_ms, generate_time_ms),
        "decode_w8a16_share_of_generate_wall_time": share(decode_w8a16_ms, generate_time_ms),
    }


def _run_generation_with_optional_profiler(
    generator: HFNaiveW8A8Generator,
    prompts: Mapping[str, str],
    *,
    generation_kwargs: Mapping[str, Any],
    profile_fp8: bool,
    profile_trace_output: str | Path | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any] | None]:
    if not profile_fp8:
        generations, _ = generator.generate(prompts, **dict(generation_kwargs))
        return generations, None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if generator.device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    set_fp8_record_functions_enabled(True)
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
            generations, _ = generator.generate(prompts, **dict(generation_kwargs))
    finally:
        set_fp8_record_functions_enabled(False)
    trace_path = None
    if profile_trace_output is not None:
        trace_path = Path(profile_trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    generate_total = sum(record.generate_time for record in generator.latency_records.values())
    summary = summarize_fp8_profile(prof, generate_time_total=generate_total)
    if trace_path is not None:
        summary["chrome_trace_path"] = str(trace_path)
    return generations, summary


def main() -> None:
    args = parse_args()
    if args.eval_num_shards <= 0:
        raise ValueError(f"--eval_num_shards must be positive, got {args.eval_num_shards}.")
    if args.eval_shard_id < 0 or args.eval_shard_id >= args.eval_num_shards:
        raise ValueError(
            "--eval_shard_id must be in "
            f"[0, {args.eval_num_shards}), got {args.eval_shard_id}."
        )
    if args.eval_num_shards > 1 and args.evaluate:
        raise ValueError(
            "Do not pass --evaluate to an individual shard. Merge all shards with "
            "python -m real_quant.merge_eval_shards, which computes metrics once."
        )
    if args.eval_num_shards > 1 and args.profile_fp8:
        raise ValueError("--profile_fp8 is not supported during sharded evaluation.")
    if args.activation_quant_mode == "static" and args.static_activation_calib_samples <= 0:
        raise ValueError("--activation_quant_mode static requires --static_activation_calib_samples > 0.")

    batch_size, batch_size_config = resolve_batch_size(
        args.batch_size,
        device=args.device,
        model_path=args.model_path,
        task=args.task,
    )
    if batch_size_config.get("auto_batch_size"):
        print(
            "[hf_naive_w8a8] auto batch_size="
            f"{batch_size} (total_memory_gb={batch_size_config.get('auto_batch_total_memory_gb'):.2f}, "
            f"model_size_b={batch_size_config.get('auto_batch_model_size_billions')}, task={args.task})"
        )

    from benchmark.tasks.v1_0.registry import get_task_config

    task_config = get_task_config(args.task)
    generation_config = dict(task_config.get("generation_config", {}))
    prompt_token = args.prompt_token
    if prompt_token is None:
        prompt_token = generation_config.get("prompt_token", "<|sid_begin|>")

    generator = HFNaiveW8A8Generator.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        target_regex=args.target_regex,
        skip_regex=args.skip_regex,
        use_fast_accum=args.use_fast_accum,
        activation_quant_mode=args.activation_quant_mode,
        decode_a16_when_single_token=args.decode_a16_single_token,
        weight_quant_mode=args.weight_quant_mode,
        task=args.task,
        split=args.split,
        data_dir=args.data_dir,
        prompt_token=prompt_token,
        gptq_calib_split=args.gptq_calib_split,
        gptq_calib_sample_size=args.gptq_calib_sample_size,
        gptq_layers=args.gptq_layers,
        gptq_damp_percent=args.gptq_damp_percent,
        gptq_block_size=args.gptq_block_size,
        gptaq_alpha=args.gptaq_alpha,
        gptaq_activation_aware=args.gptaq_activation_aware,
    )
    model_name = str(generator)
    output_root = resolve_repo_path(args.output_dir)
    run_output_dir = eval_run_output_dir(
        output_root,
        num_shards=args.eval_num_shards,
        shard_id=args.eval_shard_id,
    )
    output_file = result_path(str(run_output_dir), model_name, args.task, args.split)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    sample_size = parse_sample_size(args.sample_size)
    unsharded_test_data = load_task_data(
        task_name=args.task,
        data_dir=str(resolve_repo_path(args.data_dir)),
        tokenizer=generator.tokenizer,
        split=args.split,
        sample_size=sample_size,
    )
    unsharded_sample_count = len(unsharded_test_data)
    test_data = select_round_robin_eval_shard(
        unsharded_test_data,
        num_shards=args.eval_num_shards,
        shard_id=args.eval_shard_id,
    )
    if not test_data:
        raise ValueError(
            f"Evaluation shard {args.eval_shard_id}/{args.eval_num_shards} is empty; "
            f"the selected evaluation set contains {unsharded_sample_count} samples."
        )
    shard_description = (
        f" shard={args.eval_shard_id}/{args.eval_num_shards} "
        f"samples={len(test_data)}/{unsharded_sample_count}"
        if args.eval_num_shards > 1
        else ""
    )
    if shard_description:
        print(f"[hf_naive_w8a8] evaluation{shard_description}")
    unsharded_prompts = {
        sample_id: sample["prompt"] for sample_id, sample in unsharded_test_data.items()
    }
    prompts = {sample_id: sample["prompt"] for sample_id, sample in test_data.items()}
    generation_kwargs = {
        "prompt_token": prompt_token,
        "batch_size": batch_size,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "output_scores": args.output_scores,
    }

    static_activation_summary = None
    if args.activation_quant_mode == "static":
        calib_split = args.static_activation_calib_split
        # Every shard must derive identical static scales. When calibration
        # uses the eval split, collect from the original unsharded prefix.
        calib_prompts = unsharded_prompts
        if calib_split is not None and calib_split != args.split:
            calib_size = parse_sample_size(str(args.static_activation_calib_samples))
            calib_data = load_task_data(
                task_name=args.task,
                data_dir=str(resolve_repo_path(args.data_dir)),
                tokenizer=generator.tokenizer,
                split=calib_split,
                sample_size=calib_size,
            )
            calib_prompts = {sample_id: sample["prompt"] for sample_id, sample in calib_data.items()}
        print(
            "[hf_naive_w8a8] collecting static activation scales "
            f"from {args.static_activation_calib_samples} prompts"
        )
        static_activation_summary = generator.collect_static_activation_scales(
            calib_prompts,
            sample_size=args.static_activation_calib_samples,
            generation_kwargs=generation_kwargs,
        )
        print(
            "[hf_naive_w8a8] static activation scale summary: "
            f"modules={static_activation_summary['num_modules']}, "
            f"scale_mean={static_activation_summary['scale_mean']:.6g}, "
            f"scale_max={static_activation_summary['scale_max']:.6g}"
        )

    generations, fp8_profile_summary = _run_generation_with_optional_profiler(
        generator,
        prompts,
        generation_kwargs=generation_kwargs,
        profile_fp8=args.profile_fp8,
        profile_trace_output=args.profile_fp8_trace_output,
    )

    activation_quant_description = (
        "per_token_dynamic_absmax"
        if args.activation_quant_mode == "dynamic"
        else "per_linear_static_absmax_calibrated"
    )
    config = {
        "backend": "hf_real_naive_w8a8_scaled_mm",
        "reference": "OpenOneRec HuggingFace generate with nn.Linear replaced by torch._scaled_mm FP8 wrappers",
        "task": args.task,
        "split": args.split,
        "data_dir": args.data_dir,
        "sample_size": args.sample_size,
        "eval_num_shards": args.eval_num_shards,
        "eval_shard_id": args.eval_shard_id,
        "eval_shard_strategy": "round_robin",
        "eval_unsharded_sample_count": unsharded_sample_count,
        "eval_shard_sample_count": len(test_data),
        "eval_merged": False,
        "dtype": args.dtype,
        "device": args.device,
        "batch_size": batch_size,
        "requested_batch_size": _batch_size_value(args.batch_size),
        **batch_size_config,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "prompt_token": prompt_token,
        "output_scores": args.output_scores,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "fp8_dtype": "float8_e4m3fn",
        "kernel": "torch._scaled_mm_prefill_and_bf16_linear_decode" if args.decode_a16_single_token else "torch._scaled_mm",
        "weight_quant": args.weight_quant_mode,
        "weight_quant_detail": (
            "per_output_channel_absmax"
            if args.weight_quant_mode == "minmax"
            else "gptaq_fp8_per_output_channel_original_absmax_scale"
            if args.weight_quant_mode == "gptaq"
            else "gptq_fp8_per_output_channel_original_absmax_scale"
        ),
        "gptq_calib_split": args.gptq_calib_split,
        "gptq_calib_sample_size": args.gptq_calib_sample_size,
        "gptq_layers": args.gptq_layers,
        "gptq_damp_percent": args.gptq_damp_percent,
        "gptq_block_size": args.gptq_block_size,
        "gptaq_alpha": (
            args.gptaq_alpha
            if args.weight_quant_mode == "gptaq"
            else None
        ),
        "gptaq_activation_aware": (
            args.gptaq_activation_aware
            if args.weight_quant_mode == "gptaq"
            else None
        ),
        "gptaq_activation_target": (
            "runtime_fp8_qdq" if args.gptaq_activation_aware else "propagated_bf16_path"
        ) if args.weight_quant_mode == "gptaq" else None,
        "activation_quant": activation_quant_description,
        "activation_quant_mode": args.activation_quant_mode,
        "decode_a16_single_token": args.decode_a16_single_token,
        "static_activation_calib_samples": args.static_activation_calib_samples,
        "static_activation_calib_split": args.static_activation_calib_split,
        "activation_quant_sharing": "qkv_and_gate_up_shared_input_prefill_only" if args.decode_a16_single_token else "qkv_and_gate_up_shared_input",
        "qmax": FP8_MAX,
        "target_regex": args.target_regex,
        "skip_regex": args.skip_regex,
        "skip_module_names": ["lm_head"],
        "use_fast_accum": args.use_fast_accum,
        "profile_fp8": args.profile_fp8,
        "profile_fp8_trace_output": args.profile_fp8_trace_output,
        "replaced_linears": generator.quant_summary.replaced_linears,
        "skipped_linears": generator.quant_summary.skipped_linears,
        "shared_attention_modules": generator.quant_summary.shared_attention_modules,
        "shared_mlp_modules": generator.quant_summary.shared_mlp_modules,
    }
    if static_activation_summary is not None:
        config["static_activation_summary"] = static_activation_summary
    if fp8_profile_summary is not None:
        config["fp8_profile"] = fp8_profile_summary

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
    if fp8_profile_summary is not None:
        payload["fp8_profile"] = fp8_profile_summary

    save_generation_payload(payload, output_file)
    (output_file.parent / "hf_naive_w8a8_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if fp8_profile_summary is not None:
        profile_output = Path(args.profile_fp8_output) if args.profile_fp8_output else output_file.parent / "fp8_profile.json"
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        profile_output.write_text(json.dumps(fp8_profile_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"FP8 profile summary saved to: {profile_output}")
        if "chrome_trace_path" in fp8_profile_summary:
            print(f"FP8 Chrome trace saved to: {fp8_profile_summary['chrome_trace_path']}")
    print(f"Generation results saved to: {output_file}")
    print(
        "Latency summary: "
        f"generate_total={payload['latency']['generate_time_total']:.3f}s, "
        f"end_to_end_total={payload['latency']['end_to_end_time_total']:.3f}s, "
        f"avg_generate={payload['latency']['generate_time_avg']:.6f}s/sample"
    )
    if fp8_profile_summary is not None:
        print(
            "FP8 profile summary: "
            f"activation_prepare={fp8_profile_summary['activation_prepare_ms']:.3f}ms, "
            f"dynamic_scale={fp8_profile_summary['activation_dynamic_scale_ms']:.3f}ms, "
            f"activation_quantize={fp8_profile_summary['activation_quantize_ms']:.3f}ms, "
            f"scaled_mm={fp8_profile_summary['scaled_mm_ms']:.3f}ms, "
            f"decode_w8a16={fp8_profile_summary['decode_w8a16_ms']:.3f}ms, "
            f"activation_share={fp8_profile_summary['activation_prepare_share_of_measured_fp8']:.3f}"
        )
    print(
        "Quant summary: "
        f"replaced_linears={generator.quant_summary.replaced_linears}, "
        f"skipped_linears={generator.quant_summary.skipped_linears}, "
        f"shared_attention_modules={generator.quant_summary.shared_attention_modules}, "
        f"shared_mlp_modules={generator.quant_summary.shared_mlp_modules}"
    )

    if args.evaluate:
        maybe_evaluate(args.output_dir, args.data_dir, args.overwrite, task_name=args.task)


if __name__ == "__main__":
    main()
