#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .apply import apply_baseline_qdq, install_shared_input_activation_quantization
from .quant import (
    FAKE_QUANT_FORWARD_MODE,
    FAKE_QUANT_OPERATOR_DTYPE,
    FAKE_QUANT_QDQ_COMPUTE_DTYPE,
)
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TASK,
    PROJECT_ROOT,
    TASK_CHOICES,
    _first_tensor,
    advance_layer_input_batches,
    build_model_batches,
    capture_layer_input_batches,
    default_calib_split,
    dtype_from_name,
    format_prompt,
    get_task_config,
    get_transformer_layers,
    load_task_data,
    parse_layer_indices,
    resolve_input_device,
    resolve_repo_path,
    set_seed,
)
from .support.runtime_utils import _module_device, _move_tree_to_device
from .support.smoothquant_runtime import (
    DEFAULT_SMOOTH_FOLD,
    DEFAULT_SMOOTHQUANT_MAX_SCALE,
    DEFAULT_SMOOTHQUANT_MIN_SCALE,
    DEFAULT_SMOOTH_SCOPE,
    _batch_to_args_kwargs,
    collect_smoothquant_statistics,
    fold_smoothquant_scales_inplace,
    smoothquant_quantized_module_from_scales,
    smoothquant_scales_from_statistics,
)


DEFAULT_ALPHAS = tuple(index / 10.0 for index in range(11))
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts/results/fake_quant/smoothquant_alpha_mse_fp8w8a8_ad_calib128"
)


def _parse_alphas(value: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        alpha = float(item)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise argparse.ArgumentTypeError(
                f"SmoothQuant alpha must be finite and in [0, 1], got {item!r}."
            )
        if alpha not in parsed:
            parsed.append(alpha)
    if not parsed:
        raise argparse.ArgumentTypeError("At least one alpha is required.")
    return tuple(parsed)


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "off"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"Expected a finite float or none, got {value!r}.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search FP8-W/FP8-A SmoothQuant alpha using local transformer-block "
            "output reconstruction MSE on a fixed calibration subset."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", choices=TASK_CHOICES, default=DEFAULT_TASK)
    parser.add_argument(
        "--calib_split",
        default="auto",
        help="Calibration split name, or auto to prefer <task>_calib.parquet.",
    )
    parser.add_argument("--calib_sample_size", type=int, default=128)
    parser.add_argument("--calib_offset", type=int, default=0)
    parser.add_argument(
        "--alphas",
        type=_parse_alphas,
        default=DEFAULT_ALPHAS,
        help="Comma-separated candidates; default: 0.0,0.1,...,1.0.",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--act_quant_mode",
        choices=("per_linear", "shared_input"),
        default="shared_input",
    )
    parser.add_argument(
        "--smooth_scope",
        choices=("all", "omni"),
        default=DEFAULT_SMOOTH_SCOPE,
    )
    parser.add_argument(
        "--smooth_fold",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SMOOTH_FOLD,
    )
    parser.add_argument(
        "--smoothquant_min_scale",
        type=_optional_float,
        default=DEFAULT_SMOOTHQUANT_MIN_SCALE,
    )
    parser.add_argument(
        "--smoothquant_max_scale",
        type=_optional_float,
        default=DEFAULT_SMOOTHQUANT_MAX_SCALE,
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _batch_hidden_states(batch: Any) -> torch.Tensor:
    args, kwargs = _batch_to_args_kwargs(batch)
    if args and torch.is_tensor(args[0]):
        return args[0]
    hidden_states = kwargs.get("hidden_states")
    if torch.is_tensor(hidden_states):
        return hidden_states
    raise TypeError("Could not find hidden_states in an advanced transformer-block batch.")


def _block_mse(
    block: torch.nn.Module,
    inputs: Sequence[Any],
    targets: Sequence[Any],
) -> dict[str, float | int]:
    if len(inputs) != len(targets):
        raise ValueError("Input and target batch counts must match.")
    device = _module_device(block)
    squared_error = 0.0
    target_square = 0.0
    elements = 0
    was_training = block.training
    block.eval()
    try:
        with torch.no_grad():
            for input_batch, target_batch in zip(inputs, targets):
                args, kwargs = _batch_to_args_kwargs(input_batch)
                args = _move_tree_to_device(args, device)
                kwargs = _move_tree_to_device(kwargs, device)
                prediction = _first_tensor(block(*args, **kwargs))
                target = _batch_hidden_states(target_batch).to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                difference = prediction.float() - target.float()
                squared_error += float(difference.square().sum())
                target_square += float(target.float().square().sum())
                elements += difference.numel()
    finally:
        block.train(was_training)
    if elements == 0:
        raise ValueError("No block-output elements were evaluated.")
    return {
        "squared_error": squared_error,
        "target_squared_sum": target_square,
        "elements": elements,
        "mse": squared_error / elements,
        "relative_mse": squared_error / max(target_square, 1e-30),
    }


def _alpha_key(alpha: float) -> str:
    return format(alpha, ".8g")


def _markdown_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    baseline = result["no_sq_rtn"]
    best = result["best_by_global_mse"]
    lines = [
        "# SmoothQuant FP8 W8A8 Alpha MSE Search",
        "",
        (
            "This search uses the same fixed calibration samples for every alpha and "
            "for the no-SmoothQuant FP8 W8A8 RTN control. Each quantized block is "
            "compared with its FP block using identical FP block inputs; errors are "
            "not propagated between quantized layers."
        ),
        "",
        "Alpha=0 is still a SmoothQuant transform and is not the no-SQ control.",
        "",
        (
            "| setting | global local MSE | delta vs no-SQ | relative MSE | "
            "mean layer MSE | last layer MSE |"
        ),
        "|:---|---:|---:|---:|---:|---:|",
        (
            "| no_sq_rtn | {global_mse:.8e} | 0.0000% | {relative_mse:.8e} | "
            "{mean_layer_mse:.8e} | {last_layer_mse:.8e} |"
        ).format(**baseline),
    ]
    for item in summary:
        lines.append(
            "| alpha={alpha:.6g} | {global_mse:.8e} | "
            "{global_mse_change_percent:+.4f}% | {relative_mse:.8e} | "
            "{mean_layer_mse:.8e} | {last_layer_mse:.8e} |".format(**item)
        )
    lines.extend(
        [
            "",
            (
                "Best alpha by global local MSE: "
                f"{best['alpha']:.6g} ({best['global_mse']:.8e}, "
                f"{best['global_mse_change_percent']:+.4f}% vs no-SQ RTN)."
            ),
            "",
            (
                "MSE is a calibration proxy. The selected alpha still needs one "
                "recommendation evaluation before it is treated as the final setting."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.calib_sample_size <= 0:
        raise ValueError("--calib_sample_size must be positive.")
    if args.calib_offset < 0:
        raise ValueError("--calib_offset must be non-negative.")
    if (
        args.smoothquant_min_scale is not None
        and args.smoothquant_max_scale is not None
        and args.smoothquant_min_scale > args.smoothquant_max_scale
    ):
        raise ValueError("smoothquant_min_scale must not exceed smoothquant_max_scale.")

    set_seed(args.seed)
    output_dir = resolve_repo_path(args.output_dir)
    json_path = output_dir / "smoothquant_alpha_mse.json"
    markdown_path = output_dir / "smoothquant_alpha_mse.md"
    for path in (json_path, markdown_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_repo_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    calibration_split = (
        default_calib_split(args.data_dir, "test", task_name=args.task)
        if args.calib_split == "auto"
        else args.calib_split
    )
    calibration_data = load_task_data(
        task_name=args.task,
        tokenizer=tokenizer,
        data_dir=str(resolve_repo_path(args.data_dir)),
        split=calibration_split,
        sample_size=args.calib_sample_size,
        sample_offset=args.calib_offset,
        require_answer=False,
        drop_final_assistant=True,
    )
    if not calibration_data:
        raise ValueError("No calibration samples were loaded.")

    task_config = get_task_config(args.task)
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "")
    prompts = [
        format_prompt(str(sample["prompt"]), prompt_token)
        for sample in calibration_data.values()
    ]
    model_batches = build_model_batches(
        tokenizer=tokenizer,
        prompts=prompts,
        device=input_device,
    )

    first_layer_idx = layer_indices[0]
    fp_inputs = capture_layer_input_batches(
        model=model,
        layer=layers[first_layer_idx],
        model_batches=model_batches,
    )
    stream_layer_idx = first_layer_idx
    per_layer: dict[str, dict[str, Any]] = {}
    totals = {
        alpha: {
            "squared_error": 0.0,
            "target_squared_sum": 0.0,
            "elements": 0,
            "layer_mse_sum": 0.0,
            "last_layer_mse": None,
        }
        for alpha in args.alphas
    }
    no_sq_total: dict[str, float | int | None] = {
        "squared_error": 0.0,
        "target_squared_sum": 0.0,
        "elements": 0,
        "layer_mse_sum": 0.0,
        "last_layer_mse": None,
    }

    for layer_idx in layer_indices:
        while stream_layer_idx < layer_idx:
            fp_inputs = advance_layer_input_batches(
                layer=layers[stream_layer_idx],
                batches=fp_inputs,
            )
            stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        statistics = collect_smoothquant_statistics(
            teacher_block,
            fp_inputs,
            smooth_scope=args.smooth_scope,
        )
        fp_targets = advance_layer_input_batches(
            layer=teacher_block,
            batches=fp_inputs,
        )
        layer_result: dict[str, Any] = {}

        no_sq_block = copy.deepcopy(teacher_block)
        no_sq_quant_summary = apply_baseline_qdq(
            no_sq_block,
            weight_quant_format="fp8_e4m3fn",
            weight_quant_scheme="symmetric",
            activation_quant_format="fp8_e4m3fn",
            act_quant_mode=args.act_quant_mode,
            skip_module_names=(),
        )
        no_sq_metric = _block_mse(no_sq_block, fp_inputs, fp_targets)
        no_sq_metric.update(
            {
                "replaced_linears": no_sq_quant_summary.replaced_linears,
                "skipped_linears": no_sq_quant_summary.skipped_linears,
                "shared_attention_modules": no_sq_quant_summary.shared_attention_modules,
                "shared_mlp_modules": no_sq_quant_summary.shared_mlp_modules,
            }
        )
        layer_result["no_sq_rtn"] = no_sq_metric
        no_sq_total["squared_error"] = float(no_sq_total["squared_error"]) + float(
            no_sq_metric["squared_error"]
        )
        no_sq_total["target_squared_sum"] = float(
            no_sq_total["target_squared_sum"]
        ) + float(no_sq_metric["target_squared_sum"])
        no_sq_total["elements"] = int(no_sq_total["elements"]) + int(
            no_sq_metric["elements"]
        )
        no_sq_total["layer_mse_sum"] = float(no_sq_total["layer_mse_sum"]) + float(
            no_sq_metric["mse"]
        )
        if layer_idx == layer_indices[-1]:
            no_sq_total["last_layer_mse"] = float(no_sq_metric["mse"])
        print(
            f"[sq_alpha_mse] layer={layer_idx} setting=no_sq_rtn "
            f"mse={float(no_sq_metric['mse']):.8e} "
            f"relative_mse={float(no_sq_metric['relative_mse']):.8e}"
        )
        del no_sq_block

        for alpha in args.alphas:
            scales = smoothquant_scales_from_statistics(
                statistics,
                alpha=alpha,
                min_scale=args.smoothquant_min_scale,
                max_scale=args.smoothquant_max_scale,
            )
            quant_block = copy.deepcopy(teacher_block)
            folded_names = (
                fold_smoothquant_scales_inplace(
                    quant_block,
                    scales,
                    smooth_scope=args.smooth_scope,
                )
                if args.smooth_fold
                else set()
            )
            quant_block, replaced = smoothquant_quantized_module_from_scales(
                quant_block,
                scales,
                act_quant="per_token",
                smooth_scope=args.smooth_scope,
                folded_names=folded_names,
            )
            shared_attention_modules = 0
            shared_mlp_modules = 0
            if args.act_quant_mode == "shared_input":
                (
                    shared_attention_modules,
                    shared_mlp_modules,
                ) = install_shared_input_activation_quantization(quant_block)

            metric = _block_mse(quant_block, fp_inputs, fp_targets)
            metric.update(
                {
                    "replaced_linears": replaced,
                    "folded_linears": len(folded_names),
                    "shared_attention_modules": shared_attention_modules,
                    "shared_mlp_modules": shared_mlp_modules,
                }
            )
            layer_result[_alpha_key(alpha)] = metric
            total = totals[alpha]
            total["squared_error"] += float(metric["squared_error"])
            total["target_squared_sum"] += float(metric["target_squared_sum"])
            total["elements"] += int(metric["elements"])
            total["layer_mse_sum"] += float(metric["mse"])
            if layer_idx == layer_indices[-1]:
                total["last_layer_mse"] = float(metric["mse"])
            print(
                f"[sq_alpha_mse] layer={layer_idx} alpha={alpha:.6g} "
                f"mse={float(metric['mse']):.8e} "
                f"relative_mse={float(metric['relative_mse']):.8e}"
            )
            del quant_block

        per_layer[str(layer_idx)] = layer_result
        fp_inputs = fp_targets
        stream_layer_idx = layer_idx + 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    no_sq_elements = int(no_sq_total["elements"])
    no_sq_squared_error = float(no_sq_total["squared_error"])
    no_sq_target_squared_sum = float(no_sq_total["target_squared_sum"])
    no_sq_last_layer_mse = no_sq_total["last_layer_mse"]
    if no_sq_elements <= 0 or no_sq_last_layer_mse is None:
        raise RuntimeError("Incomplete no-SmoothQuant RTN totals.")
    no_sq_summary: dict[str, float | int | str] = {
        "setting": "no_sq_rtn",
        "global_mse": no_sq_squared_error / no_sq_elements,
        "relative_mse": no_sq_squared_error / max(no_sq_target_squared_sum, 1e-30),
        "mean_layer_mse": float(no_sq_total["layer_mse_sum"]) / len(layer_indices),
        "last_layer_mse": float(no_sq_last_layer_mse),
        "squared_error": no_sq_squared_error,
        "target_squared_sum": no_sq_target_squared_sum,
        "elements": no_sq_elements,
    }

    summary: list[dict[str, float | int]] = []
    for alpha in args.alphas:
        total = totals[alpha]
        elements = int(total["elements"])
        squared_error = float(total["squared_error"])
        target_squared_sum = float(total["target_squared_sum"])
        last_layer_mse = total["last_layer_mse"]
        if elements <= 0 or last_layer_mse is None:
            raise RuntimeError("Incomplete SmoothQuant alpha search totals.")
        global_mse = squared_error / elements
        no_sq_global_mse = float(no_sq_summary["global_mse"])
        no_sq_denominator = max(no_sq_global_mse, 1e-30)
        summary.append(
            {
                "alpha": alpha,
                "global_mse": global_mse,
                "global_mse_delta_vs_no_sq": global_mse - no_sq_global_mse,
                "global_mse_ratio_vs_no_sq": global_mse / no_sq_denominator,
                "global_mse_change_percent": (
                    global_mse / no_sq_denominator - 1.0
                ) * 100.0,
                "relative_mse": squared_error / max(target_squared_sum, 1e-30),
                "mean_layer_mse": float(total["layer_mse_sum"]) / len(layer_indices),
                "last_layer_mse": float(last_layer_mse),
                "squared_error": squared_error,
                "target_squared_sum": target_squared_sum,
                "elements": elements,
            }
        )
    best = min(summary, key=lambda item: float(item["global_mse"]))

    result = {
        "method": "smoothquant_alpha_mse_search",
        "quantization": {
            "weight_quant_format": "fp8_e4m3fn",
            "weight_granularity": "per_output_channel",
            "activation_quant_format": "fp8_e4m3fn",
            "activation_granularity": "dynamic_per_token",
            "act_quant_mode": args.act_quant_mode,
            "smooth_scope": args.smooth_scope,
            "smooth_fold": args.smooth_fold,
            "smoothquant_min_scale": args.smoothquant_min_scale,
            "smoothquant_max_scale": args.smoothquant_max_scale,
        },
        "numeric_contract": {
            "forward_mode": FAKE_QUANT_FORWARD_MODE,
            "operator_dtype": FAKE_QUANT_OPERATOR_DTYPE,
            "qdq_compute_dtype": FAKE_QUANT_QDQ_COMPUTE_DTYPE,
            "model_dtype": args.dtype,
        },
        "criterion": "local_block_output_mse_on_fp_inputs",
        "model_path": str(model_path),
        "task": args.task,
        "calib_split": calibration_split,
        "calib_sample_size": len(calibration_data),
        "calib_offset": args.calib_offset,
        "layers": layer_indices,
        "alphas": list(args.alphas),
        "per_layer": per_layer,
        "no_sq_rtn": no_sq_summary,
        "summary": summary,
        "best_by_global_mse": best,
    }
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(
        f"[sq_alpha_mse] no_sq_rtn_global_mse="
        f"{float(no_sq_summary['global_mse']):.8e}"
    )
    print(
        f"[sq_alpha_mse] best_alpha={float(best['alpha']):.6g} "
        f"global_mse={float(best['global_mse']):.8e} "
        f"change_vs_no_sq={float(best['global_mse_change_percent']):+.4f}%"
    )
    print(f"[sq_alpha_mse] json={json_path}")
    print(f"[sq_alpha_mse] markdown={markdown_path}")


if __name__ == "__main__":
    main()
