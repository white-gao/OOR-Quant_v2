#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .apply import apply_baseline_qdq
from .quant import (
    FAKE_QUANT_FORWARD_MODE,
    FAKE_QUANT_OPERATOR_DTYPE,
    FAKE_QUANT_QDQ_COMPUTE_DTYPE,
    resolve_weight_quant_scheme,
)
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TASK,
    PROJECT_ROOT,
    TASK_CHOICES,
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
from .search_smoothquant_alpha_mse import _block_mse


SETTINGS: dict[str, tuple[str, str]] = {
    "fp4w_fp4a": ("fp4_e2m1", "fp4_e2m1"),
    "fp4w_fp8a": ("fp4_e2m1", "fp8_e4m3fn"),
    "int4w_int4a": ("int4", "int4"),
    "fp4w_bf16a": ("fp4_e2m1", "none"),
    "bf16w_fp4a": ("none", "fp4_e2m1"),
}
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts/results/fake_quant/probes/fp4_int4_component_ablation_ad_calib128"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selectable FP4/INT4/FP8 weight-activation settings using "
            "local transformer-block MSE on identical full-precision inputs."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", choices=TASK_CHOICES, default=DEFAULT_TASK)
    parser.add_argument(
        "--calib_split",
        default="auto",
        help="Calibration split, or auto to prefer <task>_calib.parquet.",
    )
    parser.add_argument("--calib_sample_size", type=int, default=128)
    parser.add_argument("--calib_offset", type=int, default=0)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--settings",
        default="all",
        help="Comma-separated setting names, or all. Available: " + ",".join(SETTINGS),
    )
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
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _format_label(quant_format: str) -> str:
    labels = {
        "fp4_e2m1": "FP4 E2M1",
        "fp8_e4m3fn": "FP8 E4M3",
        "int4": "INT4",
        "none": "BF16",
    }
    return labels[quant_format]


def _setting_label(setting: str) -> str:
    weight_format, activation_format = SETTINGS[setting]
    return f"{_format_label(weight_format)}-W / {_format_label(activation_format)}-A"


def _resolve_setting_names(spec: str) -> list[str]:
    if spec == "all":
        return list(SETTINGS)
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("--settings must select at least one setting.")
    unknown = [name for name in names if name not in SETTINGS]
    if unknown:
        raise ValueError(f"Unknown settings {unknown}; available: {list(SETTINGS)}")
    return names


def _markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    setting_names = result["settings"]
    lines = [
        "# Selected Weight-Activation Format Local MSE",
        "",
        (
            f"All {len(setting_names)} selected settings use the same calibration samples, BF16 model "
            "execution, per-output-channel weight QDQ, dynamic per-token "
            "activation QDQ, and shared-input activation quantization."
        ),
        "",
        (
            "Each quantized block receives the corresponding full-precision "
            "block input. Therefore this report isolates local block "
            "reconstruction error and does not propagate quantization error "
            "between layers."
        ),
        "",
        "| setting | weight scheme | global MSE | relative MSE | mean layer MSE | last layer MSE |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for setting in setting_names:
        item = summary[setting]
        lines.append(
            "| {label} | {scheme} | {global_mse:.8e} | {relative_mse:.8e} | "
            "{mean_layer_mse:.8e} | {last_layer_mse:.8e} |".format(
                label=_setting_label(setting),
                scheme=item["weight_quant_scheme"],
                **item,
            )
        )

    winner = min(setting_names, key=lambda name: float(summary[name]["global_mse"]))
    lines.extend(
        [
            "",
            f"Lower global local MSE: **{_setting_label(winner)}**.",
            "",
            "## Per-layer local block MSE",
            "",
            "| layer | " + " | ".join(_setting_label(name) for name in setting_names) + " |",
            "|---:|" + "---:|" * len(setting_names),
        ]
    )
    for layer_idx in result["layers"]:
        layer = result["per_layer"][str(layer_idx)]
        values = " | ".join(
            f"{float(layer[setting]['mse']):.8e}" for setting in setting_names
        )
        lines.append(f"| {layer_idx} | {values} |")
    lines.extend(
        [
            "",
            (
                "This is a reconstruction-error probe, not a recommendation "
                "quality result. If the settings are close in local MSE, "
                "the next useful check is propagated final-block or logit MSE."
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
    setting_names = _resolve_setting_names(args.settings)
    active_settings = {name: SETTINGS[name] for name in setting_names}

    set_seed(args.seed)
    output_dir = resolve_repo_path(args.output_dir)
    json_path = output_dir / "quant_format_mse.json"
    markdown_path = output_dir / "quant_format_mse.md"
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
        setting: {
            "squared_error": 0.0,
            "target_squared_sum": 0.0,
            "elements": 0,
            "layer_mse_sum": 0.0,
            "last_layer_mse": None,
        }
        for setting in active_settings
    }

    for layer_idx in layer_indices:
        while stream_layer_idx < layer_idx:
            fp_inputs = advance_layer_input_batches(
                layer=layers[stream_layer_idx],
                batches=fp_inputs,
            )
            stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        fp_targets = advance_layer_input_batches(
            layer=teacher_block,
            batches=fp_inputs,
        )
        layer_result: dict[str, Any] = {}

        for setting, (weight_format, activation_format) in active_settings.items():
            effective_act_quant_mode = (
                args.act_quant_mode if activation_format != "none" else "per_linear"
            )
            quant_block = copy.deepcopy(teacher_block)
            quant_summary = apply_baseline_qdq(
                quant_block,
                weight_quant_format=weight_format,
                activation_quant_format=activation_format,
                act_quant_mode=effective_act_quant_mode,
                skip_module_names=(),
            )
            metric = _block_mse(quant_block, fp_inputs, fp_targets)
            metric.update(
                {
                    "weight_quant_format": weight_format,
                    "activation_quant_format": activation_format,
                    "weight_quant_scheme": resolve_weight_quant_scheme(weight_format),
                    "act_quant_mode": effective_act_quant_mode,
                    "replaced_linears": quant_summary.replaced_linears,
                    "skipped_linears": quant_summary.skipped_linears,
                    "shared_attention_modules": quant_summary.shared_attention_modules,
                    "shared_mlp_modules": quant_summary.shared_mlp_modules,
                }
            )
            layer_result[setting] = metric

            total = totals[setting]
            total["squared_error"] += float(metric["squared_error"])
            total["target_squared_sum"] += float(metric["target_squared_sum"])
            total["elements"] += int(metric["elements"])
            total["layer_mse_sum"] += float(metric["mse"])
            if layer_idx == layer_indices[-1]:
                total["last_layer_mse"] = float(metric["mse"])

            print(
                f"[w4a4_format_mse] layer={layer_idx} "
                f"setting={setting} "
                f"weight_format={weight_format} "
                f"activation_format={activation_format} "
                f"weight_scheme={metric['weight_quant_scheme']} "
                f"mse={float(metric['mse']):.8e} "
                f"relative_mse={float(metric['relative_mse']):.8e}"
            )
            del quant_block

        per_layer[str(layer_idx)] = layer_result
        fp_inputs = fp_targets
        stream_layer_idx = layer_idx + 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary: dict[str, dict[str, Any]] = {}
    for setting, (weight_format, activation_format) in active_settings.items():
        total = totals[setting]
        elements = int(total["elements"])
        last_layer_mse = total["last_layer_mse"]
        if elements <= 0 or last_layer_mse is None:
            raise RuntimeError(f"Incomplete MSE totals for {setting}.")
        squared_error = float(total["squared_error"])
        target_squared_sum = float(total["target_squared_sum"])
        summary[setting] = {
            "weight_quant_format": weight_format,
            "activation_quant_format": activation_format,
            "weight_quant_scheme": resolve_weight_quant_scheme(weight_format),
            "global_mse": squared_error / elements,
            "relative_mse": squared_error / max(target_squared_sum, 1e-30),
            "mean_layer_mse": float(total["layer_mse_sum"]) / len(layer_indices),
            "last_layer_mse": float(last_layer_mse),
            "squared_error": squared_error,
            "target_squared_sum": target_squared_sum,
            "elements": elements,
        }

    result = {
        "method": "selected_format_local_block_mse",
        "criterion": "local_block_output_mse_on_fp_inputs",
        "model_path": str(model_path),
        "task": args.task,
        "calib_split": calibration_split,
        "calib_sample_size": len(calibration_data),
        "calib_offset": args.calib_offset,
        "layers": layer_indices,
        "settings": setting_names,
        "quantization": {
            "settings": {
                setting: {
                    "weight_format": weight_format,
                    "activation_format": activation_format,
                    "weight_quant_scheme": resolve_weight_quant_scheme(weight_format),
                    "act_quant_mode": (
                        args.act_quant_mode if activation_format != "none" else "per_linear"
                    ),
                }
                for setting, (weight_format, activation_format) in active_settings.items()
            },
            "weight_granularity": "per_output_channel",
            "activation_granularity": "dynamic_per_token",
            "act_quant_mode": args.act_quant_mode,
            "weight_schemes": {
                setting: resolve_weight_quant_scheme(weight_format)
                for setting, (weight_format, _) in active_settings.items()
            },
        },
        "numeric_contract": {
            "forward_mode": FAKE_QUANT_FORWARD_MODE,
            "operator_dtype": FAKE_QUANT_OPERATOR_DTYPE,
            "qdq_compute_dtype": FAKE_QUANT_QDQ_COMPUTE_DTYPE,
            "model_dtype": args.dtype,
        },
        "summary": summary,
        "per_layer": per_layer,
    }
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")

    winner = min(setting_names, key=lambda name: float(summary[name]["global_mse"]))
    for setting in setting_names:
        print(
            f"[w4a4_format_mse] setting={setting} "
            f"global_mse={float(summary[setting]['global_mse']):.8e} "
            f"relative_mse={float(summary[setting]['relative_mse']):.8e}"
        )
    print(f"[w4a4_format_mse] winner={winner}")
    print(f"[w4a4_format_mse] json={json_path}")
    print(f"[w4a4_format_mse] markdown={markdown_path}")


if __name__ == "__main__":
    main()
