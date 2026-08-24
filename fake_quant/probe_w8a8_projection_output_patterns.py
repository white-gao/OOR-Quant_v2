from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import TwoSlopeNorm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .apply import apply_baseline_qdq
from .probe_activation_quant_patterns import _tensor_summary
from .quant import QuantFormat, validate_quant_format
from .run_m1_onerec_ad import (
    DEFAULT_MODEL_PATH,
    dtype_from_name,
    get_transformer_layers,
    resolve_input_device,
    resolve_repo_path,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BF16 and end-to-end fake-W8A8 layer-27 q_proj/o_proj "
            "outputs on prompts selected by the activation-input probe."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--input_probe_dir",
        default=(
            "artifacts/results/fake_quant/probes/"
            "activation_quant_patterns_layer27_ad_fp8"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "artifacts/results/fake_quant/probes/"
            "w8a8_projection_output_patterns_layer27_ad_fp8"
        ),
    )
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--channel_slice_size", type=int, default=512)
    parser.add_argument(
        "--weight_quant_format",
        default="fp8_e4m3fn",
        choices=("fp8_e4m3fn", "int8"),
    )
    parser.add_argument(
        "--activation_quant_format",
        default="fp8_e4m3fn",
        choices=("fp8_e4m3fn", "int8"),
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _centered_norm(values: np.ndarray) -> tuple[float, float, TwoSlopeNorm | None]:
    value_min = float(values.min())
    value_max = float(values.max())
    if value_min < 0.0 < value_max:
        return value_min, value_max, TwoSlopeNorm(vmin=value_min, vcenter=0.0, vmax=value_max)
    if value_min == value_max:
        delta = max(abs(value_min) * 1e-6, 1e-12)
        return value_min - delta, value_max + delta, None
    return value_min, value_max, None


def _plot_output_slice(
    *,
    bf16: np.ndarray,
    w8a8: np.ndarray,
    error: np.ndarray,
    sample_id: str,
    projection: str,
    layer_idx: int,
    channel_start: int,
    channel_end: int,
    output_path: Path,
) -> None:
    bf16_slice = bf16[:, channel_start:channel_end]
    w8a8_slice = w8a8[:, channel_start:channel_end]
    error_slice = error[:, channel_start:channel_end]
    shared_values = np.concatenate((bf16_slice.reshape(-1), w8a8_slice.reshape(-1)))
    shared_min, shared_max, shared_norm = _centered_norm(shared_values)
    error_min, error_max, error_norm = _centered_norm(error_slice)

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), constrained_layout=True)
    panels = (
        (bf16_slice, "BF16 full-model output", shared_min, shared_max, shared_norm),
        (w8a8_slice, "end-to-end W8A8 output", shared_min, shared_max, shared_norm),
        (error_slice, "W8A8 - BF16 output error", error_min, error_max, error_norm),
    )
    for axis, (values, title, value_min, value_max, norm) in zip(axes, panels):
        image_kwargs: dict[str, Any] = {
            "aspect": "auto",
            "interpolation": "nearest",
            "origin": "upper",
            "cmap": "RdBu_r",
            "extent": (channel_start, channel_end, values.shape[0], 0),
        }
        if norm is None:
            image_kwargs.update(vmin=value_min, vmax=value_max)
        else:
            image_kwargs["norm"] = norm
        image = axis.imshow(values, **image_kwargs)
        axis.set_title(
            f"{title} | true min={float(values.min()):.6g}, "
            f"true max={float(values.max()):.6g}"
        )
        axis.set_ylabel("token position")
        axis.set_xlabel("output channel")
        fig.colorbar(image, ax=axis, fraction=0.018, pad=0.01)
    fig.suptitle(
        f"sample={sample_id} layer={layer_idx} {projection} output "
        f"channels=[{channel_start}, {channel_end})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _capture_projection_outputs(
    *,
    model: nn.Module,
    layer_idx: int,
    encoded: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    layers = get_transformer_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer={layer_idx} is out of range for {len(layers)} blocks.")
    attention = layers[layer_idx].self_attn
    captured: dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            if name in captured:
                raise RuntimeError(f"{name} was invoked more than once for one prompt.")
            captured[name] = output.detach().squeeze(0).float().cpu().numpy()

        return hook

    handles = (
        attention.q_proj.register_forward_hook(make_hook("q_proj")),
        attention.o_proj.register_forward_hook(make_hook("o_proj")),
    )
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"q_proj", "o_proj"}:
        raise RuntimeError(f"Expected q_proj/o_proj captures, got {sorted(captured)}.")
    return captured


def _load_selected_prompts(input_probe_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary_path = input_probe_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing input-probe summary: {summary_path}. Run the activation-input probe first."
        )
    input_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected: list[dict[str, str]] = []
    for sample in input_summary.get("samples", []):
        sample_order = int(sample["sample_order"])
        sample_id = str(sample["sample_id"])
        sample_dir = input_probe_dir / f"sample_{sample_order:02d}_id_{sample_id}"
        prompt_path = sample_dir / "compressed_prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing compressed prompt: {prompt_path}")
        selected.append(
            {
                "sample_order": str(sample_order),
                "sample_id": sample_id,
                "prompt": prompt_path.read_text(encoding="utf-8"),
            }
        )
    if not selected:
        raise ValueError(f"No selected samples found in {summary_path}.")
    return input_summary, selected


def _run_model_captures(
    *,
    model: nn.Module,
    tokenizer: Any,
    selected: list[dict[str, str]],
    layer_idx: int,
    input_device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for sample in selected:
        encoded = tokenizer(sample["prompt"], return_tensors="pt")
        encoded = {
            key: value.to(input_device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        results[sample["sample_id"]] = _capture_projection_outputs(
            model=model,
            layer_idx=layer_idx,
            encoded=encoded,
        )
    return results


def main() -> None:
    args = parse_args()
    if args.channel_slice_size <= 0:
        raise ValueError("channel_slice_size must be positive.")
    weight_format: QuantFormat = validate_quant_format(args.weight_quant_format)
    activation_format: QuantFormat = validate_quant_format(args.activation_quant_format)
    input_probe_dir = resolve_repo_path(args.input_probe_dir)
    output_dir = resolve_repo_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_summary, selected = _load_selected_prompts(input_probe_dir)
    set_seed(args.seed)
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

    print("[w8a8-output-pattern] capturing BF16 projection outputs")
    bf16_captures = _run_model_captures(
        model=model,
        tokenizer=tokenizer,
        selected=selected,
        layer_idx=args.layer,
        input_device=input_device,
    )

    print("[w8a8-output-pattern] applying end-to-end fake W8A8 RTN")
    quant_summary = apply_baseline_qdq(
        model,
        weight_quant_format=weight_format,
        weight_group_size=None,
        activation_quant_format=activation_format,
        # Numerically equivalent to shared-input for RTN, while normal module
        # calls keep q_proj/o_proj forward hooks active.
        act_quant_mode="per_linear",
    )
    model.eval()
    torch.cuda.empty_cache()
    print("[w8a8-output-pattern] capturing end-to-end W8A8 projection outputs")
    w8a8_captures = _run_model_captures(
        model=model,
        tokenizer=tokenizer,
        selected=selected,
        layer_idx=args.layer,
        input_device=input_device,
    )

    output_summary: dict[str, Any] = {
        "schema_version": 1,
        "task": "ad",
        "layer": args.layer,
        "projections": ["q_proj", "o_proj"],
        "reference": "BF16 full-model trajectory",
        "candidate": "end-to-end fake W8A8 RTN full-model trajectory",
        "weight_quant_format": weight_format,
        "weight_quant_granularity": "per-output-channel",
        "activation_quant_format": activation_format,
        "activation_quant_granularity": "dynamic per-token",
        "activation_quant_mode": (
            "per-linear hook-compatible execution; numerically equivalent to shared-input for RTN"
        ),
        "quantized_linears": quant_summary.replaced_linears,
        "skipped_linears": quant_summary.skipped_linears,
        "error_scope": (
            "candidate-reference output difference including upstream accumulated W8A8 error, "
            "current-linear weight QDQ, and current-linear activation QDQ"
        ),
        "channel_slice_size": args.channel_slice_size,
        "plot_scaling": (
            "exact true extrema with no percentile clipping; BF16/W8A8 panels share their "
            "joint per-slice range; error uses its exact per-slice range"
        ),
        "source_input_probe": str(input_probe_dir),
        "source_selected_pool_positions": input_summary.get("selected_pool_positions", []),
        "samples": [],
    }

    for sample in selected:
        sample_order = int(sample["sample_order"])
        sample_id = sample["sample_id"]
        sample_dir = output_dir / f"sample_{sample_order:02d}_id_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_summary: dict[str, Any] = {
            "sample_order": sample_order,
            "sample_id": sample_id,
            "projections": {},
        }
        for projection in ("q_proj", "o_proj"):
            reference = bf16_captures[sample_id][projection]
            candidate = w8a8_captures[sample_id][projection]
            if reference.shape != candidate.shape:
                raise RuntimeError(
                    f"Shape mismatch for sample={sample_id} {projection}: "
                    f"BF16={reference.shape}, W8A8={candidate.shape}."
                )
            error = candidate - reference
            np.savez_compressed(
                sample_dir / f"{projection}_output_matrices.npz",
                bf16=reference,
                w8a8=candidate,
                error=error,
            )
            projection_summary = _tensor_summary(reference, candidate)
            projection_summary["channel_slices"] = []
            for channel_start in range(0, reference.shape[-1], args.channel_slice_size):
                channel_end = min(channel_start + args.channel_slice_size, reference.shape[-1])
                filename = (
                    f"layer_{args.layer:02d}_{projection}_output_channels_"
                    f"{channel_start:04d}_{channel_end:04d}.png"
                )
                _plot_output_slice(
                    bf16=reference,
                    w8a8=candidate,
                    error=error,
                    sample_id=sample_id,
                    projection=projection,
                    layer_idx=args.layer,
                    channel_start=channel_start,
                    channel_end=channel_end,
                    output_path=sample_dir / filename,
                )
                projection_summary["channel_slices"].append(filename)
            sample_summary["projections"][projection] = projection_summary
        output_summary["samples"].append(sample_summary)
        print(f"[w8a8-output-pattern] rendered sample={sample_id}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[w8a8-output-pattern] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
