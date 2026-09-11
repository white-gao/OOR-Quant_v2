from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import TwoSlopeNorm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import run_m1_onerec_ad as _runner_bootstrap  # noqa: F401
from benchmark.tasks.v1_0.registry import get_task_config

from .quant import QuantFormat, activation_per_token_qdq_by_format, validate_quant_format
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    dtype_from_name,
    format_prompt,
    get_transformer_layers,
    load_task_data,
    resolve_input_device,
    resolve_repo_path,
    set_seed,
)


SID_GROUP_RE = re.compile(
    r"<\|sid_begin\|>"
    r"<s_a_[^>]+>"
    r"<s_b_[^>]+>"
    r"<s_c_[^>]+>"
    r"<\|sid_end\|>"
)
SID_SLOT_RES = tuple(re.compile(rf"<s_{slot}_[^>]+>") for slot in ("a", "b", "c"))


@dataclass(frozen=True)
class CompressedPrompt:
    text: str
    original_token_count: int
    compressed_token_count: int
    original_sid_groups: int
    retained_sid_groups: int
    removed_sid_groups: int
    sid_runs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize layer-27 q_proj/o_proj FP and locally fake-quantized "
            "activation matrices for five reproducibly sampled AD prompts."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_pool_size", type=int, default=3000)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_tokens", type=int, default=200)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--channel_slice_size", type=int, default=512)
    parser.add_argument(
        "--activation_quant_format",
        default="fp8_e4m3fn",
        choices=("fp8_e4m3fn", "fp4_e2m1", "int8", "int6", "int4"),
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default=(
            "artifacts/results/fake_quant/probes/"
            "activation_quant_patterns_layer27_ad_fp8"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sid_runs(prompt: str, matches: Sequence[re.Match[str]]) -> list[list[int]]:
    if not matches:
        return []
    runs: list[list[int]] = [[0]]
    for match_idx in range(1, len(matches)):
        gap = prompt[matches[match_idx - 1].end() : matches[match_idx].start()]
        if gap.strip():
            runs.append([match_idx])
        else:
            runs[-1].append(match_idx)
    return runs


def _remove_matches(
    prompt: str,
    matches: Sequence[re.Match[str]],
    removed_indices: set[int],
) -> str:
    pieces: list[str] = []
    cursor = 0
    for match_idx, match in enumerate(matches):
        pieces.append(prompt[cursor : match.start()])
        if match_idx not in removed_indices:
            pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(prompt[cursor:])
    return "".join(pieces)


def _validate_sid_groups(prompt: str, *, expected_groups: int) -> None:
    group_count = len(SID_GROUP_RE.findall(prompt))
    slot_counts = tuple(len(pattern.findall(prompt)) for pattern in SID_SLOT_RES)
    if group_count != expected_groups or slot_counts != (expected_groups,) * 3:
        raise ValueError(
            "SID compression produced an invalid prompt: "
            f"groups={group_count}, slot_counts={slot_counts}, expected={expected_groups}."
        )


def compress_sid_prompt(
    prompt: str,
    *,
    tokenizer: Any,
    target_tokens: int,
) -> CompressedPrompt:
    """Remove complete, oldest SID-ABC groups until the prompt is near the token budget.

    Contiguous SID sequences are treated as separate history runs. Removal is
    balanced across runs by their retained fraction, while at least one recent
    SID group is preserved per run whenever the fixed prompt permits it.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    matches = list(SID_GROUP_RE.finditer(prompt))
    if not matches:
        raise ValueError("The selected prompt contains no complete SID-ABC group.")
    runs = _sid_runs(prompt, matches)
    original_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    removed: set[int] = set()
    retained_by_run = [list(run) for run in runs]
    candidate = prompt
    candidate_tokens = original_tokens

    while candidate_tokens > target_tokens:
        removable_runs = [
            run_idx for run_idx, retained in enumerate(retained_by_run) if len(retained) > 1
        ]
        if not removable_runs:
            break
        # Remove from the run retaining the largest fraction of its original
        # history. Ties favor the longer run, then the earlier run.
        run_idx = max(
            removable_runs,
            key=lambda idx: (
                len(retained_by_run[idx]) / len(runs[idx]),
                len(retained_by_run[idx]),
                -idx,
            ),
        )
        removed_idx = retained_by_run[run_idx].pop(0)
        removed.add(removed_idx)
        candidate = _remove_matches(prompt, matches, removed)
        candidate_tokens = len(tokenizer(candidate, add_special_tokens=True)["input_ids"])

    retained_groups = len(matches) - len(removed)
    _validate_sid_groups(candidate, expected_groups=retained_groups)
    return CompressedPrompt(
        text=candidate,
        original_token_count=original_tokens,
        compressed_token_count=candidate_tokens,
        original_sid_groups=len(matches),
        retained_sid_groups=retained_groups,
        removed_sid_groups=len(removed),
        sid_runs=len(runs),
    )


def _centered_norm(values: np.ndarray) -> tuple[float, float, TwoSlopeNorm | None]:
    value_min = float(values.min())
    value_max = float(values.max())
    if value_min < 0.0 < value_max:
        return value_min, value_max, TwoSlopeNorm(vmin=value_min, vcenter=0.0, vmax=value_max)
    if value_min == value_max:
        delta = max(abs(value_min) * 1e-6, 1e-12)
        return value_min - delta, value_max + delta, None
    return value_min, value_max, None


def _plot_slice(
    *,
    raw: np.ndarray,
    quantized: np.ndarray,
    error: np.ndarray,
    sample_id: str,
    projection: str,
    layer_idx: int,
    channel_start: int,
    channel_end: int,
    quant_format: str,
    output_path: Path,
) -> None:
    raw_slice = raw[:, channel_start:channel_end]
    quant_slice = quantized[:, channel_start:channel_end]
    error_slice = error[:, channel_start:channel_end]
    shared = np.concatenate((raw_slice.reshape(-1), quant_slice.reshape(-1)))
    shared_min, shared_max, shared_norm = _centered_norm(shared)
    error_min, error_max, error_norm = _centered_norm(error_slice)

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), constrained_layout=True)
    panels = (
        (raw_slice, "BF16 input", shared_min, shared_max, shared_norm),
        (quant_slice, f"{quant_format} QDQ input", shared_min, shared_max, shared_norm),
        (error_slice, "QDQ - BF16 error", error_min, error_max, error_norm),
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
        axis.set_xlabel("hidden channel")
        fig.colorbar(image, ax=axis, fraction=0.018, pad=0.01)

    fig.suptitle(
        f"sample={sample_id} layer={layer_idx} {projection} "
        f"channels=[{channel_start}, {channel_end})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _tensor_summary(raw: np.ndarray, quantized: np.ndarray) -> dict[str, Any]:
    error = quantized - raw
    raw_energy = float(np.mean(np.square(raw, dtype=np.float64)))
    mse = float(np.mean(np.square(error, dtype=np.float64)))
    return {
        "shape": list(raw.shape),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "quantized_min": float(quantized.min()),
        "quantized_max": float(quantized.max()),
        "error_min": float(error.min()),
        "error_max": float(error.max()),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "mse": mse,
        "relative_mse": mse / max(raw_energy, 1e-30),
    }


def _capture_projection_inputs(
    *,
    model: nn.Module,
    layer_idx: int,
    encoded: dict[str, torch.Tensor],
    quant_format: QuantFormat,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    layers = get_transformer_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer={layer_idx} is out of range for {len(layers)} blocks.")
    attention = layers[layer_idx].self_attn
    captured: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def make_hook(name: str):
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if name in captured:
                raise RuntimeError(f"{name} was invoked more than once for one prompt.")
            raw = args[0].detach()
            quantized = activation_per_token_qdq_by_format(
                raw,
                quant_format=quant_format,
            )
            captured[name] = (
                raw.squeeze(0).float().cpu().numpy(),
                quantized.squeeze(0).float().cpu().numpy(),
            )

        return hook

    handles = (
        attention.q_proj.register_forward_pre_hook(make_hook("q_proj")),
        attention.o_proj.register_forward_pre_hook(make_hook("o_proj")),
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


def main() -> None:
    args = parse_args()
    if args.sample_pool_size < args.num_samples or args.num_samples <= 0:
        raise ValueError("sample_pool_size must be at least num_samples > 0.")
    if args.channel_slice_size <= 0:
        raise ValueError("channel_slice_size must be positive.")
    quant_format = validate_quant_format(args.activation_quant_format)
    output_dir = resolve_repo_path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model_path = resolve_repo_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "")
    sample_pool = load_task_data(
        task_name="ad",
        tokenizer=tokenizer,
        data_dir=str(resolve_repo_path(args.data_dir)),
        split=args.split,
        sample_size=args.sample_pool_size,
    )
    pool_items = list(sample_pool.items())
    selected_positions = sorted(random.Random(args.seed).sample(range(len(pool_items)), args.num_samples))
    selected = [pool_items[position] for position in selected_positions]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "task": "ad",
        "split": args.split,
        "sample_pool_size": args.sample_pool_size,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "selected_pool_positions": selected_positions,
        "target_tokens": args.target_tokens,
        "layer": args.layer,
        "projections": ["q_proj", "o_proj"],
        "activation_quant_format": quant_format,
        "activation_quant_granularity": "dynamic per-token over the hidden dimension",
        "error_scope": "local QDQ error on BF16 model activations; weights remain BF16",
        "channel_slice_size": args.channel_slice_size,
        "plot_scaling": (
            "exact true extrema with no percentile clipping; BF16/QDQ panels share their "
            "joint per-slice range; error uses its exact per-slice range"
        ),
        "samples": [],
    }

    for sample_order, (sample_id, sample) in enumerate(selected):
        prompt = format_prompt(str(sample["prompt"]), prompt_token)
        compressed = compress_sid_prompt(
            prompt,
            tokenizer=tokenizer,
            target_tokens=args.target_tokens,
        )
        encoded = tokenizer(compressed.text, return_tensors="pt")
        encoded = {
            key: value.to(input_device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        captures = _capture_projection_inputs(
            model=model,
            layer_idx=args.layer,
            encoded=encoded,
            quant_format=quant_format,
        )

        sample_dir = output_dir / f"sample_{sample_order:02d}_id_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_summary: dict[str, Any] = {
            "sample_order": sample_order,
            "sample_id": str(sample_id),
            "pool_position": selected_positions[sample_order],
            "original_token_count": compressed.original_token_count,
            "compressed_token_count": compressed.compressed_token_count,
            "original_sid_groups": compressed.original_sid_groups,
            "retained_sid_groups": compressed.retained_sid_groups,
            "removed_sid_groups": compressed.removed_sid_groups,
            "sid_runs": compressed.sid_runs,
            "projections": {},
        }
        (sample_dir / "compressed_prompt.txt").write_text(compressed.text, encoding="utf-8")

        for projection, (raw, quantized) in captures.items():
            error = quantized - raw
            np.savez_compressed(
                sample_dir / f"{projection}_activation_matrices.npz",
                raw=raw,
                quantized=quantized,
                error=error,
            )
            projection_summary = _tensor_summary(raw, quantized)
            projection_summary["channel_slices"] = []
            for channel_start in range(0, raw.shape[-1], args.channel_slice_size):
                channel_end = min(channel_start + args.channel_slice_size, raw.shape[-1])
                filename = (
                    f"layer_{args.layer:02d}_{projection}_channels_"
                    f"{channel_start:04d}_{channel_end:04d}.png"
                )
                _plot_slice(
                    raw=raw,
                    quantized=quantized,
                    error=error,
                    sample_id=str(sample_id),
                    projection=projection,
                    layer_idx=args.layer,
                    channel_start=channel_start,
                    channel_end=channel_end,
                    quant_format=quant_format,
                    output_path=sample_dir / filename,
                )
                projection_summary["channel_slices"].append(filename)
            sample_summary["projections"][projection] = projection_summary
        summary["samples"].append(sample_summary)
        print(
            f"[activation-pattern] sample={sample_id} "
            f"tokens={compressed.original_token_count}->{compressed.compressed_token_count} "
            f"sid_groups={compressed.original_sid_groups}->{compressed.retained_sid_groups}"
        )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[activation-pattern] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
