#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .apply import (
    BaselineQuantSummary,
    SmoothScope,
    apply_baseline_qdq,
    install_shared_input_activation_quantization,
)
from .gptq import (
    DEFAULT_GPTQ_BLOCK_SIZE,
    DEFAULT_GPTQ_DAMP_PERCENT,
    collect_gptq_hessians,
    gptq_quantized_module_from_hessians,
)
from .modules import BaselineFakeQuantLinear
from .omniquant import OmniQuantConfig, apply_omniquant_layers
from .quant import ActQuant, ActQuantMode, QUANT_FORMAT_CHOICES, QuantFormat
from .support.runtime_utils import _detach_tree, _module_device, _move_tree_to_device
from .support.smoothquant_runtime import (
    Batch,
    DEFAULT_SMOOTHQUANT_ALPHA,
    DEFAULT_SMOOTHQUANT_MAX_SCALE,
    DEFAULT_SMOOTHQUANT_MIN_SCALE,
    DEFAULT_SMOOTH_FOLD,
    DEFAULT_SMOOTH_SCOPE,
    _batch_to_args_kwargs,
    collect_smoothquant_scales,
    fold_smoothquant_scales_inplace,
    smoothquant_quantized_module_from_scales,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark  # noqa: E402
from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402
from shared.paths import data_root, fake_results_root, model_root  # noqa: E402


DEFAULT_MODEL_PATH = str(model_root() / "1.7B")
DEFAULT_DATA_DIR = str(data_root() / "onerec_data" / "benchmark_data")
DEFAULT_OUTPUT_DIR = str(fake_results_root() / "recommender" / "ptq_ad")
DEFAULT_TASK = "ad"
TASK_CHOICES = ("ad", "product", "video")
DEFAULT_SPLIT = "test"
DEFAULT_CALIB_OFFSET = 0
DEFAULT_EVAL_OFFSET = 0
DEFAULT_ACT_QUANT: ActQuant = "per_token"
DEFAULT_ACT_QUANT_MODE: ActQuantMode = "shared_input"
DEFAULT_WEIGHT_QUANT_FORMAT: QuantFormat = "fp8_e4m3fn"
DEFAULT_ACTIVATION_QUANT_FORMAT: QuantFormat = "fp8_e4m3fn"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_NUM_BEAMS = 32
DEFAULT_NUM_RETURN_SEQUENCES = 32
DEFAULT_MAX_NEW_TOKENS = 3
DEFAULT_SEED = 42
DEFAULT_SID_PPL_MAX_ITEMS = 1
SID_ITEM_RE = re.compile(
    r"<\|sid_begin\|>"
    r"(?P<sid><s_a_[^>]+><s_b_[^>]+><s_c_[^>]+>)"
    r"<\|sid_end\|>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneRec recommendation evaluation with full precision or composable fake-QDQ quantization."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, choices=TASK_CHOICES)
    parser.add_argument(
        "--mode",
        default="baseline_w8a8",
        choices=[
            "full_precision",
            "baseline_w8a8",
            "baseline_qdq",
            "smoothquant_w8a8",
            "gptq_fp8_w8a8",
            "omniquant",
        ],
    )
    parser.add_argument(
        "--weight_quant_format",
        choices=QUANT_FORMAT_CHOICES,
        default=DEFAULT_WEIGHT_QUANT_FORMAT,
        help="Fake-QDQ weight format. INT formats use per-output-channel quantization.",
    )
    parser.add_argument(
        "--omni_weight_quant_scheme",
        choices=("symmetric", "asymmetric"),
        default="symmetric",
        help=(
            "OmniQuant weight quantizer: legacy signed symmetric, or paper-style "
            "asymmetric LWC with separate upper/lower clipping and a zero point."
        ),
    )
    parser.add_argument(
        "--omni_lwc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-output-channel learnable weight clipping.",
    )
    parser.add_argument("--no-omni-lwc", dest="omni_lwc", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--omni_let",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Qwen3 scale-only LET (QKV, gate/up, and GQA-aware V/O).",
    )
    parser.add_argument("--no-omni-let", dest="omni_let", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--omni_let_mode",
        choices=("none", "fixed", "learned"),
        default=None,
        help=(
            "LET ablation: none disables smoothing, fixed freezes the shared "
            "SmoothQuant initialization, learned optimizes it blockwise."
        ),
    )
    parser.add_argument("--omni_epochs", type=int, default=10)
    parser.add_argument("--omni_lwc_lr", type=float, default=1e-2)
    parser.add_argument("--omni_let_lr", type=float, default=1e-3)
    parser.add_argument("--omni_init_lwc_logit", type=float, default=4.0)
    parser.add_argument("--omni_min_let_scale", type=float, default=5e-2)
    parser.add_argument("--omni_max_let_scale", type=float, default=20.0)
    parser.add_argument("--omni_max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--activation_quant_format",
        choices=QUANT_FORMAT_CHOICES,
        default=DEFAULT_ACTIVATION_QUANT_FORMAT,
        help="Fake-QDQ activation format. Non-none formats use dynamic per-token symmetric quantization.",
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "last:K", "0,2-4".')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calib_sample_size", default="1024")
    parser.add_argument("--eval_sample_size", default="full")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--compute_sid_ppl",
        action="store_true",
        help="Compute auxiliary teacher-forcing NLL/PPL on ground-truth SID tokens.",
    )
    parser.add_argument(
        "--sid_ppl_max_items",
        type=int,
        default=DEFAULT_SID_PPL_MAX_ITEMS,
        help="Maximum ground-truth SID items per sample for teacher-forcing NLL/PPL.",
    )
    args = parser.parse_args()
    _attach_fixed_defaults(args)
    return args


def _attach_fixed_defaults(args: argparse.Namespace) -> None:
    if args.omni_let_mode is None:
        args.omni_let_mode = "learned" if args.omni_let else "none"
    else:
        args.omni_let = args.omni_let_mode != "none"
    args.split = DEFAULT_SPLIT
    args.calib_offset = DEFAULT_CALIB_OFFSET
    args.eval_offset = DEFAULT_EVAL_OFFSET
    args.act_quant = "none" if args.activation_quant_format == "none" else DEFAULT_ACT_QUANT
    args.act_quant_mode = "per_linear" if args.act_quant == "none" else DEFAULT_ACT_QUANT_MODE
    args.dtype = DEFAULT_DTYPE
    args.num_beams = DEFAULT_NUM_BEAMS
    args.num_return_sequences = DEFAULT_NUM_RETURN_SEQUENCES
    args.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    args.seed = DEFAULT_SEED
    args.smoothquant_alpha = DEFAULT_SMOOTHQUANT_ALPHA
    args.smooth_scope = DEFAULT_SMOOTH_SCOPE
    args.smooth_fold = DEFAULT_SMOOTH_FOLD
    args.smoothquant_min_scale = DEFAULT_SMOOTHQUANT_MIN_SCALE
    args.smoothquant_max_scale = DEFAULT_SMOOTHQUANT_MAX_SCALE
    args.gptq_damp_percent = DEFAULT_GPTQ_DAMP_PERCENT
    args.gptq_block_size = DEFAULT_GPTQ_BLOCK_SIZE


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_sample_size(value: Any) -> Any:
    if value is None or value == "":
        return None
    if value == "full":
        return "full"
    return int(value)


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def default_calib_split(
    data_dir: str | os.PathLike[str],
    fallback_split: str,
    *,
    task_name: str = DEFAULT_TASK,
) -> str:
    calib_file = resolve_repo_path(data_dir) / task_name / f"{task_name}_calib.parquet"
    return "calib" if calib_file.exists() else fallback_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_device(model: nn.Module, fallback: str) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("Could not find transformer layers at model.model.layers or model.layers.")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Expected layers to be nn.ModuleList, got {type(layers)!r}.")
    return layers


def parse_layer_indices(spec: str, *, num_layers: int) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("last:"):
        count = int(spec.split(":", 1)[1])
        if count <= 0 or count > num_layers:
            raise ValueError(f"Invalid last layer count: {count}")
        return list(range(num_layers - count, num_layers))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    deduped = sorted(set(indices))
    if not deduped:
        raise ValueError("No layer indices selected.")
    for idx in deduped:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers.")
    return deduped


def load_task_data(
    *,
    task_name: str,
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: Any,
    sample_offset: int = 0,
    require_answer: bool = True,
    drop_final_assistant: bool = False,
) -> dict[str, dict[str, Any]]:
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be non-negative, got {sample_offset}")

    loader_sample_size = sample_size
    if isinstance(sample_size, int):
        loader_sample_size = sample_size + sample_offset

    loader = get_loader(
        task_name=task_name,
        data_dir=data_dir,
        enable_thinking=False,
        tokenizer=tokenizer,
    )
    load_kwargs: dict[str, Any] = {"split": split, "sample_size": loader_sample_size}
    # Keep the default path compatible with lightweight external/custom
    # loaders that predate calibration-only options.
    if not require_answer:
        load_kwargs["require_answer"] = False
    if drop_final_assistant:
        load_kwargs["drop_final_assistant"] = True
    data = loader.load_data(**load_kwargs)
    if sample_offset == 0 and not isinstance(sample_size, int):
        return data

    items = list(data.items())
    if sample_offset:
        if sample_offset >= len(items):
            raise ValueError(
                f"sample_offset={sample_offset} leaves no samples after loading {len(items)} rows."
            )
        items = items[sample_offset:]
    if isinstance(sample_size, int):
        items = items[:sample_size]
    return dict(items)


def load_ad_data(
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: Any,
    sample_offset: int = 0,
) -> dict[str, dict[str, Any]]:
    return load_task_data(
        task_name="ad",
        tokenizer=tokenizer,
        data_dir=data_dir,
        split=split,
        sample_size=sample_size,
        sample_offset=sample_offset,
    )


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def build_model_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        batches.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
        )
    return batches


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Could not find a tensor in output type {type(value)!r}.")


def capture_layer_input_batches(
    *,
    model: nn.Module,
    layer: nn.Module,
    model_batches: Iterable[Mapping[str, Any]],
) -> list[Batch]:
    captured: list[Batch] = []

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((_detach_tree(args), _detach_tree(kwargs)))

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in model_batches:
                try:
                    model(**batch, use_cache=False)
                except TypeError:
                    model(**batch)
    finally:
        handle.remove()
        model.train(was_training)
    return captured


def advance_layer_input_batches(
    *,
    layer: nn.Module,
    batches: Sequence[Batch],
) -> list[Batch]:
    """Run one transformer block and build input batches for the next block."""
    advanced: list[Batch] = []
    was_training = layer.training
    target_device = _module_device(layer)
    layer.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, target_device)
                kwargs = _move_tree_to_device(kwargs, target_device)
                output = layer(*args, **kwargs)
                hidden = _first_tensor(output).detach()
                next_args: tuple[Any, ...] = (hidden, *args[1:]) if args else ()
                next_kwargs = dict(kwargs)
                if not next_args:
                    if "hidden_states" in next_kwargs:
                        next_kwargs["hidden_states"] = hidden
                    else:
                        next_args = (hidden,)
                advanced.append((_detach_tree(next_args), _detach_tree(next_kwargs)))
    finally:
        layer.train(was_training)
    return advanced


def apply_smoothquant_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    act_quant: ActQuant,
    act_quant_mode: ActQuantMode = "per_linear",
    smoothquant_alpha: float = DEFAULT_SMOOTHQUANT_ALPHA,
    smoothquant_min_scale: float | None = DEFAULT_SMOOTHQUANT_MIN_SCALE,
    smoothquant_max_scale: float | None = DEFAULT_SMOOTHQUANT_MAX_SCALE,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    smooth_fold: bool = True,
) -> dict[int, BaselineQuantSummary]:
    """Apply SmoothQuant-equivalent W8A8 fake quantization to selected layers."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None

    for layer_idx in selected_layer_indices:
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None:
                raise RuntimeError("Internal error: stream_layer_idx is not initialized.")
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=fp_inputs,
                )
                stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        scales = collect_smoothquant_scales(
            teacher_block,
            fp_inputs,
            alpha=smoothquant_alpha,
            min_scale=smoothquant_min_scale,
            max_scale=smoothquant_max_scale,
            smooth_scope=smooth_scope,
        )
        next_fp_inputs = advance_layer_input_batches(layer=teacher_block, batches=fp_inputs)
        quant_block = copy.deepcopy(teacher_block)
        folded_names = (
            fold_smoothquant_scales_inplace(quant_block, scales, smooth_scope=smooth_scope)
            if smooth_fold
            else set()
        )
        quant_block, replaced = smoothquant_quantized_module_from_scales(
            quant_block,
            scales,
            act_quant=act_quant,
            smooth_scope=smooth_scope,
            folded_names=folded_names,
        )
        shared_attention_modules = 0
        shared_mlp_modules = 0
        if act_quant == "per_token" and act_quant_mode == "shared_input":
            shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(quant_block)
        layers[layer_idx] = quant_block
        summaries[layer_idx] = BaselineQuantSummary(
            replaced_linears=replaced,
            skipped_linears=0,
            shared_attention_modules=shared_attention_modules,
            shared_mlp_modules=shared_mlp_modules,
        )
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(
            f"[smoothquant_w8a8] layer={layer_idx} replaced_linears={replaced}, "
            f"smooth_scope={smooth_scope}, "
            f"smooth_fold={int(smooth_fold)}, folded={len(folded_names)}, "
            f"shared_attention_modules={shared_attention_modules}, "
            f"shared_mlp_modules={shared_mlp_modules}"
        )
    return summaries


def apply_gptq_fp8_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    act_quant: ActQuant,
    act_quant_mode: ActQuantMode = "per_linear",
    damp_percent: float = DEFAULT_GPTQ_DAMP_PERCENT,
    block_size: int = DEFAULT_GPTQ_BLOCK_SIZE,
) -> dict[int, BaselineQuantSummary]:
    """Apply GPTQ-calibrated FP8 weight + W8A8 fake quantization to selected layers."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None

    for layer_idx in selected_layer_indices:
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None:
                raise RuntimeError("Internal error: stream_layer_idx is not initialized.")
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=fp_inputs,
                )
                stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        hessians = collect_gptq_hessians(
            teacher_block,
            fp_inputs,
        )
        next_fp_inputs = advance_layer_input_batches(layer=teacher_block, batches=fp_inputs)
        quant_block = copy.deepcopy(teacher_block)
        quant_block, replaced = gptq_quantized_module_from_hessians(
            quant_block,
            hessians,
            act_quant=act_quant,
            damp_percent=damp_percent,
            block_size=block_size,
        )
        shared_attention_modules = 0
        shared_mlp_modules = 0
        if act_quant == "per_token" and act_quant_mode == "shared_input":
            shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(quant_block)
        layers[layer_idx] = quant_block
        summaries[layer_idx] = BaselineQuantSummary(
            replaced_linears=replaced,
            skipped_linears=0,
            shared_attention_modules=shared_attention_modules,
            shared_mlp_modules=shared_mlp_modules,
        )
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(
            f"[gptq_fp8_w8a8] layer={layer_idx} replaced_linears={replaced}, "
            f"damp_percent={damp_percent}, block_size={block_size}, "
            f"shared_attention_modules={shared_attention_modules}, "
            f"shared_mlp_modules={shared_mlp_modules}"
        )
    return summaries


def apply_baseline_layers(
    *,
    model: nn.Module,
    layer_indices: Sequence[int],
    act_quant: ActQuant,
    act_quant_mode: ActQuantMode = "per_linear",
    weight_quant_format: QuantFormat = DEFAULT_WEIGHT_QUANT_FORMAT,
    activation_quant_format: QuantFormat = DEFAULT_ACTIVATION_QUANT_FORMAT,
) -> dict[int, BaselineQuantSummary]:
    """Apply min-max fake QDQ with independently selected weight/activation formats."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        if isinstance(layer, nn.Linear):
            layers[layer_idx] = BaselineFakeQuantLinear(
                layer,
                act_quant=act_quant,
                weight_quant_format=weight_quant_format,
                activation_quant_format=activation_quant_format,
            )
            summary = BaselineQuantSummary(replaced_linears=1, skipped_linears=0)
        else:
            summary = apply_baseline_qdq(
                layer,
                weight_quant_format=weight_quant_format,
                activation_quant_format=activation_quant_format,
                act_quant_mode=act_quant_mode,
            )
        summaries[layer_idx] = summary
        print(
            f"[baseline_qdq w={weight_quant_format} a={activation_quant_format}] "
            f"layer={layer_idx} replaced_linears={summary.replaced_linears} "
            f"skipped_linears={summary.skipped_linears}, "
            f"shared_attention_modules={summary.shared_attention_modules}, "
            f"shared_mlp_modules={summary.shared_mlp_modules}"
        )
    return summaries


def decode_generations(tokenizer: Any, sequences: torch.Tensor, prompt_len: int) -> list[str]:
    generations = []
    for seq in sequences:
        generated_ids = seq[prompt_len:]
        generations.append(tokenizer.decode(generated_ids, skip_special_tokens=False))
    return generations


def generate_one(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    input_device: torch.device,
    args: argparse.Namespace,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {
        key: value.to(input_device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return decode_generations(tokenizer, output.detach().cpu(), prompt_len)


def extract_sid_teacher_forcing_targets(ground_truth: str, *, max_items: int) -> list[str]:
    """Return SID target triples without the surrounding sid_begin/sid_end tokens."""
    if max_items <= 0:
        return []
    targets: list[str] = []
    for match in SID_ITEM_RE.finditer(ground_truth or ""):
        targets.append(match.group("sid"))
        if len(targets) >= max_items:
            break
    return targets


def _safe_exp(value: float) -> float:
    return float(math.exp(min(value, 50.0)))


def compute_sid_teacher_forcing_metrics(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    ground_truth: str,
    input_device: torch.device,
    max_items: int = DEFAULT_SID_PPL_MAX_ITEMS,
) -> dict[str, Any]:
    """Compute teacher-forcing NLL/PPL for ground-truth SID tokens."""
    targets = extract_sid_teacher_forcing_targets(ground_truth, max_items=max_items)
    if not targets:
        return {
            "sid_tf_valid": False,
            "sid_tf_num_items": 0,
            "sid_tf_num_tokens": 0,
        }

    prompt_encoded = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = prompt_encoded["input_ids"]
    prompt_len = int(prompt_input_ids.shape[-1])
    prompt_attention_mask = prompt_encoded.get("attention_mask")

    total_loss = 0.0
    total_tokens = 0
    valid_items = 0
    first_target = targets[0]

    for target_text in targets:
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        target_len = int(target_ids.shape[-1])
        if target_len == 0:
            continue

        input_ids = torch.cat([prompt_input_ids, target_ids], dim=-1).to(input_device)
        model_inputs: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if prompt_attention_mask is not None:
            target_mask = torch.ones_like(target_ids)
            attention_mask = torch.cat([prompt_attention_mask, target_mask], dim=-1).to(input_device)
            model_inputs["attention_mask"] = attention_mask

        labels = target_ids.reshape(-1).to(input_device)
        positions = torch.arange(
            prompt_len - 1,
            prompt_len - 1 + target_len,
            device=input_device,
        )
        with torch.inference_mode():
            try:
                outputs = model(**model_inputs, use_cache=False)
            except TypeError:
                outputs = model(**model_inputs)
        logits = outputs.logits[0, positions, :].float()
        losses = F.cross_entropy(logits, labels, reduction="none")
        total_loss += float(losses.sum().item())
        total_tokens += target_len
        valid_items += 1

    if total_tokens == 0:
        return {
            "sid_tf_valid": False,
            "sid_tf_num_items": 0,
            "sid_tf_num_tokens": 0,
        }

    mean_nll = total_loss / total_tokens
    return {
        "sid_tf_valid": True,
        "sid_tf_nll": mean_nll,
        "sid_tf_ppl": _safe_exp(mean_nll),
        "sid_tf_num_items": valid_items,
        "sid_tf_num_tokens": total_tokens,
        "sid_tf_target": first_target,
    }


def aggregate_sid_teacher_forcing_metrics(
    sample_metrics: Mapping[str, Mapping[str, Any]],
    *,
    max_items: int,
) -> dict[str, Any]:
    total_samples = len(sample_metrics)
    valid_samples = 0
    total_tokens = 0
    weighted_nll = 0.0
    total_items = 0
    for metrics in sample_metrics.values():
        if not metrics.get("sid_tf_valid"):
            continue
        num_tokens = int(metrics.get("sid_tf_num_tokens", 0))
        if num_tokens <= 0:
            continue
        valid_samples += 1
        total_items += int(metrics.get("sid_tf_num_items", 0))
        total_tokens += num_tokens
        weighted_nll += float(metrics["sid_tf_nll"]) * num_tokens

    mean_nll = weighted_nll / total_tokens if total_tokens else None
    return {
        "sid_tf_enabled": True,
        "sid_tf_definition": "teacher_forcing_gt_sid_tokens_excluding_sid_begin_end",
        "sid_tf_max_items_per_sample": max_items,
        "sid_tf_total_samples": total_samples,
        "sid_tf_valid_samples": valid_samples,
        "sid_tf_invalid_samples": total_samples - valid_samples,
        "sid_tf_num_items": total_items,
        "sid_tf_num_tokens": total_tokens,
        "sid_tf_nll": mean_nll,
        "sid_tf_ppl": None if mean_nll is None else _safe_exp(mean_nll),
    }


def result_path(output_dir: str, model_name: str, task_name: str, split: str) -> Path:
    return resolve_repo_path(output_dir) / model_name / task_name / f"{split}_generated.json"


def save_results(
    *,
    output_file: Path,
    model_name: str,
    split: str,
    test_data: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, list[str]],
    total_time: float,
    config: Mapping[str, Any],
    task_name: str = DEFAULT_TASK,
    sample_aux_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    samples: dict[str, dict[str, Any]] = {}
    for sample_id, sample in test_data.items():
        item = {
            "prompt": sample.get("prompt", ""),
            "generations": generations.get(sample_id, []),
            "ground_truth": sample.get("ground_truth", ""),
        }
        if "metadata" in sample:
            item["metadata"] = sample["metadata"]
        if sample_aux_metrics and sample_id in sample_aux_metrics:
            item.update(sample_aux_metrics[sample_id])
        samples[sample_id] = item

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "task_name": task_name,
        "split": split,
        "total_time": total_time,
        "avg_time_per_sample": total_time / len(samples) if samples else 0.0,
        "quant_config": dict(config),
        "samples": samples,
    }
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_evaluate(output_dir: str, data_dir: str, overwrite: bool, *, task_name: str = DEFAULT_TASK) -> None:
    output_root = resolve_repo_path(output_dir)
    data_root = resolve_repo_path(data_dir)
    Benchmark.evaluate_dev(
        generation_results_dir=str(output_root),
        output_path=str(output_root / "eval_results.json"),
        data_dir=str(data_root),
        overwrite=overwrite,
        task_types=[task_name],
    )


def merge_sid_teacher_forcing_metrics_into_eval(
    *,
    output_dir: str,
    model_name: str,
    split: str,
    metrics: Mapping[str, Any],
    task_name: str = DEFAULT_TASK,
) -> None:
    eval_path = resolve_repo_path(output_dir) / "eval_results.json"
    if not eval_path.exists():
        return
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    model_metrics = data.setdefault(model_name, {})
    task_metrics = model_metrics.setdefault(task_name, {})
    split_metrics = task_metrics.setdefault(split, {})
    split_metrics.update(dict(metrics))
    eval_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def summaries_to_jsonable(summaries: Mapping[int, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for layer_idx, summary in summaries.items():
        item = {
            "replaced_linears": summary.replaced_linears,
            "skipped_linears": getattr(summary, "skipped_linears", 0),
            "shared_attention_modules": summary.shared_attention_modules,
            "shared_mlp_modules": summary.shared_mlp_modules,
        }
        if hasattr(summary, "initial_loss"):
            item["initial_loss"] = summary.initial_loss
            item["final_loss"] = summary.final_loss
            item["let_scales"] = list(summary.let_scales)
        serialized[str(layer_idx)] = item
    return serialized


def main() -> None:
    args = parse_args()
    if args.sid_ppl_max_items <= 0:
        raise ValueError(f"--sid_ppl_max_items must be positive, got {args.sid_ppl_max_items}")
    set_seed(args.seed)

    model_name = Path(args.model_path.rstrip("/")).name
    output_file = result_path(args.output_dir, model_name, args.task, args.split)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model = model.to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    task_config = get_task_config(args.task)
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data_dir = args.data_dir
    eval_data_dir = args.data_dir
    calib_split = default_calib_split(calib_data_dir, args.split, task_name=args.task)

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    baseline_summaries: dict[int, Any] = {}

    if args.mode == "full_precision":
        pass
    elif args.mode in {"baseline_w8a8", "baseline_qdq"}:
        baseline_summaries = apply_baseline_layers(
            model=model,
            layer_indices=layer_indices,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
            weight_quant_format=args.weight_quant_format,
            activation_quant_format=args.activation_quant_format,
        )
    elif args.mode in {"smoothquant_w8a8", "gptq_fp8_w8a8", "omniquant"}:
        if args.mode == "omniquant" and args.weight_quant_format not in {"int4", "int8"}:
            raise ValueError("omniquant currently requires --weight_quant_format int4 or int8.")
        if args.mode != "omniquant" and (
            args.weight_quant_format != "fp8_e4m3fn"
            or args.activation_quant_format != "fp8_e4m3fn"
        ):
            raise ValueError(
                f"{args.mode} currently implements only FP8 E4M3 weight/activation QDQ. "
                "Use --mode baseline_qdq for mixed INT4/INT8/FP8 experiments."
            )
        calib_data = load_task_data(
            task_name=args.task,
            tokenizer=tokenizer,
            data_dir=str(resolve_repo_path(calib_data_dir)),
            split=calib_split,
            sample_size=parse_sample_size(args.calib_sample_size),
            sample_offset=args.calib_offset,
            require_answer=False,
            drop_final_assistant=True,
        )
        if not calib_data:
            raise ValueError(
                "No calibration samples were loaded. Check --data_dir, --task, and the calibration parquet."
            )
        calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
        calib_batches = build_model_batches(
            tokenizer=tokenizer,
            prompts=calib_prompts,
            device=input_device,
        )
        if args.mode == "smoothquant_w8a8":
            baseline_summaries = apply_smoothquant_layers(
                model=model,
                model_batches=calib_batches,
                layer_indices=layer_indices,
                act_quant=args.act_quant,
                act_quant_mode=args.act_quant_mode,
                smoothquant_alpha=args.smoothquant_alpha,
                smoothquant_min_scale=args.smoothquant_min_scale,
                smoothquant_max_scale=args.smoothquant_max_scale,
                smooth_scope=args.smooth_scope,
                smooth_fold=args.smooth_fold,
            )
        elif args.mode == "gptq_fp8_w8a8":
            baseline_summaries = apply_gptq_fp8_layers(
                model=model,
                model_batches=calib_batches,
                layer_indices=layer_indices,
                act_quant=args.act_quant,
                act_quant_mode=args.act_quant_mode,
                damp_percent=args.gptq_damp_percent,
                block_size=args.gptq_block_size,
            )
        else:
            baseline_summaries = apply_omniquant_layers(
                model=model,
                model_batches=calib_batches,
                layer_indices=layer_indices,
                config=OmniQuantConfig(
                    weight_quant_format=args.weight_quant_format,
                    activation_quant_format=args.activation_quant_format,
                    weight_quant_scheme=args.omni_weight_quant_scheme,
                    use_lwc=args.omni_lwc,
                    use_let=args.omni_let,
                    learn_let=args.omni_let_mode == "learned",
                    epochs=args.omni_epochs,
                    lwc_lr=args.omni_lwc_lr,
                    let_lr=args.omni_let_lr,
                    init_lwc_logit=args.omni_init_lwc_logit,
                    min_let_scale=args.omni_min_let_scale,
                    max_let_scale=args.omni_max_let_scale,
                    max_grad_norm=args.omni_max_grad_norm,
                ),
                capture_layer_input_batches=capture_layer_input_batches,
                act_quant_mode=args.act_quant_mode,
                checkpoint_dir=output_file.parent / "omniquant_calibration",
            )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    config = {
        "method": args.mode,
        "task": args.task,
        "layers": layer_indices,
        "smoothquant_alpha": args.smoothquant_alpha,
        "smooth_scope": args.smooth_scope,
        "smooth_fold": args.smooth_fold,
        "smoothquant_min_scale": args.smoothquant_min_scale,
        "smoothquant_max_scale": args.smoothquant_max_scale,
        "gptq_damp_percent": args.gptq_damp_percent,
        "gptq_block_size": args.gptq_block_size,
        "omni_lwc": args.omni_lwc,
        "omni_weight_quant_scheme": args.omni_weight_quant_scheme,
        "omni_let": args.omni_let,
        "omni_let_mode": args.omni_let_mode,
        "omni_epochs": args.omni_epochs,
        "omni_lwc_lr": args.omni_lwc_lr,
        "omni_let_lr": args.omni_let_lr,
        "omni_init_lwc_logit": args.omni_init_lwc_logit,
        "omni_min_let_scale": args.omni_min_let_scale,
        "omni_max_let_scale": args.omni_max_let_scale,
        "omni_max_grad_norm": args.omni_max_grad_norm,
        "omni_checkpoint_dir": (
            str(output_file.parent / "omniquant_calibration") if args.mode == "omniquant" else None
        ),
        "act_quant": args.act_quant,
        "act_quant_mode": args.act_quant_mode,
        "weight_quant_format": "none" if args.mode == "full_precision" else args.weight_quant_format,
        "activation_quant_format": "none" if args.mode == "full_precision" else args.activation_quant_format,
        "quantization_execution": (
            "full_precision"
            if args.mode == "full_precision"
            else "fake_qdq_then_f_linear_in_model_dtype"
        ),
        "data_dir": args.data_dir,
        "calib_data_dir": calib_data_dir,
        "eval_data_dir": eval_data_dir,
        "split": args.split,
        "calib_split": calib_split,
        "calib_sample_size": args.calib_sample_size,
        "calib_offset": args.calib_offset,
        "eval_sample_size": args.eval_sample_size,
        "eval_offset": args.eval_offset,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "compute_sid_ppl": args.compute_sid_ppl,
        "sid_ppl_max_items": args.sid_ppl_max_items,
        "seed": args.seed,
        "baseline_summaries": summaries_to_jsonable(baseline_summaries),
    }
    config_filename = {
        "full_precision": "full_precision_config.json",
        "smoothquant_w8a8": "smoothquant_w8a8_config.json",
        "gptq_fp8_w8a8": "gptq_fp8_w8a8_config.json",
        "omniquant": "omniquant_config.json",
    }.get(
        args.mode,
        (
            "baseline_w8a8_config.json"
            if (
                args.mode == "baseline_w8a8"
                and args.weight_quant_format == "fp8_e4m3fn"
                and args.activation_quant_format == "fp8_e4m3fn"
            )
            else "baseline_qdq_config.json"
        ),
    )
    (output_file.parent / config_filename).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    test_data = load_task_data(
        task_name=args.task,
        tokenizer=tokenizer,
        data_dir=str(resolve_repo_path(eval_data_dir)),
        split=args.split,
        sample_size=parse_sample_size(args.eval_sample_size),
        sample_offset=args.eval_offset,
    )
    test_items = list(test_data.items())
    generations: dict[str, list[str]] = {}
    sample_aux_metrics: dict[str, dict[str, Any]] = {}
    sid_tf_total_time = 0.0
    start = time.time()
    for sample_id, sample in tqdm(
        test_items,
        total=len(test_items),
        desc=f"{args.mode} {args.task} generation",
    ):
        prompt = format_prompt(sample["prompt"], prompt_token)
        if args.compute_sid_ppl:
            sid_tf_start = time.time()
            sample_aux_metrics[sample_id] = compute_sid_teacher_forcing_metrics(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                ground_truth=sample.get("ground_truth", ""),
                input_device=input_device,
                max_items=args.sid_ppl_max_items,
            )
            sid_tf_total_time += time.time() - sid_tf_start
        generations[sample_id] = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            input_device=input_device,
            args=args,
        )
    raw_total_time = time.time() - start
    total_time = raw_total_time - sid_tf_total_time if args.compute_sid_ppl else raw_total_time

    sid_tf_metrics: dict[str, Any] = {}
    if args.compute_sid_ppl:
        sid_tf_metrics = aggregate_sid_teacher_forcing_metrics(
            sample_aux_metrics,
            max_items=args.sid_ppl_max_items,
        )
        sid_tf_metrics["sid_tf_total_time"] = sid_tf_total_time
        sid_tf_metrics["sid_tf_avg_time_per_sample"] = sid_tf_total_time / len(test_items) if test_items else 0.0
        config["sid_teacher_forcing_metrics"] = sid_tf_metrics
        (output_file.parent / config_filename).write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    save_results(
        output_file=output_file,
        model_name=model_name,
        split=args.split,
        test_data=test_data,
        generations=generations,
        total_time=total_time,
        config=config,
        task_name=args.task,
        sample_aux_metrics=sample_aux_metrics if args.compute_sid_ppl else None,
    )
    if args.evaluate:
        maybe_evaluate(args.output_dir, eval_data_dir, args.overwrite, task_name=args.task)
        if sid_tf_metrics:
            merge_sid_teacher_forcing_metrics_into_eval(
                output_dir=args.output_dir,
                model_name=model_name,
                split=args.split,
                metrics=sid_tf_metrics,
                task_name=args.task,
            )


if __name__ == "__main__":
    main()
