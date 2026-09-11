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
from .flatquant import (
    DEFAULT_FLATQUANT_EPOCHS,
    DEFAULT_FLATQUANT_INIT_LAC_LOGIT,
    DEFAULT_FLATQUANT_INIT_LWC_LOGIT,
    DEFAULT_FLATQUANT_LAC_LR,
    DEFAULT_FLATQUANT_LWC_LR,
    DEFAULT_FLATQUANT_TRANSFORM_LR,
    DEFAULT_FLATQUANT_WEIGHT_DECAY,
    FlatQuantCoreConfig,
    apply_flatquant_core_layers,
    restore_flatquant_core_layers_from_checkpoints,
)
from .gptq import (
    DEFAULT_GPTQ_BLOCK_SIZE,
    DEFAULT_GPTQ_DAMP_PERCENT,
    collect_gptq_hessians,
    gptq_quantized_module_from_hessians,
)
from .modules import BaselineFakeQuantLinear
from .omniquant import (
    DEFAULT_OMNIQUANT_EPOCHS,
    DEFAULT_OMNIQUANT_INIT_LWC_LOGIT,
    DEFAULT_OMNIQUANT_LET_LR,
    DEFAULT_OMNIQUANT_LWC_LR,
    DEFAULT_OMNIQUANT_MAX_GRAD_NORM,
    DEFAULT_OMNIQUANT_WEIGHT_DECAY,
    OMNIQUANT_CALIBRATION_COMPUTE_DTYPE,
    OMNIQUANT_CALIBRATION_FORWARD_MODE,
    OMNIQUANT_LOSS_COMPUTE_DTYPE,
    OMNIQUANT_QUANTIZATION_COMPUTE_DTYPE,
    OmniQuantConfig,
    apply_omniquant_layers,
    restore_omniquant_layers_from_checkpoints,
)
from .quant import (
    ActQuant,
    ActQuantMode,
    FAKE_QUANT_FORWARD_MODE,
    FAKE_QUANT_LOSS_DTYPE,
    FAKE_QUANT_OPERATOR_DTYPE,
    FAKE_QUANT_QDQ_COMPUTE_DTYPE,
    QUANT_FORMAT_CHOICES,
    WEIGHT_QUANT_SCHEME_CHOICES,
    QuantFormat,
    WeightQuantScheme,
    resolve_weight_quant_scheme,
)
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
from shared.paths import benchmark_data_root, fake_results_root, model_root  # noqa: E402


DEFAULT_MODEL_PATH = str(model_root() / "1.7B")
DEFAULT_DATA_DIR = str(benchmark_data_root())
DEFAULT_OUTPUT_DIR = str(fake_results_root() / "recommender" / "ptq_ad")
DEFAULT_TASK = "ad"
TASK_CHOICES = ("ad", "product", "video", "label_pred")
DEFAULT_SPLIT = "test"
DEFAULT_CALIB_OFFSET = 0
DEFAULT_EVAL_OFFSET = 0
DEFAULT_ACT_QUANT: ActQuant = "per_token"
DEFAULT_ACT_QUANT_MODE: ActQuantMode = "shared_input"
DEFAULT_WEIGHT_QUANT_FORMAT: QuantFormat = "int8"
DEFAULT_ACTIVATION_QUANT_FORMAT: QuantFormat = "int8"
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
SID_SLOT_NAMES = ("a", "b", "c")
SID_SLOT_TOKEN_RES = {
    slot: re.compile(rf"<s_{slot}_(?P<index>\d+)>")
    for slot in SID_SLOT_NAMES
}


def sid_slot_token_ids(tokenizer: Any, slot: str) -> tuple[int, ...]:
    """Return one semantic-ID slot's token IDs in semantic-index order."""

    if slot not in SID_SLOT_TOKEN_RES:
        raise ValueError(f"Unsupported SID slot {slot!r}; expected one of {SID_SLOT_NAMES}.")
    indexed_ids: list[tuple[int, int]] = []
    pattern = SID_SLOT_TOKEN_RES[slot]
    for token, token_id in tokenizer.get_vocab().items():
        match = pattern.fullmatch(token)
        if match is not None:
            indexed_ids.append((int(match.group("index")), int(token_id)))
    if not indexed_ids:
        raise ValueError(f"The tokenizer vocabulary does not contain any <s_{slot}_*> tokens.")

    indexed_ids.sort()
    semantic_indices = [index for index, _token_id in indexed_ids]
    token_ids = [token_id for _index, token_id in indexed_ids]
    if len(set(semantic_indices)) != len(semantic_indices):
        raise ValueError(f"The tokenizer contains duplicate SID_{slot} semantic indices.")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"The tokenizer maps multiple SID_{slot} tokens to one token ID.")
    return tuple(token_ids)


def sid_a_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Backward-compatible helper for the SID_a semantic vocabulary."""

    return sid_slot_token_ids(tokenizer, "a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneRec recommendation evaluation with full precision or composable fake-QDQ quantization."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, choices=TASK_CHOICES)
    parser.add_argument(
        "--mode",
        default="baseline_qdq",
        choices=[
            "full_precision",
            "baseline_w8a8",
            "baseline_qdq",
            "smoothquant_w8a8",
            "gptq_fp8_w8a8",
            "omniquant",
            "flatquant_core",
        ],
    )
    parser.add_argument(
        "--weight_quant_format",
        choices=QUANT_FORMAT_CHOICES,
        default=DEFAULT_WEIGHT_QUANT_FORMAT,
        help=(
            "Fake-QDQ weight format. Weights use per-output-channel scaling by "
            "default, or finer input-dimension groups with --weight_group_size."
        ),
    )
    parser.add_argument(
        "--weight_quant_scheme",
        "--omni_weight_quant_scheme",
        dest="weight_quant_scheme",
        choices=WEIGHT_QUANT_SCHEME_CHOICES,
        default=None,
        help=(
            "Weight quantizer scheme for RTN, SmoothQuant, and OmniQuant. By "
            "default integer weights use asymmetric affine QDQ with a zero point, while floating "
            "formats use required zero-centered symmetric QDQ. The old "
            "--omni_weight_quant_scheme name remains a compatibility alias."
        ),
    )
    parser.add_argument(
        "--weight_group_size",
        type=int,
        default=0,
        help=(
            "RTN/SmoothQuant/OmniQuant-LWC weight group size along Linear "
            "in_features. 0 keeps one range per output channel."
        ),
    )
    parser.add_argument(
        "--omni_lwc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable learnable weight clipping per output channel, or per "
            "input group when --weight_group_size is positive."
        ),
    )
    parser.add_argument("--no-omni-lwc", dest="omni_lwc", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--omni_symmetric_lwc_mode",
        choices=("absmax", "two_sided"),
        default="absmax",
        help=(
            "Symmetric weight LWC parameterization: one absmax factor, or "
            "independent positive/negative clipping factors before the same "
            "zero-centered FP/INT QDQ."
        ),
    )
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
    parser.add_argument(
        "--omni_let_init",
        choices=("smoothquant", "ones"),
        default="smoothquant",
        help=(
            "LET initialization: SmoothQuant scales, or identity scales. "
            "The latter initializes every LET scale to one."
        ),
    )
    parser.add_argument(
        "--smoothquant_alpha",
        type=float,
        default=DEFAULT_SMOOTHQUANT_ALPHA,
        help=(
            "SmoothQuant exponent in [0, 1]. It controls both standard "
            "SmoothQuant and OmniQuant smoothquant-initialized LET."
        ),
    )
    parser.add_argument(
        "--omni_load_checkpoint_dir",
        default=None,
        help="Restore OmniQuant calibration checkpoints and skip blockwise optimization.",
    )
    parser.add_argument(
        "--omni_prefix_checkpoint_dir",
        default=None,
        help="Restore every block before the final block, then optimize only the final block.",
    )
    parser.add_argument(
        "--omni_final_objective",
        choices=("mse", "lfq_ce"),
        default="mse",
        help="Final-block objective: hidden-state MSE or SID-slot LFQ soft CE.",
    )
    parser.add_argument(
        "--omni_lfq_token_scope",
        choices=("sid_slots",),
        default="sid_slots",
        help="LFQ token positions: the three positions predicting SID_a, SID_b, and SID_c.",
    )
    parser.add_argument(
        "--omni_lfq_vocab_scope",
        choices=("s_abc",),
        default="s_abc",
        help="LFQ output vocabularies: the slot-specific <s_a_*>, <s_b_*>, and <s_c_*> sets.",
    )
    parser.add_argument(
        "--omni_lfq_slot_weights",
        type=float,
        nargs=3,
        metavar=("A", "B", "C"),
        default=(1.0, 1.0, 1.0),
        help=(
            "Non-negative SID_a/SID_b/SID_c LFQ weights. They are normalized to "
            "sum to one; the default gives every slot equal weight."
        ),
    )
    parser.add_argument(
        "--omni_lfq_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Weight of final-block GT-prefix LFQ soft cross-entropy."
        ),
    )
    parser.add_argument(
        "--omni_lfq_boundary_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the tie-aware FP top-K versus next-N boundary gap loss. "
            "Zero preserves the original ABC-LFQ objective."
        ),
    )
    parser.add_argument(
        "--omni_lfq_boundary_topk",
        type=int,
        default=32,
        help="Number of FP top-ranked SID tokens treated as boundary positives.",
    )
    parser.add_argument(
        "--omni_lfq_boundary_negative_count",
        type=int,
        default=32,
        help="Number of FP tokens immediately below top-K used as negatives.",
    )
    parser.add_argument(
        "--omni_lfq_boundary_tie_threshold",
        type=float,
        default=1e-2,
        help="Ignore teacher positive/negative logit gaps at or below this value.",
    )
    parser.add_argument(
        "--omni_lfq_boundary_gap_scale",
        type=float,
        default=1.0,
        help="Teacher logit gap at which a boundary pair reaches unit weight.",
    )
    parser.add_argument(
        "--omni_epochs",
        type=int,
        default=DEFAULT_OMNIQUANT_EPOCHS,
    )
    parser.add_argument(
        "--omni_epoch_eval_interval",
        type=int,
        default=0,
        help=(
            "Evaluate fixed parameters on the full calibration set every N epochs "
            "and restore the best evaluated epoch. Zero keeps final-only evaluation."
        ),
    )
    parser.add_argument(
        "--omni_validation_sample_size",
        type=int,
        default=0,
        help=(
            "Hold out the last N loaded LFQ calibration samples for per-epoch "
            "validation. They never participate in backprop or checkpoint selection."
        ),
    )
    parser.add_argument(
        "--omni_train_sample_size",
        type=int,
        default=0,
        help=(
            "With LFQ validation enabled, train on only the first N loaded "
            "samples while keeping the final validation tail fixed. Zero uses "
            "all samples before the validation tail."
        ),
    )
    parser.add_argument(
        "--omni_lwc_lr",
        type=float,
        default=DEFAULT_OMNIQUANT_LWC_LR,
    )
    parser.add_argument(
        "--omni_let_lr",
        type=float,
        default=DEFAULT_OMNIQUANT_LET_LR,
    )
    parser.add_argument(
        "--omni_weight_decay",
        type=float,
        default=DEFAULT_OMNIQUANT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--omni_init_lwc_logit",
        type=float,
        default=DEFAULT_OMNIQUANT_INIT_LWC_LOGIT,
    )
    parser.add_argument(
        "--omni_max_grad_norm",
        type=float,
        default=DEFAULT_OMNIQUANT_MAX_GRAD_NORM,
        help="Optional gradient clipping norm. The official default disables clipping.",
    )
    parser.add_argument(
        "--flat_load_checkpoint_dir",
        default=None,
        help="Restore formal FlatQuant checkpoints and skip calibration.",
    )
    parser.add_argument(
        "--flat_prefix_checkpoint_dir",
        default=None,
        help=(
            "Restore layers before the final block from a formal FlatQuant prefix, "
            "then optimize only the final block on the current calibration split."
        ),
    )
    parser.add_argument(
        "--flat_finetune_checkpoint_dir",
        default=None,
        help=(
            "Restore one complete MSE-trained FlatQuant checkpoint, keep layers "
            "before the final block fixed, and continue optimizing the final block."
        ),
    )
    parser.add_argument(
        "--flat_transform_init",
        choices=("identity", "random_orthogonal"),
        default="random_orthogonal",
        help="Initialize official Cayley factors (random orthogonal by default).",
    )
    parser.add_argument(
        "--flat_transform_kind",
        choices=("kronecker", "smoothquant"),
        default="kronecker",
        help=(
            "Use learnable Kronecker transforms, or fixed SmoothQuant diagonal "
            "transforms inside the matched FlatQuant MSE/LWC harness."
        ),
    )
    parser.add_argument(
        "--flat_learn_transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the official matrix and diagonal transforms.",
    )
    parser.add_argument(
        "--flat_lac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply official two-sided activation clipping parameters.",
    )
    parser.add_argument(
        "--flat_learn_lac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Optimize LAC when it is enabled. Disable this to retain restored "
            "MSE-trained clipping values while freezing them during LFQ."
        ),
    )
    parser.add_argument(
        "--flat_diag_alpha",
        type=float,
        default=0.5,
        help="Training-prefix statistic exponent for diagonal initialization.",
    )
    parser.add_argument(
        "--flat_epochs",
        type=int,
        default=DEFAULT_FLATQUANT_EPOCHS,
    )
    parser.add_argument(
        "--flat_train_sample_size",
        type=int,
        default=0,
        help="Use only the first N pre-held-out samples for FlatQuant backprop.",
    )
    parser.add_argument(
        "--flat_validation_sample_size",
        type=int,
        default=0,
        help="Reserve the final N calibration samples as diagnostic held-out data.",
    )
    parser.add_argument(
        "--flat_epoch_eval_interval",
        type=int,
        default=0,
        help=(
            "Evaluate fixed train/held-out MSE every N epochs and restore the "
            "best training-MSE epoch. Zero keeps final-only evaluation."
        ),
    )
    parser.add_argument(
        "--flat_transform_lr",
        type=float,
        default=DEFAULT_FLATQUANT_TRANSFORM_LR,
    )
    parser.add_argument(
        "--flat_lwc_lr",
        type=float,
        default=DEFAULT_FLATQUANT_LWC_LR,
    )
    parser.add_argument(
        "--flat_lac_lr",
        type=float,
        default=DEFAULT_FLATQUANT_LAC_LR,
    )
    parser.add_argument(
        "--flat_init_lwc_logit",
        type=float,
        default=DEFAULT_FLATQUANT_INIT_LWC_LOGIT,
    )
    parser.add_argument(
        "--flat_init_lac_logit",
        type=float,
        default=DEFAULT_FLATQUANT_INIT_LAC_LOGIT,
    )
    parser.add_argument(
        "--flat_weight_decay",
        type=float,
        default=DEFAULT_FLATQUANT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--flat_normalize_mse_gradient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official FlatQuant loss/loss.detach gradient normalization.",
    )
    parser.add_argument(
        "--flat_max_grad_norm",
        type=float,
        default=None,
    )
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
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--eval_sample_size", default="full")
    parser.add_argument(
        "--eval_num_shards",
        type=int,
        default=1,
        help=(
            "Split the selected evaluation samples into this many deterministic "
            "round-robin shards. Each shard must run in a separate process."
        ),
    )
    parser.add_argument(
        "--eval_shard_id",
        type=int,
        default=0,
        help="Zero-based evaluation shard index used with --eval_num_shards.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--calibration_only",
        action="store_true",
        help="Stop after calibration checkpoints/config are written; skip generation and metrics.",
    )
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
    if args.weight_group_size < 0:
        raise ValueError("--weight_group_size must be 0 or a positive integer.")
    groupwise_modes = {"baseline_w8a8", "baseline_qdq", "smoothquant_w8a8", "omniquant", "flatquant_core"}
    if args.weight_group_size > 0 and args.mode not in groupwise_modes:
        raise ValueError(
            "--weight_group_size is currently supported only for RTN "
            "(baseline_qdq/baseline_w8a8), SmoothQuant, OmniQuant, and FlatQuant-core."
        )
    if args.weight_group_size > 0 and args.weight_quant_format == "none":
        raise ValueError("--weight_group_size requires a quantized weight format.")
    args.weight_quant_scheme = resolve_weight_quant_scheme(
        args.weight_quant_format, args.weight_quant_scheme
    )
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


def build_lfq_sid_slot_batches(
    *,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    prompt_token: str,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Build prompt+a_gt+b_gt batches whose last three positions predict a/b/c."""

    prompt_token_id = int(tokenizer.convert_tokens_to_ids(prompt_token))
    batches: list[dict[str, torch.Tensor]] = []
    for sample_idx, sample in enumerate(samples):
        ground_truth = str(sample.get("ground_truth", ""))
        sid_match = next(SID_ITEM_RE.finditer(ground_truth), None)
        if sid_match is None:
            raise ValueError(
                f"LFQ calibration sample {sample_idx} has no parseable ground-truth SID."
            )
        sid_ids = tuple(
            int(token_id)
            for token_id in tokenizer(
                sid_match.group("sid"),
                add_special_tokens=False,
            )["input_ids"]
        )
        if len(sid_ids) != len(SID_SLOT_NAMES):
            raise ValueError(
                f"LFQ calibration SID must tokenize to exactly three tokens, got {sid_ids}."
            )
        for slot, token_id in zip(SID_SLOT_NAMES, sid_ids):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if SID_SLOT_TOKEN_RES[slot].fullmatch(token) is None:
                raise ValueError(
                    f"LFQ calibration expected an SID_{slot} token, got {token!r}."
                )

        prompt = format_prompt(str(sample["prompt"]), prompt_token)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if int(input_ids[0, -1]) != prompt_token_id:
            raise ValueError(
                "LFQ prompt must end with the SID-begin token before appending a_gt and b_gt."
            )
        prefix_ids = torch.tensor(
            sid_ids[:2],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).view(1, -1)
        encoded["input_ids"] = torch.cat((input_ids, prefix_ids), dim=-1)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        else:
            encoded["attention_mask"] = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 2),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=-1,
            )
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
    weight_quant_format: QuantFormat = "fp8_e4m3fn",
    weight_quant_scheme: WeightQuantScheme | None = None,
    weight_group_size: int | None = None,
    activation_quant_format: QuantFormat = "fp8_e4m3fn",
    smoothquant_alpha: float = DEFAULT_SMOOTHQUANT_ALPHA,
    smoothquant_min_scale: float | None = DEFAULT_SMOOTHQUANT_MIN_SCALE,
    smoothquant_max_scale: float | None = DEFAULT_SMOOTHQUANT_MAX_SCALE,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    smooth_fold: bool = True,
) -> dict[int, BaselineQuantSummary]:
    """Apply SmoothQuant followed by format-explicit weight/activation fake QDQ."""
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
            weight_quant_format=weight_quant_format,
            weight_quant_scheme=weight_quant_scheme,
            weight_group_size=weight_group_size,
            activation_quant_format=activation_quant_format,
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
            f"[smoothquant_qdq w={weight_quant_format}/{weight_quant_scheme} "
            f"a={activation_quant_format}] layer={layer_idx} replaced_linears={replaced}, "
            f"weight_group_size={weight_group_size or 'per_channel'}, "
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
    weight_quant_scheme: WeightQuantScheme | None = None,
    weight_group_size: int | None = None,
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
                weight_quant_scheme=weight_quant_scheme,
                weight_group_size=weight_group_size,
                activation_quant_format=activation_quant_format,
            )
            summary = BaselineQuantSummary(replaced_linears=1, skipped_linears=0)
        else:
            summary = apply_baseline_qdq(
                layer,
                weight_quant_format=weight_quant_format,
                weight_quant_scheme=weight_quant_scheme,
                weight_group_size=weight_group_size,
                activation_quant_format=activation_quant_format,
                act_quant_mode=act_quant_mode,
            )
        summaries[layer_idx] = summary
        print(
            f"[baseline_qdq w={weight_quant_format}/{weight_quant_scheme} a={activation_quant_format}] "
            f"layer={layer_idx} replaced_linears={summary.replaced_linears} "
            f"weight_group_size={weight_group_size or 'per_channel'} "
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


def resolve_classification_token_ids(
    tokenizer: Any,
    target_tokens: Sequence[str],
) -> tuple[int, ...]:
    """Resolve classification labels that must each be exactly one model token."""

    if len(target_tokens) < 2:
        raise ValueError(
            f"Classification requires at least two target tokens, got {target_tokens!r}."
        )
    token_ids: list[int] = []
    for token in target_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Classification target {token!r} must encode to one token, got {encoded}."
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            f"Classification targets map to duplicate token IDs: {target_tokens!r}."
        )
    return tuple(token_ids)


def classify_one(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    input_device: torch.device,
    target_tokens: Sequence[str],
    target_token_ids: Sequence[int],
) -> list[str]:
    """Return conditional target-token probabilities in benchmark JSON format."""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {
        key: value.to(input_device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1]
        selected_logits = logits[list(target_token_ids)].float()
        probabilities = torch.softmax(selected_logits, dim=-1).detach().cpu().tolist()

    payload = {
        str(token): float(probability)
        for token, probability in zip(target_tokens, probabilities)
    }
    return [json.dumps(payload, ensure_ascii=False, separators=(",", ":"))]


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


def eval_run_output_dir(
    output_dir: str | os.PathLike[str],
    *,
    num_shards: int,
    shard_id: int,
) -> Path:
    """Return a collision-free output directory for one evaluation process."""

    output_root = resolve_repo_path(output_dir)
    if num_shards == 1:
        return output_root
    shard_root = Path(f"{output_root}.shards")
    return shard_root / f"shard_{shard_id:03d}_of_{num_shards:03d}"


def select_round_robin_eval_shard(
    data: Mapping[str, Mapping[str, Any]],
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Mapping[str, Any]]:
    """Select one deterministic shard while preserving its source order."""

    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    if num_shards == 1:
        return dict(data)
    return dict(list(data.items())[shard_id::num_shards])


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
        if hasattr(summary, "initial_loss") and not hasattr(summary, "transform_factors"):
            item["initial_loss"] = summary.initial_loss
            item["final_loss"] = summary.final_loss
            item["objective"] = getattr(summary, "objective", "mse")
            item["let_scales"] = list(summary.let_scales)
            initial_slot_losses = dict(
                getattr(summary, "initial_lfq_slot_losses", ())
            )
            final_slot_losses = dict(
                getattr(summary, "final_lfq_slot_losses", ())
            )
            if initial_slot_losses:
                item["initial_lfq_slot_losses"] = initial_slot_losses
                item["final_lfq_slot_losses"] = final_slot_losses
            lfq_slot_weights = getattr(summary, "lfq_slot_weights", None)
            if lfq_slot_weights is not None:
                item["lfq_slot_weights"] = list(lfq_slot_weights)
                item["lfq_loss_weight"] = getattr(summary, "lfq_loss_weight", 1.0)
                item["lfq_boundary_loss_weight"] = getattr(
                    summary, "lfq_boundary_loss_weight", 0.0
                )
            initial_boundary_loss = getattr(
                summary, "initial_lfq_boundary_loss", None
            )
            final_boundary_loss = getattr(
                summary, "final_lfq_boundary_loss", None
            )
            if initial_boundary_loss is not None and final_boundary_loss is not None:
                item["initial_lfq_boundary_loss"] = initial_boundary_loss
                item["final_lfq_boundary_loss"] = final_boundary_loss
                item["initial_lfq_boundary_slot_losses"] = dict(
                    getattr(summary, "initial_lfq_boundary_slot_losses", ())
                )
                item["final_lfq_boundary_slot_losses"] = dict(
                    getattr(summary, "final_lfq_boundary_slot_losses", ())
                )
            initial_mse_loss = getattr(summary, "initial_mse_loss", None)
            final_mse_loss = getattr(summary, "final_mse_loss", None)
            if initial_mse_loss is not None and final_mse_loss is not None:
                item["initial_mse_loss"] = initial_mse_loss
                item["final_mse_loss"] = final_mse_loss
            best_epoch = getattr(summary, "best_epoch", None)
            if best_epoch is not None:
                item["best_epoch"] = best_epoch
            epoch_metrics = getattr(summary, "epoch_metrics", ())
            if epoch_metrics:
                item["epoch_metrics"] = [
                    {
                        "epoch": metric.epoch,
                        "train_loss": metric.train_loss,
                        "mean_grad_norm": metric.mean_grad_norm,
                        "max_grad_norm": metric.max_grad_norm,
                        "eval_loss": metric.eval_loss,
                        "eval_mse_loss": metric.eval_mse_loss,
                        "train_lfq_boundary_loss": (
                            metric.train_lfq_boundary_loss
                        ),
                        "validation_lfq_boundary_loss": (
                            metric.validation_lfq_boundary_loss
                        ),
                        "validation_loss": metric.validation_loss,
                        "validation_lfq_slot_losses": dict(
                            metric.validation_lfq_slot_losses
                        ),
                    }
                    for metric in epoch_metrics
                ]
        elif hasattr(summary, "initial_mse_loss"):
            item.update(
                {
                    "initial_mse_loss": summary.initial_mse_loss,
                    "final_mse_loss": summary.final_mse_loss,
                    "initial_validation_mse_loss": (
                        summary.initial_validation_mse_loss
                    ),
                    "final_validation_mse_loss": (
                        summary.final_validation_mse_loss
                    ),
                    "objective": summary.objective,
                    "initial_loss": summary.initial_loss,
                    "final_loss": summary.final_loss,
                    "initial_lfq_slot_losses": dict(summary.initial_lfq_slot_losses),
                    "final_lfq_slot_losses": dict(summary.final_lfq_slot_losses),
                    "lfq_slot_weights": (
                        None
                        if summary.lfq_slot_weights is None
                        else list(summary.lfq_slot_weights)
                    ),
                    "lfq_loss_weight": summary.lfq_loss_weight,
                    "lfq_boundary_loss_weight": summary.lfq_boundary_loss_weight,
                    "initial_lfq_boundary_loss": summary.initial_lfq_boundary_loss,
                    "final_lfq_boundary_loss": summary.final_lfq_boundary_loss,
                    "initial_lfq_boundary_slot_losses": dict(
                        summary.initial_lfq_boundary_slot_losses
                    ),
                    "final_lfq_boundary_slot_losses": dict(
                        summary.final_lfq_boundary_slot_losses
                    ),
                    "initial_validation_loss": summary.initial_validation_loss,
                    "final_validation_loss": summary.final_validation_loss,
                    "initial_validation_lfq_slot_losses": dict(
                        summary.initial_validation_lfq_slot_losses
                    ),
                    "final_validation_lfq_slot_losses": dict(
                        summary.final_validation_lfq_slot_losses
                    ),
                    "best_epoch": summary.best_epoch,
                    "transform_factors": [list(value) for value in summary.transform_factors],
                    "effective_transform_parameters": summary.effective_transform_parameters,
                    "trainable_transform_parameters": summary.trainable_transform_parameters,
                    "trainable_lwc_parameters": summary.trainable_lwc_parameters,
                    "trainable_lac_parameters": summary.trainable_lac_parameters,
                    "epoch_metrics": [
                        {
                            "epoch": metric.epoch,
                            "train_mse": metric.train_mse,
                            "eval_mse": metric.eval_mse,
                            "mean_grad_norm": metric.mean_grad_norm,
                            "max_grad_norm": metric.max_grad_norm,
                            "transform_lr": metric.transform_lr,
                            "lwc_lr": metric.lwc_lr,
                            "lac_lr": metric.lac_lr,
                            "validation_mse": metric.validation_mse,
                            "train_loss": metric.train_loss,
                            "eval_loss": metric.eval_loss,
                            "validation_loss": metric.validation_loss,
                            "validation_lfq_slot_losses": dict(
                                metric.validation_lfq_slot_losses
                            ),
                            "train_lfq_boundary_loss": (
                                metric.train_lfq_boundary_loss
                            ),
                            "validation_lfq_boundary_loss": (
                                metric.validation_lfq_boundary_loss
                            ),
                        }
                        for metric in summary.epoch_metrics
                    ],
                }
            )
        serialized[str(layer_idx)] = item
    return serialized


def build_omniquant_config(args: argparse.Namespace) -> OmniQuantConfig:
    return OmniQuantConfig(
        weight_quant_format=args.weight_quant_format,
        activation_quant_format=args.activation_quant_format,
        weight_quant_scheme=args.weight_quant_scheme,
        symmetric_lwc_mode=args.omni_symmetric_lwc_mode,
        weight_group_size=args.weight_group_size,
        use_lwc=args.omni_lwc,
        use_let=args.omni_let,
        learn_let=args.omni_let_mode == "learned",
        let_init=args.omni_let_init,
        smoothquant_alpha=args.smoothquant_alpha,
        final_objective=args.omni_final_objective,
        lfq_token_scope=args.omni_lfq_token_scope,
        lfq_vocab_scope=args.omni_lfq_vocab_scope,
        lfq_slot_weights=tuple(args.omni_lfq_slot_weights),
        lfq_loss_weight=args.omni_lfq_loss_weight,
        lfq_boundary_loss_weight=args.omni_lfq_boundary_loss_weight,
        lfq_boundary_topk=args.omni_lfq_boundary_topk,
        lfq_boundary_negative_count=args.omni_lfq_boundary_negative_count,
        lfq_boundary_tie_threshold=args.omni_lfq_boundary_tie_threshold,
        lfq_boundary_gap_scale=args.omni_lfq_boundary_gap_scale,
        epochs=args.omni_epochs,
        validation_sample_size=args.omni_validation_sample_size,
        train_sample_size=args.omni_train_sample_size,
        epoch_eval_interval=args.omni_epoch_eval_interval,
        lwc_lr=args.omni_lwc_lr,
        let_lr=args.omni_let_lr,
        weight_decay=args.omni_weight_decay,
        init_lwc_logit=args.omni_init_lwc_logit,
        max_grad_norm=args.omni_max_grad_norm,
    )


def build_flatquant_config(args: argparse.Namespace) -> FlatQuantCoreConfig:
    return FlatQuantCoreConfig(
        weight_quant_format=args.weight_quant_format,
        activation_quant_format=args.activation_quant_format,
        weight_quant_scheme=args.weight_quant_scheme,
        weight_group_size=args.weight_group_size,
        use_lwc=args.omni_lwc,
        use_lac=args.flat_lac,
        learn_lac=args.flat_learn_lac,
        learn_transform=args.flat_learn_transform,
        transform_kind=args.flat_transform_kind,
        transform_init=args.flat_transform_init,
        smoothquant_alpha=args.smoothquant_alpha,
        diag_alpha=args.flat_diag_alpha,
        final_objective=args.omni_final_objective,
        lfq_token_scope=args.omni_lfq_token_scope,
        lfq_vocab_scope=args.omni_lfq_vocab_scope,
        lfq_slot_weights=tuple(args.omni_lfq_slot_weights),
        lfq_loss_weight=args.omni_lfq_loss_weight,
        lfq_boundary_loss_weight=args.omni_lfq_boundary_loss_weight,
        lfq_boundary_topk=args.omni_lfq_boundary_topk,
        lfq_boundary_negative_count=args.omni_lfq_boundary_negative_count,
        lfq_boundary_tie_threshold=args.omni_lfq_boundary_tie_threshold,
        lfq_boundary_gap_scale=args.omni_lfq_boundary_gap_scale,
        epochs=args.flat_epochs,
        train_sample_size=args.flat_train_sample_size,
        validation_sample_size=args.flat_validation_sample_size,
        epoch_eval_interval=args.flat_epoch_eval_interval,
        transform_lr=args.flat_transform_lr,
        lwc_lr=args.flat_lwc_lr,
        lac_lr=args.flat_lac_lr,
        weight_decay=args.flat_weight_decay,
        init_lwc_logit=args.flat_init_lwc_logit,
        init_lac_logit=args.flat_init_lac_logit,
        normalize_mse_gradient=args.flat_normalize_mse_gradient,
        max_grad_norm=args.flat_max_grad_norm,
    )


def load_hf_model(args: argparse.Namespace) -> nn.Module:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model = model.to(args.device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.task == "label_pred" and args.compute_sid_ppl:
        raise ValueError("--compute_sid_ppl is only valid for SID recommendation tasks.")
    if args.sid_ppl_max_items <= 0:
        raise ValueError(f"--sid_ppl_max_items must be positive, got {args.sid_ppl_max_items}")
    if args.omni_validation_sample_size < 0:
        raise ValueError("--omni_validation_sample_size must be non-negative.")
    if args.omni_train_sample_size < 0:
        raise ValueError("--omni_train_sample_size must be non-negative.")
    if args.omni_train_sample_size > 0 and args.omni_validation_sample_size == 0:
        raise ValueError(
            "--omni_train_sample_size requires --omni_validation_sample_size."
        )
    if args.omni_validation_sample_size > 0 and (
        args.mode != "omniquant"
        or args.omni_final_objective != "lfq_ce"
        or args.omni_load_checkpoint_dir
        or not args.omni_prefix_checkpoint_dir
    ):
        raise ValueError(
            "--omni_validation_sample_size requires final-block LFQ training "
            "with --omni_final_objective lfq_ce and "
            "--omni_prefix_checkpoint_dir."
        )
    if args.eval_num_shards <= 0:
        raise ValueError(
            f"--eval_num_shards must be positive, got {args.eval_num_shards}"
        )
    if args.eval_shard_id < 0 or args.eval_shard_id >= args.eval_num_shards:
        raise ValueError(
            "--eval_shard_id must be in "
            f"[0, {args.eval_num_shards}), got {args.eval_shard_id}"
        )
    if args.eval_num_shards > 1 and args.evaluate:
        raise ValueError(
            "Do not pass --evaluate to an individual shard. Merge all shards with "
            "python -m fake_quant.merge_eval_shards, which computes metrics once."
        )
    if (
        args.eval_num_shards > 1
        and args.mode == "omniquant"
        and not args.omni_load_checkpoint_dir
    ):
        raise ValueError(
            "Sharded OmniQuant evaluation requires --omni_load_checkpoint_dir so "
            "blockwise optimization is performed only once."
        )
    if (
        args.eval_num_shards > 1
        and args.mode == "flatquant_core"
        and not args.flat_load_checkpoint_dir
    ):
        raise ValueError(
            "Sharded FlatQuant evaluation requires --flat_load_checkpoint_dir."
        )
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in args.omni_lfq_slot_weights
    ) or sum(args.omni_lfq_slot_weights) <= 0.0:
        raise ValueError(
            "--omni_lfq_slot_weights must be finite, non-negative, and not all zero."
        )
    if args.omni_final_objective != "mse" and args.mode not in (
        "omniquant",
        "flatquant_core",
    ):
        raise ValueError(
            "Non-MSE --omni_final_objective values require OmniQuant or FlatQuant."
        )
    if args.omni_load_checkpoint_dir and args.omni_prefix_checkpoint_dir:
        raise ValueError(
            "--omni_load_checkpoint_dir and --omni_prefix_checkpoint_dir are mutually exclusive."
        )
    if (
        args.mode != "omniquant"
        and (args.omni_load_checkpoint_dir or args.omni_prefix_checkpoint_dir)
    ):
        raise ValueError("OmniQuant checkpoint options require --mode omniquant.")
    flat_checkpoint_options = (
        args.flat_load_checkpoint_dir,
        args.flat_prefix_checkpoint_dir,
        args.flat_finetune_checkpoint_dir,
    )
    if sum(value is not None for value in flat_checkpoint_options) > 1:
        raise ValueError(
            "FlatQuant load, prefix, and fine-tune checkpoint options are mutually exclusive."
        )
    if args.mode != "flatquant_core" and any(flat_checkpoint_options):
        raise ValueError("FlatQuant checkpoint options require --mode flatquant_core.")
    if (
        args.mode == "flatquant_core"
        and args.omni_final_objective == "lfq_ce"
        and args.flat_load_checkpoint_dir
    ):
        raise ValueError("FlatQuant LFQ is a training objective, not a load-only mode.")
    if args.mode == "flatquant_core":
        build_flatquant_config(args).validate()
    if not math.isfinite(args.omni_lfq_loss_weight) or args.omni_lfq_loss_weight < 0.0:
        raise ValueError("--omni_lfq_loss_weight must be finite and non-negative.")
    if (
        not math.isfinite(args.omni_lfq_boundary_loss_weight)
        or args.omni_lfq_boundary_loss_weight < 0.0
    ):
        raise ValueError("--omni_lfq_boundary_loss_weight must be finite and non-negative.")
    if (
        args.omni_final_objective == "lfq_ce"
        and args.omni_lfq_loss_weight == 0.0
        and args.omni_lfq_boundary_loss_weight == 0.0
    ):
        raise ValueError(
            "LFQ final-block training requires a positive --omni_lfq_loss_weight "
            "or --omni_lfq_boundary_loss_weight."
        )
    if args.calibration_only and args.evaluate:
        raise ValueError("--calibration_only and --evaluate are mutually exclusive.")
    set_seed(args.seed)

    model_name = Path(args.model_path.rstrip("/")).name
    run_output_dir = eval_run_output_dir(
        args.output_dir,
        num_shards=args.eval_num_shards,
        shard_id=args.eval_shard_id,
    )
    output_file = result_path(str(run_output_dir), model_name, args.task, args.split)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_hf_model(args)
    input_device = resolve_input_device(model, args.device)

    task_config = get_task_config(args.task)
    generation_config = task_config.get("generation_config", {})
    prompt_token = generation_config.get("prompt_token", "")
    classification_tokens = tuple(generation_config.get("target_tokens", ()))
    classification_token_ids = (
        resolve_classification_token_ids(tokenizer, classification_tokens)
        if classification_tokens
        else ()
    )
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
            weight_quant_scheme=args.weight_quant_scheme,
            weight_group_size=args.weight_group_size,
            act_quant_mode=args.act_quant_mode,
            weight_quant_format=args.weight_quant_format,
            activation_quant_format=args.activation_quant_format,
        )
    elif args.mode in {"smoothquant_w8a8", "gptq_fp8_w8a8", "omniquant", "flatquant_core"}:
        if args.mode == "omniquant":
            build_omniquant_config(args).validate()
        elif args.mode == "flatquant_core":
            build_flatquant_config(args).validate()
        if args.mode == "gptq_fp8_w8a8" and (
            args.weight_quant_format != "fp8_e4m3fn"
            or args.activation_quant_format != "fp8_e4m3fn"
        ):
            raise ValueError(
                "gptq_fp8_w8a8 implements only FP8 E4M3 weight/activation QDQ."
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
        slot_token_ids = (
            {
                slot: sid_slot_token_ids(tokenizer, slot)
                for slot in SID_SLOT_NAMES
            }
            if args.mode in ("omniquant", "flatquant_core")
            and args.omni_final_objective == "lfq_ce"
            else None
        )
        train_lfq = (
            args.omni_final_objective == "lfq_ce"
            and (
                (args.mode == "omniquant" and not args.omni_load_checkpoint_dir)
                or (
                    args.mode == "flatquant_core"
                    and not args.flat_load_checkpoint_dir
                )
            )
        )
        if train_lfq:
            calib_batches = build_lfq_sid_slot_batches(
                tokenizer=tokenizer,
                samples=list(calib_data.values()),
                prompt_token=prompt_token,
                device=input_device,
            )
        else:
            calib_prompts = [
                format_prompt(sample["prompt"], prompt_token)
                for sample in calib_data.values()
            ]
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
                weight_quant_format=args.weight_quant_format,
                weight_quant_scheme=args.weight_quant_scheme,
                weight_group_size=args.weight_group_size,
                activation_quant_format=args.activation_quant_format,
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
        elif args.mode == "flatquant_core" and args.flat_load_checkpoint_dir:
            baseline_summaries = restore_flatquant_core_layers_from_checkpoints(
                model=model,
                layer_indices=layer_indices,
                config=build_flatquant_config(args),
                checkpoint_dir=resolve_repo_path(args.flat_load_checkpoint_dir),
                act_quant_mode=args.act_quant_mode,
            )
        elif args.mode == "flatquant_core":
            baseline_summaries = apply_flatquant_core_layers(
                model=model,
                model_batches=calib_batches,
                layer_indices=layer_indices,
                config=build_flatquant_config(args),
                capture_layer_input_batches=capture_layer_input_batches,
                act_quant_mode=args.act_quant_mode,
                checkpoint_dir=output_file.parent / "flatquant_calibration",
                prefix_checkpoint_dir=(
                    resolve_repo_path(args.flat_prefix_checkpoint_dir)
                    if args.flat_prefix_checkpoint_dir
                    else None
                ),
                finetune_checkpoint_dir=(
                    resolve_repo_path(args.flat_finetune_checkpoint_dir)
                    if args.flat_finetune_checkpoint_dir
                    else None
                ),
                lfq_token_ids=(
                    slot_token_ids
                    if args.omni_final_objective == "lfq_ce"
                    else None
                ),
            )
        elif args.omni_load_checkpoint_dir:
            baseline_summaries = restore_omniquant_layers_from_checkpoints(
                model=model,
                layer_indices=layer_indices,
                config=build_omniquant_config(args),
                checkpoint_dir=resolve_repo_path(args.omni_load_checkpoint_dir),
                act_quant_mode=args.act_quant_mode,
            )
        else:
            baseline_summaries = apply_omniquant_layers(
                model=model,
                model_batches=calib_batches,
                layer_indices=layer_indices,
                config=build_omniquant_config(args),
                capture_layer_input_batches=capture_layer_input_batches,
                act_quant_mode=args.act_quant_mode,
                checkpoint_dir=output_file.parent / "omniquant_calibration",
                prefix_checkpoint_dir=(
                    resolve_repo_path(args.omni_prefix_checkpoint_dir)
                    if args.omni_prefix_checkpoint_dir
                    else None
                ),
                lfq_token_ids=(
                    slot_token_ids
                    if args.omni_final_objective == "lfq_ce"
                    else None
                ),
            )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    config = {
        "method": args.mode,
        "task": args.task,
        "layers": layer_indices,
        "fake_quant_forward_mode": FAKE_QUANT_FORWARD_MODE,
        "fake_quant_operator_dtype": FAKE_QUANT_OPERATOR_DTYPE,
        "fake_quant_qdq_compute_dtype": FAKE_QUANT_QDQ_COMPUTE_DTYPE,
        "fake_quant_loss_dtype": FAKE_QUANT_LOSS_DTYPE,
        "smoothquant_alpha": args.smoothquant_alpha,
        "smooth_scope": args.smooth_scope,
        "smooth_fold": args.smooth_fold,
        "smoothquant_min_scale": args.smoothquant_min_scale,
        "smoothquant_max_scale": args.smoothquant_max_scale,
        "gptq_damp_percent": args.gptq_damp_percent,
        "gptq_block_size": args.gptq_block_size,
        "omni_lwc": args.omni_lwc,
        "omni_symmetric_lwc_mode": args.omni_symmetric_lwc_mode,
        "weight_quant_scheme": args.weight_quant_scheme,
        "weight_group_size": (
            None if args.mode == "full_precision" else args.weight_group_size
        ),
        "weight_quant_granularity": (
            "none"
            if args.mode == "full_precision"
            else (
                "per_output_channel"
                if args.weight_group_size == 0
                else "per_output_channel_input_group"
            )
        ),
        "omni_weight_quant_scheme": args.weight_quant_scheme,
        "omni_let": args.omni_let,
        "omni_let_mode": args.omni_let_mode,
        "omni_let_init": args.omni_let_init,
        "omni_let_scale_parameterization": "unbounded_log",
        "omni_calibration_forward_mode": (
            OMNIQUANT_CALIBRATION_FORWARD_MODE
            if args.mode == "omniquant"
            else None
        ),
        "omni_calibration_compute_dtype": (
            OMNIQUANT_CALIBRATION_COMPUTE_DTYPE
            if args.mode == "omniquant"
            else None
        ),
        "omni_quantization_compute_dtype": (
            OMNIQUANT_QUANTIZATION_COMPUTE_DTYPE
            if args.mode == "omniquant"
            else None
        ),
        "omni_loss_compute_dtype": (
            OMNIQUANT_LOSS_COMPUTE_DTYPE
            if args.mode == "omniquant"
            else None
        ),
        "omni_final_objective": args.omni_final_objective,
        "omni_lfq_token_scope": args.omni_lfq_token_scope,
        "omni_lfq_vocab_scope": args.omni_lfq_vocab_scope,
        "omni_lfq_slot_weights": list(args.omni_lfq_slot_weights),
        "omni_lfq_loss_weight": args.omni_lfq_loss_weight,
        "omni_lfq_boundary_loss_weight": args.omni_lfq_boundary_loss_weight,
        "omni_lfq_boundary_topk": args.omni_lfq_boundary_topk,
        "omni_lfq_boundary_negative_count": (
            args.omni_lfq_boundary_negative_count
        ),
        "omni_lfq_boundary_tie_threshold": (
            args.omni_lfq_boundary_tie_threshold
        ),
        "omni_lfq_boundary_gap_scale": args.omni_lfq_boundary_gap_scale,
        "omni_epochs": args.omni_epochs,
        "omni_validation_sample_size": args.omni_validation_sample_size,
        "omni_train_sample_size": args.omni_train_sample_size,
        "omni_epoch_eval_interval": args.omni_epoch_eval_interval,
        "omni_lwc_lr": args.omni_lwc_lr,
        "omni_let_lr": args.omni_let_lr,
        "omni_weight_decay": args.omni_weight_decay,
        "omni_init_lwc_logit": args.omni_init_lwc_logit,
        "omni_max_grad_norm": args.omni_max_grad_norm,
        "omni_checkpoint_dir": (
            str(output_file.parent / "omniquant_calibration") if args.mode == "omniquant" else None
        ),
        "omni_load_checkpoint_dir": (
            str(resolve_repo_path(args.omni_load_checkpoint_dir)) if args.omni_load_checkpoint_dir else None
        ),
        "omni_prefix_checkpoint_dir": (
            str(resolve_repo_path(args.omni_prefix_checkpoint_dir)) if args.omni_prefix_checkpoint_dir else None
        ),
        "flatquant_variant": (
            (
                f"fixed_smoothquant_diagonal_lwc_lac_{args.omni_final_objective}"
                if args.flat_transform_kind == "smoothquant"
                else f"official_svd_cayley_diag_lwc_lac_{args.omni_final_objective}"
            )
            if args.mode == "flatquant_core"
            else None
        ),
        "flatquant_transform_sites": (
            ["qkv_shared_input", "o_input", "gate_up_shared_input", "down_input"]
            if args.mode == "flatquant_core"
            else None
        ),
        "flat_transform_init": args.flat_transform_init,
        "flat_transform_kind": args.flat_transform_kind,
        "flat_learn_transform": args.flat_learn_transform,
        "flat_lac": args.flat_lac,
        "flat_learn_lac": args.flat_learn_lac,
        "flat_diag_alpha": args.flat_diag_alpha,
        "flat_final_objective": args.omni_final_objective,
        "flat_lfq_token_scope": args.omni_lfq_token_scope,
        "flat_lfq_vocab_scope": args.omni_lfq_vocab_scope,
        "flat_lfq_slot_weights": list(args.omni_lfq_slot_weights),
        "flat_lfq_loss_weight": args.omni_lfq_loss_weight,
        "flat_lfq_boundary_loss_weight": args.omni_lfq_boundary_loss_weight,
        "flat_lfq_boundary_topk": args.omni_lfq_boundary_topk,
        "flat_lfq_boundary_negative_count": args.omni_lfq_boundary_negative_count,
        "flat_lfq_boundary_tie_threshold": args.omni_lfq_boundary_tie_threshold,
        "flat_lfq_boundary_gap_scale": args.omni_lfq_boundary_gap_scale,
        "flat_epochs": args.flat_epochs,
        "flat_train_sample_size": args.flat_train_sample_size,
        "flat_validation_sample_size": args.flat_validation_sample_size,
        "flat_epoch_eval_interval": args.flat_epoch_eval_interval,
        "flat_transform_lr": args.flat_transform_lr,
        "flat_lwc_lr": args.flat_lwc_lr,
        "flat_lac_lr": args.flat_lac_lr,
        "flat_weight_decay": args.flat_weight_decay,
        "flat_init_lwc_logit": args.flat_init_lwc_logit,
        "flat_init_lac_logit": args.flat_init_lac_logit,
        "flat_normalize_mse_gradient": args.flat_normalize_mse_gradient,
        "flat_max_grad_norm": args.flat_max_grad_norm,
        "flat_checkpoint_dir": (
            str(output_file.parent / "flatquant_calibration") if args.mode == "flatquant_core" else None
        ),
        "flat_load_checkpoint_dir": (
            str(resolve_repo_path(args.flat_load_checkpoint_dir)) if args.flat_load_checkpoint_dir else None
        ),
        "flat_prefix_checkpoint_dir": (
            str(resolve_repo_path(args.flat_prefix_checkpoint_dir))
            if args.flat_prefix_checkpoint_dir
            else None
        ),
        "flat_finetune_checkpoint_dir": (
            str(resolve_repo_path(args.flat_finetune_checkpoint_dir))
            if args.flat_finetune_checkpoint_dir
            else None
        ),
        "act_quant": args.act_quant,
        "act_quant_mode": args.act_quant_mode,
        "weight_quant_format": "none" if args.mode == "full_precision" else args.weight_quant_format,
        "activation_quant_format": "none" if args.mode == "full_precision" else args.activation_quant_format,
        "quantization_execution": (
            "full_precision"
            if args.mode == "full_precision"
            else (
                (
                    "online_fixed_smoothquant_then_fake_qdq_f_linear_in_model_dtype"
                    if args.flat_transform_kind == "smoothquant"
                    else "official_online_svd_cayley_transform_lac_then_fake_qdq_f_linear"
                )
                if args.mode == "flatquant_core"
                else "fake_qdq_then_f_linear_in_model_dtype"
            )
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
        "eval_num_shards": args.eval_num_shards,
        "eval_shard_id": args.eval_shard_id,
        "eval_shard_strategy": "round_robin",
        "eval_merged": False,
        "calibration_only": args.calibration_only,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "classification_target_tokens": list(classification_tokens),
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
        "flatquant_core": "flatquant_core_config.json",
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

    if args.calibration_only:
        print(
            "[calibration_only] completed; skipped generation and metrics "
            f"config={output_file.parent / config_filename}"
        )
        return

    unsharded_test_data = load_task_data(
        task_name=args.task,
        tokenizer=tokenizer,
        data_dir=str(resolve_repo_path(eval_data_dir)),
        split=args.split,
        sample_size=parse_sample_size(args.eval_sample_size),
        sample_offset=args.eval_offset,
    )
    unsharded_sample_count = len(unsharded_test_data)
    shard_description = (
        f" shard={args.eval_shard_id}/{args.eval_num_shards}"
        if args.eval_num_shards > 1
        else ""
    )
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
    config["eval_unsharded_sample_count"] = unsharded_sample_count
    config["eval_shard_sample_count"] = len(test_data)
    (output_file.parent / config_filename).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    test_items = list(test_data.items())
    generations: dict[str, list[str]] = {}
    sample_aux_metrics: dict[str, dict[str, Any]] = {}
    sid_tf_total_time = 0.0
    start = time.time()
    for sample_id, sample in tqdm(
        test_items,
        total=len(test_items),
        desc=f"{args.mode} {args.task} generation{shard_description}",
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
        if classification_tokens:
            generations[sample_id] = classify_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                input_device=input_device,
                target_tokens=classification_tokens,
                target_token_ids=classification_token_ids,
            )
        else:
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
