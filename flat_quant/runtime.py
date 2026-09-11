"""Shared HuggingFace loader for composable fake-QDQ evaluations.

This module intentionally does not provide a low-bit GEMM.  It replaces
eligible ``nn.Linear`` modules with QDQ wrappers and then evaluates the model
through ordinary model-dtype ``F.linear`` calls.  It is therefore appropriate
for quality comparisons across mixed FP4/FP8 and INT4/INT6/INT8 formats.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .apply import apply_baseline_qdq
from .quant import (
    ActQuantMode,
    QuantFormat,
    WeightQuantScheme,
    resolve_weight_quant_scheme,
    validate_quant_format,
)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported model dtype {name!r}; expected 'bfloat16' or 'float16'.")


def load_fake_qdq_causal_lm(
    model_path: str,
    *,
    device: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    weight_quant_format: QuantFormat = "int8",
    weight_quant_scheme: WeightQuantScheme | None = None,
    weight_group_size: int | None = None,
    activation_quant_format: QuantFormat = "int8",
    activation_quant_mode: ActQuantMode = "shared_input",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load an HF causal LM and apply the baseline composable fake-QDQ path."""
    weight_format = validate_quant_format(weight_quant_format)
    weight_scheme = resolve_weight_quant_scheme(weight_format, weight_quant_scheme)
    activation_format = validate_quant_format(activation_quant_format)
    effective_act_mode: ActQuantMode = (
        "per_linear" if activation_format == "none" else activation_quant_mode
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype_from_name(dtype),
        trust_remote_code=trust_remote_code,
    ).to(device).eval()
    summary = apply_baseline_qdq(
        model,
        weight_quant_format=weight_format,
        weight_quant_scheme=weight_scheme,
        weight_group_size=weight_group_size,
        activation_quant_format=activation_format,
        act_quant_mode=effective_act_mode,
    )
    return model, tokenizer, {
        "mode": "fake_qdq",
        "weight_quant_format": weight_format,
        "weight_quant_scheme": weight_scheme,
        "weight_group_size": weight_group_size,
        "activation_quant_format": activation_format,
        "activation_quant_mode": effective_act_mode,
        "execution": "fake_qdq_then_f_linear_in_model_dtype",
        "replaced_linears": summary.replaced_linears,
        "skipped_linears": summary.skipped_linears,
        "shared_attention_modules": summary.shared_attention_modules,
        "shared_mlp_modules": summary.shared_mlp_modules,
    }
