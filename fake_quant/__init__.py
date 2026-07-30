"""Composable fake-QDQ PTQ utilities for OneRec experiments."""

from .apply import (
    BaselineQuantSummary,
    apply_baseline_qdq,
    apply_baseline_w8a8,
    install_shared_input_activation_quantization,
    iter_baseline_w8a8_modules,
    iter_baseline_qdq_modules,
    iter_gptq_w8a8_modules,
    iter_smoothquant_w8a8_modules,
)
from .gptq import collect_gptq_hessians, gptaq_fp8_quantize_weight, gptq_fp8_quantize_weight, gptq_quantized_module_from_hessians
from .modules import BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear
from .quant import (
    QUANT_FORMAT_CHOICES,
    QuantFormat,
    activation_per_token_qdq_by_format,
    fp8_e4m3_qdq_forward,
    weight_per_output_channel_qdq_forward,
)
from .support.smoothquant_runtime import (
    collect_smoothquant_scales,
    fold_smoothquant_scales_inplace,
    smoothquant_quantized_module_from_scales,
)

__all__ = [
    "BaselineFakeQuantLinear",
    "BaselineQuantSummary",
    "GPTQFakeQuantLinear",
    "SmoothQuantFakeQuantLinear",
    "QUANT_FORMAT_CHOICES",
    "QuantFormat",
    "activation_per_token_qdq_by_format",
    "apply_baseline_qdq",
    "apply_baseline_w8a8",
    "collect_gptq_hessians",
    "collect_smoothquant_scales",
    "fold_smoothquant_scales_inplace",
    "fp8_e4m3_qdq_forward",
    "gptaq_fp8_quantize_weight",
    "gptq_fp8_quantize_weight",
    "gptq_quantized_module_from_hessians",
    "install_shared_input_activation_quantization",
    "iter_baseline_w8a8_modules",
    "iter_baseline_qdq_modules",
    "iter_gptq_w8a8_modules",
    "iter_smoothquant_w8a8_modules",
    "smoothquant_quantized_module_from_scales",
    "weight_per_output_channel_qdq_forward",
]
