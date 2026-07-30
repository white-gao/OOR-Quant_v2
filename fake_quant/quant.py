from __future__ import annotations

from typing import Literal, cast

import torch


FP8_MAX = 448.0
ActQuant = Literal["none", "per_token"]
ActQuantMode = Literal["per_linear", "shared_input"]
QuantFormat = Literal["none", "fp8_e4m3fn", "int8", "int4"]
QUANT_FORMAT_CHOICES: tuple[QuantFormat, ...] = ("none", "fp8_e4m3fn", "int8", "int4")


def validate_quant_format(quant_format: str) -> QuantFormat:
    """Validate and normalize a fake-QDQ numeric format name."""
    if quant_format not in QUANT_FORMAT_CHOICES:
        raise ValueError(
            f"Unsupported quant_format {quant_format!r}; expected one of {QUANT_FORMAT_CHOICES}."
        )
    return cast(QuantFormat, quant_format)


def quant_format_bits(quant_format: str) -> int | None:
    """Return the storage precision of a QDQ format, or ``None`` for BF16/FP16."""
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return None
    if normalized == "int4":
        return 4
    return 8


def quant_format_qmax(quant_format: str) -> float:
    """Return the symmetric absmax denominator used to construct QDQ scales."""
    normalized = validate_quant_format(quant_format)
    if normalized == "fp8_e4m3fn":
        return FP8_MAX
    if normalized == "int8":
        return 127.0
    if normalized == "int4":
        return 7.0
    raise ValueError("The unquantized 'none' format has no quantization range.")


def require_fp8() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for FP8 fake quantization. "
            "Please use a PyTorch build with FP8 dtype support."
        )
    return torch.float8_e4m3fn


def fp8_e4m3_qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 E4M3 quantize/dequantize forward path using explicit scales."""
    fp8_dtype = require_fp8()
    orig_dtype = x.dtype
    scale_float = torch.clamp(scale.float(), min=eps)
    q = torch.clamp(x.float() / scale_float, min=-qmax, max=qmax)
    q = q.to(fp8_dtype)
    return (q.float() * scale_float).to(orig_dtype)


def int_symmetric_qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    bits: Literal[4, 8],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Symmetric uniform integer QDQ with an explicit broadcastable scale.

    The integer values are intentionally dequantized before the following
    ``F.linear``.  This is a controlled fake-quant quality path, not an INT4
    or INT8 GEMM implementation.
    """
    qmax = float((1 << (int(bits) - 1)) - 1)
    orig_dtype = x.dtype
    scale_float = torch.clamp(scale.float(), min=eps)
    q = torch.round(x.float() / scale_float).clamp(min=-qmax, max=qmax)
    return (q * scale_float).to(orig_dtype)


def qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    quant_format: QuantFormat,
    eps: float = 1e-12,
    fp8_qmax: float = FP8_MAX,
) -> torch.Tensor:
    """Apply a format-specific QDQ transform using a caller-provided scale.

    ``fp8_qmax`` exists only for backwards-compatible FP8 experiments.  INT4
    and INT8 always use their canonical symmetric signed ranges, respectively
    ``[-7, 7]`` and ``[-127, 127]``.
    """
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return x
    if scale is None:
        raise ValueError(f"quant_format={normalized!r} requires an explicit scale.")
    if normalized == "fp8_e4m3fn":
        return fp8_e4m3_qdq_forward(x, scale, qmax=fp8_qmax, eps=eps)
    return int_symmetric_qdq_forward(
        x,
        scale,
        bits=4 if normalized == "int4" else 8,
        eps=eps,
    )


def _absmax_scale(
    x: torch.Tensor,
    *,
    dim: int | tuple[int, ...],
    keepdim: bool,
    quant_format: QuantFormat,
    eps: float,
    fp8_qmax: float,
) -> torch.Tensor:
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        raise ValueError("The unquantized 'none' format does not use a QDQ scale.")
    qmax = fp8_qmax if normalized == "fp8_e4m3fn" else quant_format_qmax(normalized)
    absmax = x.detach().float().abs().amax(dim=dim, keepdim=keepdim)
    return torch.clamp(absmax / float(qmax), min=eps)


def weight_per_output_channel_qdq_forward(
    weight: torch.Tensor,
    *,
    quant_format: QuantFormat = "fp8_e4m3fn",
    eps: float = 1e-12,
    fp8_qmax: float = FP8_MAX,
) -> torch.Tensor:
    """QDQ Linear weights per output channel for an arbitrary fake format."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return weight
    scale = _absmax_scale(
        weight,
        dim=1,
        keepdim=True,
        quant_format=normalized,
        eps=eps,
        fp8_qmax=fp8_qmax,
    )
    return qdq_forward(
        weight,
        scale,
        quant_format=normalized,
        eps=eps,
        fp8_qmax=fp8_qmax,
    )


def activation_per_token_qdq_by_format(
    x: torch.Tensor,
    *,
    quant_format: QuantFormat = "fp8_e4m3fn",
    eps: float = 1e-12,
    fp8_qmax: float = FP8_MAX,
) -> torch.Tensor:
    """QDQ activations per token/row along their hidden dimension."""
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return x
    scale = _absmax_scale(
        x,
        dim=-1,
        keepdim=True,
        quant_format=normalized,
        eps=eps,
        fp8_qmax=fp8_qmax,
    )
    return qdq_forward(
        x,
        scale,
        quant_format=normalized,
        eps=eps,
        fp8_qmax=fp8_qmax,
    )


def fp8_weight_per_channel_forward(
    weight: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 fake quantize Linear weights per output channel."""
    return weight_per_output_channel_qdq_forward(
        weight,
        quant_format="fp8_e4m3fn",
        eps=eps,
        fp8_qmax=qmax,
    )


def activation_per_token_qdq_forward(
    x: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Backward-compatible FP8 activation QDQ helper.

    New callers should use :func:`activation_per_token_qdq_by_format` so the
    activation representation is explicit in experiment metadata.
    """
    return activation_per_token_qdq_by_format(
        x,
        quant_format="fp8_e4m3fn",
        eps=eps,
        fp8_qmax=qmax,
    )
