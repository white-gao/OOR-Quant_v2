from __future__ import annotations

from functools import lru_cache
from typing import Literal, cast

import torch


FP8_MAX = 448.0
FP4_E2M1_MAX = 6.0

# Repository-wide numerical contract for every fake-quant baseline.
FAKE_QUANT_FORWARD_MODE = "deployment_matched"
FAKE_QUANT_OPERATOR_DTYPE = "model_dtype"
FAKE_QUANT_QDQ_COMPUTE_DTYPE = "float32"
FAKE_QUANT_LOSS_DTYPE = "float32"

ActQuant = Literal["none", "per_token"]
ActQuantMode = Literal["per_linear", "shared_input"]
QuantFormat = Literal["none", "fp8_e4m3fn", "fp4_e2m1", "int8", "int6", "int4"]
WeightQuantScheme = Literal["symmetric", "asymmetric"]
QUANT_FORMAT_CHOICES: tuple[QuantFormat, ...] = (
    "none",
    "fp8_e4m3fn",
    "fp4_e2m1",
    "int8",
    "int6",
    "int4",
)
WEIGHT_QUANT_SCHEME_CHOICES: tuple[WeightQuantScheme, ...] = (
    "symmetric",
    "asymmetric",
)


def validate_quant_format(quant_format: str) -> QuantFormat:
    """Validate and normalize a fake-QDQ numeric format name."""
    if quant_format not in QUANT_FORMAT_CHOICES:
        raise ValueError(
            f"Unsupported quant_format {quant_format!r}; expected one of {QUANT_FORMAT_CHOICES}."
        )
    return cast(QuantFormat, quant_format)


def resolve_weight_quant_scheme(
    quant_format: QuantFormat,
    quant_scheme: WeightQuantScheme | None = None,
) -> WeightQuantScheme:
    """Resolve the repository-wide default weight quantization scheme.

    Integer weights default to affine asymmetric QDQ with a zero point.
    Floating-point formats are zero-centered and therefore require symmetric
    QDQ. The unquantized format uses symmetric only as a harmless metadata
    value.
    """
    normalized = validate_quant_format(quant_format)
    if quant_scheme is not None and quant_scheme not in WEIGHT_QUANT_SCHEME_CHOICES:
        raise ValueError(
            f"Unsupported weight quantization scheme {quant_scheme!r}; "
            f"expected one of {WEIGHT_QUANT_SCHEME_CHOICES}."
        )
    if normalized in ("fp8_e4m3fn", "fp4_e2m1"):
        if quant_scheme == "asymmetric":
            raise ValueError(
                f"{normalized} has no affine integer zero point; use symmetric weights."
            )
        return "symmetric"
    if normalized == "none":
        return "symmetric" if quant_scheme is None else quant_scheme
    return "asymmetric" if quant_scheme is None else quant_scheme


def quant_format_bits(quant_format: str) -> int | None:
    """Return the storage precision of a QDQ format, or ``None`` for BF16/FP16."""
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return None
    if normalized in ("fp4_e2m1", "int4"):
        return 4
    if normalized == "int6":
        return 6
    return 8


def quant_format_qmax(quant_format: str) -> float:
    """Return the symmetric absmax denominator used to construct QDQ scales."""
    normalized = validate_quant_format(quant_format)
    if normalized == "fp8_e4m3fn":
        return FP8_MAX
    if normalized == "fp4_e2m1":
        return FP4_E2M1_MAX
    if normalized == "int8":
        return 127.0
    if normalized == "int6":
        return 31.0
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


@lru_cache(maxsize=None)
def _fp4_e2m1_tables(device_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache positive FP4 values and nearest-neighbor boundaries per device."""
    device = torch.device(device_key)
    values = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=device,
    )
    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        dtype=torch.float32,
        device=device,
    )
    return values, midpoints


def fp4_e2m1_quantize(x: torch.Tensor) -> torch.Tensor:
    """Round floating values to the finite FP4 E2M1 codebook.

    Conversion saturates to ``[-6, 6]`` and uses round-to-nearest,
    ties-to-even. Positive and negative zero have the same dequantized value,
    so the returned tensor uses a single numerical zero.
    """
    if not x.is_floating_point():
        raise TypeError(f"FP4 E2M1 quantization expects floating input, got {x.dtype}.")
    orig_dtype = x.dtype
    x_float = x.float()
    magnitude = x_float.abs()
    values, midpoints = _fp4_e2m1_tables(str(x.device))
    indices = torch.bucketize(magnitude, midpoints, right=False)

    # bucketize selects the lower code at every exact midpoint. E2M1
    # roundTiesToEven instead selects the code whose least-significant stored
    # mantissa bit is zero. For the 0..7 positive encodings, the upper code is
    # even at midpoint indices 1, 3, and 5.
    choose_upper = (
        magnitude.eq(midpoints[1])
        | magnitude.eq(midpoints[3])
        | magnitude.eq(midpoints[5])
    )
    indices = indices + choose_upper.to(dtype=indices.dtype)
    quantized = values[indices]
    quantized = torch.where(x_float < 0.0, -quantized, quantized)
    return quantized.to(dtype=orig_dtype)


def fp4_e2m1_qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP4 E2M1 quantize/dequantize with an explicit broadcastable scale."""
    orig_dtype = x.dtype
    scale_float = torch.clamp(scale.float(), min=eps)
    normalized = (x.float() / scale_float).clamp(
        min=-FP4_E2M1_MAX,
        max=FP4_E2M1_MAX,
    )
    quantized = fp4_e2m1_quantize(normalized)
    return (quantized.float() * scale_float).to(orig_dtype)


def int_symmetric_qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    bits: Literal[4, 6, 8],
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

    ``fp8_qmax`` exists only for backwards-compatible FP8 experiments.  INT4,
    INT6, and INT8 always use their canonical symmetric signed ranges:
    ``[-7, 7]``, ``[-31, 31]``, and ``[-127, 127]``.
    """
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return x
    if scale is None:
        raise ValueError(f"quant_format={normalized!r} requires an explicit scale.")
    if normalized == "fp8_e4m3fn":
        return fp8_e4m3_qdq_forward(x, scale, qmax=fp8_qmax, eps=eps)
    if normalized == "fp4_e2m1":
        return fp4_e2m1_qdq_forward(x, scale, eps=eps)
    bits: Literal[4, 6, 8]
    if normalized == "int4":
        bits = 4
    elif normalized == "int6":
        bits = 6
    else:
        bits = 8
    return int_symmetric_qdq_forward(
        x,
        scale,
        bits=bits,
        eps=eps,
    )


def int_asymmetric_qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    bits: Literal[4, 6, 8],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Affine asymmetric integer QDQ with broadcastable scale/zero point."""
    qmax = float((1 << int(bits)) - 1)
    orig_dtype = x.dtype
    scale_float = torch.clamp(scale.float(), min=eps)
    zero_point_float = zero_point.float()
    q = (torch.round(x.float() / scale_float) + zero_point_float).clamp(
        min=0.0,
        max=qmax,
    )
    return ((q - zero_point_float) * scale_float).to(orig_dtype)


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
    quant_format: QuantFormat = "int8",
    quant_scheme: WeightQuantScheme | None = None,
    eps: float = 1e-12,
    fp8_qmax: float = FP8_MAX,
) -> torch.Tensor:
    """QDQ Linear weights per output channel for an arbitrary fake format.

    Integer formats default to affine asymmetric min-max quantization. Pass
    quant_scheme="symmetric" for a zero-centered control experiment.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return weight
    scheme = resolve_weight_quant_scheme(normalized, quant_scheme)
    if scheme == "asymmetric":
        weight_float = weight.detach().float()
        upper = weight_float.amax(dim=1, keepdim=True)
        lower = weight_float.amin(dim=1, keepdim=True)
        bits = quant_format_bits(normalized)
        assert bits in (4, 6, 8)
        qmax = float((1 << bits) - 1)
        scale = ((upper - lower) / qmax).clamp_min(eps)
        zero_point = -torch.round(lower / scale)
        return int_asymmetric_qdq_forward(
            weight,
            scale,
            zero_point,
            bits=cast(Literal[4, 6, 8], bits),
            eps=eps,
        )
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
    quant_format: QuantFormat = "int8",
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
