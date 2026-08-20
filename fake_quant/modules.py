from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import (
    ActQuant,
    FP8_MAX,
    QuantFormat,
    WeightQuantScheme,
    activation_per_token_qdq_by_format,
    quant_format_qmax,
    resolve_weight_quant_scheme,
    validate_quant_format,
    weight_per_output_channel_qdq_forward,
)


class BaselineFakeQuantLinear(nn.Module):
    """Inference-time min-max fake-QDQ Linear wrapper.

    Weights are QDQ'd once per output channel.  Activations are QDQ'd per
    token during ``forward``.  The following matrix multiplication remains
    ``F.linear`` in the model dtype, so this wrapper measures numeric quality
    rather than low-bit kernel latency or packed-weight memory use.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        act_quant: ActQuant = "none",
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        weight_quant_format: QuantFormat = "int8",
        weight_quant_scheme: WeightQuantScheme | None = None,
        activation_quant_format: QuantFormat | None = None,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")

        weight_format = validate_quant_format(weight_quant_format)
        weight_scheme = resolve_weight_quant_scheme(weight_format, weight_quant_scheme)
        if activation_quant_format is None:
            activation_format: QuantFormat = "int8" if act_quant == "per_token" else "none"
        else:
            activation_format = validate_quant_format(activation_quant_format)
        if act_quant == "none" and activation_format != "none":
            raise ValueError("activation_quant_format must be 'none' when act_quant='none'.")
        if act_quant == "per_token" and activation_format == "none":
            raise ValueError("activation_quant_format='none' requires act_quant='none'.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_quant = act_quant
        self.weight_quant_format = weight_format
        self.weight_quant_scheme = weight_scheme
        self.activation_quant_format = activation_format
        # Retained for legacy FP8 callers and archived decode-A16 support.
        self.qmax = (
            float(qmax)
            if activation_format == "fp8_e4m3fn"
            else (quant_format_qmax(activation_format) if activation_format != "none" else float(qmax))
        )
        self.eps = float(eps)

        with torch.no_grad():
            weight_qdq = weight_per_output_channel_qdq_forward(
                linear.weight.detach(),
                quant_format=self.weight_quant_format,
                quant_scheme=self.weight_quant_scheme,
                eps=self.eps,
                fp8_qmax=float(qmax),
            )
        self.register_buffer("weight_qdq", weight_qdq, persistent=True)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone(), persistent=True)
        else:
            self.register_buffer("bias", None, persistent=True)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            return activation_per_token_qdq_by_format(
                x,
                quant_format=self.activation_quant_format,
                eps=self.eps,
                fp8_qmax=self.qmax,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_prepared(self.quantize_activation(x))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_quant_format={self.weight_quant_format}, "
            f"weight_quant_scheme={self.weight_quant_scheme}, "
            f"activation_quant_format={self.activation_quant_format}, "
            f"act_quant={self.act_quant}"
        )


class OmniQuantFakeQuantLinear(BaselineFakeQuantLinear):
    """Inference wrapper holding a static OmniQuant weight-QDQ tensor.

    Unlike :class:`BaselineFakeQuantLinear`, this class must not recompute its
    weight quantization from absmax: ``weight_qdq`` already contains the
    learned LWC result after LET has been folded into the weight. It subclasses
    the baseline wrapper so shared input activation QDQ can reuse it.
    """

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        act_quant: ActQuant = "none",
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        weight_quant_format: QuantFormat = "int4",
        activation_quant_format: QuantFormat = "none",
    ) -> None:
        nn.Module.__init__(self)
        if weight_qdq.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight_qdq.shape)}")
        weight_format = validate_quant_format(weight_quant_format)
        activation_format = validate_quant_format(activation_quant_format)
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        if act_quant == "none" and activation_format != "none":
            raise ValueError("activation_quant_format must be 'none' when act_quant='none'.")
        if act_quant == "per_token" and activation_format == "none":
            raise ValueError("activation_quant_format='none' requires act_quant='none'.")

        self.in_features = int(weight_qdq.shape[1])
        self.out_features = int(weight_qdq.shape[0])
        self.act_quant = act_quant
        self.weight_quant_format = weight_format
        self.activation_quant_format = activation_format
        self.qmax = (
            float(qmax)
            if activation_format == "fp8_e4m3fn"
            else (quant_format_qmax(activation_format) if activation_format != "none" else float(qmax))
        )
        self.eps = float(eps)
        self.register_buffer("weight_qdq", weight_qdq.detach().clone(), persistent=True)
        self.register_buffer("bias", None if bias is None else bias.detach().clone(), persistent=True)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_quant_format={self.weight_quant_format}, "
            f"activation_quant_format={self.activation_quant_format}, "
            f"act_quant={self.act_quant}"
        )


class SmoothQuantFakeQuantLinear(nn.Module):
    """Fixed SmoothQuant fake-QDQ Linear wrapper.

    If ``input_scale`` is present, forward uses x / scale before activation
    quantization. If SmoothQuant folding has already moved the scale into the
    previous module, ``input_scale`` is None and the wrapper behaves like a
    normal fake-QDQ wrapper using the already-smoothed weight.
    """

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        act_quant: ActQuant = "none",
        input_scale: torch.Tensor | None = None,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        weight_quant_format: QuantFormat = "fp8_e4m3fn",
        activation_quant_format: QuantFormat | None = None,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        weight_format = validate_quant_format(weight_quant_format)
        activation_format = validate_quant_format(
            activation_quant_format
            if activation_quant_format is not None
            else ("fp8_e4m3fn" if act_quant == "per_token" else "none")
        )
        if act_quant == "none" and activation_format != "none":
            raise ValueError("activation_quant_format must be 'none' when act_quant='none'.")
        if act_quant == "per_token" and activation_format == "none":
            raise ValueError("activation_quant_format='none' requires act_quant='none'.")
        if weight_qdq.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight_qdq.shape)}")

        self.in_features = int(weight_qdq.shape[1])
        self.out_features = int(weight_qdq.shape[0])
        self.act_quant = act_quant
        self.weight_quant_format = weight_format
        self.activation_quant_format = activation_format
        self.qmax = (
            float(qmax)
            if activation_format == "fp8_e4m3fn"
            else (
                quant_format_qmax(activation_format)
                if activation_format != "none"
                else float(qmax)
            )
        )
        self.eps = float(eps)
        self.register_buffer("weight_qdq", weight_qdq.detach().clone(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)
        if input_scale is None:
            self.register_buffer("input_scale", None, persistent=True)
        else:
            scale = input_scale.detach().float().reshape(-1)
            if scale.numel() != self.in_features:
                raise ValueError(
                    f"Expected input_scale shape ({self.in_features},), got {tuple(input_scale.shape)}"
                )
            self.register_buffer("input_scale", scale, persistent=True)

    def smooth_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is None:
            return x
        view_shape = (1,) * (x.ndim - 1) + (-1,)
        return x / self.input_scale.to(device=x.device, dtype=x.dtype).view(view_shape)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smooth_activation(x)
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_by_format(
                x,
                quant_format=self.activation_quant_format,
                eps=self.eps,
                fp8_qmax=self.qmax,
            )
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_quant_format={self.weight_quant_format}, "
            f"activation_quant_format={self.activation_quant_format}, "
            f"act_quant={self.act_quant}, folded={self.input_scale is None}, qmax={self.qmax}"
        )


class GPTQFakeQuantLinear(nn.Module):
    """Inference-time GPTQ-calibrated FP8 weight + optional FP8 activation wrapper."""

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        act_quant: ActQuant = "none",
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        if weight_qdq.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight_qdq.shape)}")

        self.in_features = int(weight_qdq.shape[1])
        self.out_features = int(weight_qdq.shape[0])
        self.act_quant = act_quant
        self.weight_quant_format: QuantFormat = "fp8_e4m3fn"
        self.activation_quant_format: QuantFormat = "fp8_e4m3fn" if act_quant == "per_token" else "none"
        self.qmax = float(qmax)
        self.eps = float(eps)
        self.register_buffer("weight_qdq", weight_qdq.detach().clone(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_by_format(
                x,
                quant_format=self.activation_quant_format,
                eps=self.eps,
                fp8_qmax=self.qmax,
            )
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_quant={self.act_quant}, qmax={self.qmax}"
        )
