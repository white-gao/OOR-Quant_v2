from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_MAX = 448.0
ActivationQuantMode = Literal["dynamic", "static"]
_FP8_RECORD_FUNCTIONS = False
_VLLM_SCALED_FP8_QUANT = None


@dataclass(frozen=True)
class FP8PreparedInput:
    x_fp8: torch.Tensor
    scale: torch.Tensor
    leading_shape: tuple[int, ...]


def set_fp8_record_functions_enabled(enabled: bool) -> None:
    global _FP8_RECORD_FUNCTIONS
    _FP8_RECORD_FUNCTIONS = bool(enabled)


def _get_vllm_scaled_fp8_quant():
    global _VLLM_SCALED_FP8_QUANT
    if _VLLM_SCALED_FP8_QUANT is None:
        try:
            from vllm import _custom_ops as ops
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("vLLM custom ops are required for fused dynamic FP8 activation quantization.") from exc
        _VLLM_SCALED_FP8_QUANT = ops.scaled_fp8_quant
    return _VLLM_SCALED_FP8_QUANT


def _vllm_scaled_fp8_quant(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp8, scale = _get_vllm_scaled_fp8_quant()(
        x_2d,
        scale=None,
        use_per_token_if_dynamic=True,
    )
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    if scale.dtype != torch.float32:
        scale = scale.float()
    return x_fp8, scale


def require_fp8_runtime() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for real FP8 W8A8 inference.")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is required for real FP8 W8A8 inference.")
    return torch.float8_e4m3fn


def _safe_scale(absmax: torch.Tensor, *, qmax: float, eps: float) -> torch.Tensor:
    return torch.clamp(absmax.float() / float(qmax), min=float(eps))


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor, *, qmax: float) -> torch.Tensor:
    fp8_dtype = require_fp8_runtime()
    return torch.clamp(x.float() / scale.float(), -float(qmax), float(qmax)).to(fp8_dtype)


def weight_scale_per_output_channel(weight: torch.Tensor, *, qmax: float, eps: float) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    return _safe_scale(weight.detach().float().abs().amax(dim=1, keepdim=True), qmax=qmax, eps=eps)


def activation_scale_per_token(x_2d: torch.Tensor, *, qmax: float, eps: float) -> torch.Tensor:
    if x_2d.ndim != 2:
        raise ValueError(f"Expected 2D activation matrix, got shape {tuple(x_2d.shape)}")
    return _safe_scale(x_2d.detach().float().abs().amax(dim=1, keepdim=True), qmax=qmax, eps=eps)


def activation_qdq_like_runtime(
    x: torch.Tensor,
    *,
    activation_quant_mode: ActivationQuantMode,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
    static_scale: torch.Tensor | float | None = None,
    decode_a16_when_single_token: bool = False,
) -> torch.Tensor:
    """Return the activation values consumed by the runtime Linear path.

    The result is FP8-QDQ in the FP8 path and the original BF16/FP32 value in
    the decode-A16 path. It is intended for offline calibration statistics,
    not for the latency-critical inference path.
    """
    mode = _validate_activation_quant_mode(activation_quant_mode)
    if x.ndim < 1:
        raise ValueError(f"Expected activation with at least one dimension, got {tuple(x.shape)}")

    if decode_a16_when_single_token and x.ndim >= 3 and int(x.shape[-2]) == 1:
        return x.detach().float()

    x_2d = x.detach().float().reshape(-1, x.shape[-1]).contiguous()
    if mode == "dynamic":
        x_fp8, scale = _vllm_scaled_fp8_quant(x_2d)
    else:
        if static_scale is None:
            raise ValueError("static_activation_scales must provide every GPTAQ Linear runtime scale.")
        scale = torch.as_tensor(static_scale, device=x_2d.device, dtype=torch.float32).reshape(1, 1)
        scale = torch.clamp(scale, min=float(eps)).expand(x_2d.shape[0], 1).contiguous()
        x_fp8 = quantize_fp8(x_2d, scale, qmax=qmax)
    return (x_fp8.float() * scale.float()).reshape_as(x)


def _validate_activation_quant_mode(mode: str) -> ActivationQuantMode:
    if mode not in ("dynamic", "static"):
        raise ValueError(f"Unsupported activation_quant_mode {mode!r}; expected 'dynamic' or 'static'.")
    return mode  # type: ignore[return-value]


class RealFP8Linear(nn.Module):
    """Naive W8A8 FP8 Linear using torch._scaled_mm.

    Weight is quantized once per output channel. Activation is dynamically
    quantized per token/row by default. Static activation mode uses one
    calibration-derived scale per Linear and therefore removes the runtime
    per-token absmax/scale computation, while still quantizing activations to
    FP8 before ``torch._scaled_mm``.
    """

    def __init__(
        self,
        *,
        weight_fp8_t: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = False,
        activation_quant_mode: ActivationQuantMode = "dynamic",
        decode_a16_when_single_token: bool = False,
    ) -> None:
        super().__init__()
        require_fp8_runtime()
        if weight_fp8_t.ndim != 2:
            raise ValueError(f"Expected 2D transposed weight, got shape {tuple(weight_fp8_t.shape)}")
        if tuple(weight_fp8_t.shape) != (int(in_features), int(out_features)):
            raise ValueError(
                "weight_fp8_t shape must be "
                f"({int(in_features)}, {int(out_features)}), got {tuple(weight_fp8_t.shape)}"
            )
        if weight_scale.shape != (1, int(out_features)):
            raise ValueError(f"Expected weight_scale shape (1, {out_features}), got {tuple(weight_scale.shape)}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.qmax = float(qmax)
        self.eps = float(eps)
        self.output_dtype = output_dtype
        self.use_fast_accum = bool(use_fast_accum)
        self.activation_quant_mode: ActivationQuantMode = _validate_activation_quant_mode(activation_quant_mode)
        self.decode_a16_when_single_token = bool(decode_a16_when_single_token)
        self._observe_activation_scales = False

        self.register_buffer("weight_fp8_t", weight_fp8_t.detach(), persistent=True)
        self.register_buffer("weight_scale", weight_scale.detach().float(), persistent=True)
        weight_qdq = (weight_fp8_t.detach().float() * weight_scale.detach().float()).t().contiguous().to(output_dtype)
        self.register_buffer("weight_qdq", weight_qdq, persistent=True)
        self.register_buffer("static_activation_scale", torch.ones(1, 1, dtype=torch.float32), persistent=True)
        self.register_buffer("_observed_activation_scale", torch.zeros(1, 1, dtype=torch.float32), persistent=False)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        output_dtype: torch.dtype | None = None,
        use_fast_accum: bool = False,
        activation_quant_mode: ActivationQuantMode = "dynamic",
        decode_a16_when_single_token: bool = False,
    ) -> "RealFP8Linear":
        if linear.weight.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(linear.weight.shape)}")
        out_features, in_features = linear.weight.shape
        chosen_output_dtype = output_dtype
        if chosen_output_dtype is None:
            chosen_output_dtype = linear.weight.dtype
            if chosen_output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
                chosen_output_dtype = torch.bfloat16

        weight = linear.weight.detach()
        scale = weight_scale_per_output_channel(weight, qmax=qmax, eps=eps)
        weight_fp8 = quantize_fp8(weight, scale, qmax=qmax)
        return cls(
            weight_fp8_t=weight_fp8.t(),
            weight_scale=scale.t().contiguous(),
            bias=linear.bias,
            in_features=int(in_features),
            out_features=int(out_features),
            qmax=qmax,
            eps=eps,
            output_dtype=chosen_output_dtype,
            use_fast_accum=use_fast_accum,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
        )

    @classmethod
    def from_gptq_linear(
        cls,
        linear: nn.Linear,
        hessian: torch.Tensor,
        *,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        output_dtype: torch.dtype | None = None,
        use_fast_accum: bool = False,
        activation_quant_mode: ActivationQuantMode = "dynamic",
        decode_a16_when_single_token: bool = False,
        damp_percent: float = 0.01,
        block_size: int = 128,
    ) -> "RealFP8Linear":
        if linear.weight.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(linear.weight.shape)}")
        out_features, in_features = linear.weight.shape
        if hessian.shape != (int(in_features), int(in_features)):
            raise ValueError(
                f"Expected GPTQ Hessian shape ({int(in_features)}, {int(in_features)}), got {tuple(hessian.shape)}"
            )
        chosen_output_dtype = output_dtype
        if chosen_output_dtype is None:
            chosen_output_dtype = linear.weight.dtype
            if chosen_output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
                chosen_output_dtype = torch.bfloat16

        from fake_quant.gptq import gptq_fp8_quantize_weight

        weight = linear.weight.detach()
        weight_qdq = gptq_fp8_quantize_weight(
            weight,
            hessian,
            damp_percent=damp_percent,
            block_size=block_size,
            qmax=qmax,
            eps=eps,
        )
        scale = weight_scale_per_output_channel(weight, qmax=qmax, eps=eps)
        weight_fp8 = quantize_fp8(weight_qdq, scale, qmax=qmax)
        return cls(
            weight_fp8_t=weight_fp8.t(),
            weight_scale=scale.t().contiguous(),
            bias=linear.bias,
            in_features=int(in_features),
            out_features=int(out_features),
            qmax=qmax,
            eps=eps,
            output_dtype=chosen_output_dtype,
            use_fast_accum=use_fast_accum,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
        )

    @classmethod
    def from_gptaq_linear(
        cls,
        linear: nn.Linear,
        hessian_q: torch.Tensor,
        dxx_t: torch.Tensor,
        *,
        alpha: float = 1.0,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        output_dtype: torch.dtype | None = None,
        use_fast_accum: bool = False,
        activation_quant_mode: ActivationQuantMode = "dynamic",
        decode_a16_when_single_token: bool = False,
        damp_percent: float = 0.01,
        block_size: int = 128,
    ) -> "RealFP8Linear":
        if linear.weight.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(linear.weight.shape)}")
        out_features, in_features = linear.weight.shape
        expected_shape = (int(in_features), int(in_features))
        if hessian_q.shape != expected_shape:
            raise ValueError(f"Expected GPTAQ Hessian shape {expected_shape}, got {tuple(hessian_q.shape)}")
        if dxx_t.shape != expected_shape:
            raise ValueError(f"Expected GPTAQ dxx_t shape {expected_shape}, got {tuple(dxx_t.shape)}")
        chosen_output_dtype = output_dtype
        if chosen_output_dtype is None:
            chosen_output_dtype = linear.weight.dtype
            if chosen_output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
                chosen_output_dtype = torch.bfloat16

        from fake_quant.gptq import gptaq_fp8_quantize_weight

        weight = linear.weight.detach()
        weight_qdq = gptaq_fp8_quantize_weight(
            weight,
            hessian_q,
            dxx_t,
            alpha=alpha,
            damp_percent=damp_percent,
            block_size=block_size,
            qmax=qmax,
            eps=eps,
        )
        scale = weight_scale_per_output_channel(weight, qmax=qmax, eps=eps)
        weight_fp8 = quantize_fp8(weight_qdq, scale, qmax=qmax)
        return cls(
            weight_fp8_t=weight_fp8.t(),
            weight_scale=scale.t().contiguous(),
            bias=linear.bias,
            in_features=int(in_features),
            out_features=int(out_features),
            qmax=qmax,
            eps=eps,
            output_dtype=chosen_output_dtype,
            use_fast_accum=use_fast_accum,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
        )

    def set_activation_quant_mode(self, mode: str) -> None:
        self.activation_quant_mode = _validate_activation_quant_mode(mode)

    def set_static_activation_scale(self, scale: torch.Tensor | float) -> None:
        scale_tensor = torch.as_tensor(
            scale,
            device=self.static_activation_scale.device,
            dtype=torch.float32,
        ).reshape(1, 1)
        self.static_activation_scale.copy_(torch.clamp(scale_tensor, min=self.eps))

    def reset_activation_scale_observer(self) -> None:
        self._observed_activation_scale.zero_()

    def enable_activation_scale_observer(self, enabled: bool = True) -> None:
        self._observe_activation_scales = bool(enabled)

    def observed_activation_scale(self) -> float:
        return float(self._observed_activation_scale.detach().float().max().item())

    def freeze_static_activation_scale_from_observer(self) -> float:
        observed = self.observed_activation_scale()
        if observed <= 0.0:
            observed = float(self.eps)
        self.set_static_activation_scale(observed)
        return observed

    def _flatten_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim {self.in_features}, got {tuple(x.shape)}")
        return x.reshape(-1, self.in_features).contiguous(), tuple(x.shape[:-1])

    def _dynamic_activation_scale(self, x_2d: torch.Tensor) -> torch.Tensor:
        if _FP8_RECORD_FUNCTIONS:
            with torch.profiler.record_function("real_fp8/activation_dynamic_scale"):
                return activation_scale_per_token(x_2d, qmax=self.qmax, eps=self.eps)
        return activation_scale_per_token(x_2d, qmax=self.qmax, eps=self.eps)

    def _dynamic_activation_quantize(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if _FP8_RECORD_FUNCTIONS:
            with torch.profiler.record_function("real_fp8/activation_dynamic_fused_quantize"):
                return _vllm_scaled_fp8_quant(x_2d)
        return _vllm_scaled_fp8_quant(x_2d)

    def _static_activation_scale(self, rows: int, device: torch.device) -> torch.Tensor:
        def build_scale() -> torch.Tensor:
            return self.static_activation_scale.to(device=device, dtype=torch.float32).expand(rows, 1).contiguous()

        if _FP8_RECORD_FUNCTIONS:
            with torch.profiler.record_function("real_fp8/activation_static_scale"):
                return build_scale()
        return build_scale()

    def _quantize_activation(self, x_2d: torch.Tensor, act_scale: torch.Tensor) -> torch.Tensor:
        if _FP8_RECORD_FUNCTIONS:
            with torch.profiler.record_function("real_fp8/activation_quantize"):
                return quantize_fp8(x_2d, act_scale, qmax=self.qmax)
        return quantize_fp8(x_2d, act_scale, qmax=self.qmax)

    def prepare_input(self, x: torch.Tensor) -> FP8PreparedInput:
        x_2d, leading_shape = self._flatten_input(x)
        if self.activation_quant_mode == "static":
            act_scale = self._static_activation_scale(x_2d.shape[0], x_2d.device)
            x_fp8 = self._quantize_activation(x_2d, act_scale)
        else:
            x_fp8, act_scale = self._dynamic_activation_quantize(x_2d)

        if self._observe_activation_scales:
            observed = act_scale.detach().float().amax().reshape(1, 1).to(self._observed_activation_scale.device)
            self._observed_activation_scale.copy_(torch.maximum(self._observed_activation_scale, observed))
        return FP8PreparedInput(x_fp8=x_fp8, scale=act_scale, leading_shape=leading_shape)

    def _bias_for_output(self, *, device: torch.device) -> torch.Tensor | None:
        if self.bias is None:
            return None
        if self.bias.device == device and self.bias.dtype == self.output_dtype:
            return self.bias
        return self.bias.to(device=device, dtype=self.output_dtype)

    def _input_for_output_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == self.output_dtype:
            return x
        return x.to(self.output_dtype)

    def forward_prepared(self, prepared: FP8PreparedInput) -> torch.Tensor:
        if prepared.x_fp8.ndim != 2 or prepared.x_fp8.shape[1] != self.in_features:
            raise ValueError(
                f"Prepared input must have shape [M, {self.in_features}], got {tuple(prepared.x_fp8.shape)}"
            )
        if prepared.x_fp8.device.type == "cuda":
            if _FP8_RECORD_FUNCTIONS:
                with torch.profiler.record_function("real_fp8/scaled_mm"):
                    y_2d = torch._scaled_mm(
                        prepared.x_fp8,
                        self.weight_fp8_t,
                        scale_a=prepared.scale,
                        scale_b=self.weight_scale,
                        out_dtype=self.output_dtype,
                        use_fast_accum=self.use_fast_accum,
                    )
            else:
                y_2d = torch._scaled_mm(
                    prepared.x_fp8,
                    self.weight_fp8_t,
                    scale_a=prepared.scale,
                    scale_b=self.weight_scale,
                    out_dtype=self.output_dtype,
                    use_fast_accum=self.use_fast_accum,
                )
        else:
            if _FP8_RECORD_FUNCTIONS:
                with torch.profiler.record_function("real_fp8/cpu_reference_mm"):
                    x_qdq = prepared.x_fp8.float() * prepared.scale.float()
                    weight_qdq_t = self.weight_fp8_t.float() * self.weight_scale.float()
                    y_2d = (x_qdq @ weight_qdq_t).to(self.output_dtype)
            else:
                x_qdq = prepared.x_fp8.float() * prepared.scale.float()
                weight_qdq_t = self.weight_fp8_t.float() * self.weight_scale.float()
                y_2d = (x_qdq @ weight_qdq_t).to(self.output_dtype)
        if _FP8_RECORD_FUNCTIONS:
            with torch.profiler.record_function("real_fp8/bias_reshape"):
                bias = self._bias_for_output(device=y_2d.device)
                if bias is not None:
                    y_2d = y_2d + bias
                return y_2d.reshape(*prepared.leading_shape, self.out_features)
        bias = self._bias_for_output(device=y_2d.device)
        if bias is not None:
            y_2d = y_2d + bias
        return y_2d.reshape(*prepared.leading_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.should_use_decode_a16(x):
            return self.forward_w8a16(x)
        return self.forward_prepared(self.prepare_input(x))

    def should_use_decode_a16(self, x: torch.Tensor) -> bool:
        return bool(self.decode_a16_when_single_token and x.ndim >= 3 and int(x.shape[-2]) == 1)

    def forward_w8a16(self, x: torch.Tensor) -> torch.Tensor:
        x_2d, leading_shape = self._flatten_input(x)
        if _FP8_RECORD_FUNCTIONS:
            with torch.profiler.record_function("real_fp8/decode_w8a16_linear"):
                return self._forward_w8a16_flattened(x_2d, leading_shape)
        return self._forward_w8a16_flattened(x_2d, leading_shape)

    def _forward_w8a16_flattened(self, x_2d: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        bias = self._bias_for_output(device=x_2d.device)
        y_2d = F.linear(self._input_for_output_dtype(x_2d), self.weight_qdq, bias)
        return y_2d.reshape(*leading_shape, self.out_features)

    def reference_w8a16_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_w8a16(x)

    def reference_qdq_forward(self, x: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare_input(x)
        x_qdq = prepared.x_fp8.float() * prepared.scale.float()
        weight_qdq_t = self.weight_fp8_t.float() * self.weight_scale.float()
        y_2d = x_qdq @ weight_qdq_t
        if self.bias is not None:
            y_2d = y_2d + self.bias.to(device=y_2d.device, dtype=y_2d.dtype)
        return y_2d.to(self.output_dtype).reshape(*prepared.leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_scale=per_output_channel, act_scale={self.activation_quant_mode}, "
            f"output_dtype={self.output_dtype}, use_fast_accum={self.use_fast_accum}, "
            f"decode_a16_when_single_token={self.decode_a16_when_single_token}"
        )

    def _apply(self, fn: Any, recurse: bool = True) -> "RealFP8Linear":
        super()._apply(fn, recurse=recurse)
        return self
