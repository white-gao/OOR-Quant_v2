"""Blockwise OmniQuant-style calibration for Qwen3 decoder blocks.

The weight path supports symmetric signed integer/finite-float LWC and
paper-style asymmetric integer LWC with
independently learned upper/lower clipping factors and a zero point.  All
variants operate per output channel.
Scale-only LET is supported at the Qwen3 locations that can be folded exactly
without adding inference work:

* input RMSNorm -> Q/K/V,
* post-attention RMSNorm -> gate/up,
* V -> O, including Qwen3 grouped-query attention.

The original FP parameters remain frozen.  Calibration optimizes a copied
block against its FP counterpart, then folds LET and writes static QDQ linear
wrappers back into the model.

"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..apply import BaselineQuantSummary, install_shared_input_activation_quantization
from ..modules import OmniQuantFakeQuantLinear
from ..quant import (
    ActQuant,
    FAKE_QUANT_FORWARD_MODE,
    FAKE_QUANT_LOSS_DTYPE,
    FAKE_QUANT_OPERATOR_DTYPE,
    FAKE_QUANT_QDQ_COMPUTE_DTYPE,
    FP4_E2M1_MAX,
    FP8_MAX,
    QuantFormat,
    fp4_e2m1_quantize,
    quant_format_qmax,
    require_fp8,
    validate_quant_format,
)
from ..support.runtime_utils import _module_device, _move_tree_to_device
from ..support.smoothquant_runtime import (
    Batch,
    DEFAULT_SMOOTHQUANT_ALPHA,
    _batch_to_args_kwargs,
    collect_smoothquant_scales,
)


LFQ_SLOT_NAMES = ("a", "b", "c")
OMNIQUANT_CALIBRATION_FORWARD_MODE = FAKE_QUANT_FORWARD_MODE
OMNIQUANT_CALIBRATION_COMPUTE_DTYPE = FAKE_QUANT_OPERATOR_DTYPE
OMNIQUANT_QUANTIZATION_COMPUTE_DTYPE = FAKE_QUANT_QDQ_COMPUTE_DTYPE
OMNIQUANT_LOSS_COMPUTE_DTYPE = FAKE_QUANT_LOSS_DTYPE
DEFAULT_OMNIQUANT_EPOCHS = 20
DEFAULT_OMNIQUANT_LWC_LR = 1e-2
DEFAULT_OMNIQUANT_LET_LR = 5e-3
DEFAULT_OMNIQUANT_WEIGHT_DECAY = 0.0
DEFAULT_OMNIQUANT_INIT_LWC_LOGIT = 4.0
DEFAULT_OMNIQUANT_MAX_GRAD_NORM: float | None = None

OmniQuantObjective = Literal["mse", "lfq_ce"]

@dataclass(frozen=True)
class OmniQuantConfig:
    """Calibration parameters for LWC and scale-only LET."""

    weight_quant_format: QuantFormat = "int8"
    activation_quant_format: QuantFormat = "int8"
    weight_quant_scheme: Literal["symmetric", "asymmetric"] = "asymmetric"
    use_lwc: bool = True
    use_let: bool = True
    learn_let: bool = True
    let_init: Literal["smoothquant", "ones"] = "smoothquant"
    smoothquant_alpha: float = DEFAULT_SMOOTHQUANT_ALPHA
    final_objective: OmniQuantObjective = "mse"
    lfq_token_scope: Literal["sid_slots"] = "sid_slots"
    lfq_vocab_scope: Literal["s_abc"] = "s_abc"
    lfq_slot_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lfq_loss_weight: float = 1.0
    epochs: int = DEFAULT_OMNIQUANT_EPOCHS
    # Hold out the last N calibration samples for LFQ validation only. These
    # samples are forwarded every epoch but never participate in backprop.
    validation_sample_size: int = 0
    # Optionally use only the first N samples before the held-out tail for
    # training. Zero uses every non-validation sample.
    train_sample_size: int = 0
    # Evaluate the fixed post-update parameters on the full calibration set
    # every N epochs. Zero preserves the legacy final-only evaluation path.
    epoch_eval_interval: int = 0
    lwc_lr: float = DEFAULT_OMNIQUANT_LWC_LR
    # Log-space LET keeps scales positive without imposing hard bounds.
    let_lr: float = DEFAULT_OMNIQUANT_LET_LR
    weight_decay: float = DEFAULT_OMNIQUANT_WEIGHT_DECAY
    init_lwc_logit: float = DEFAULT_OMNIQUANT_INIT_LWC_LOGIT
    # The official optimizer reports gradient norms without clipping.
    max_grad_norm: float | None = DEFAULT_OMNIQUANT_MAX_GRAD_NORM
    eps: float = 1e-12

    def validate(self) -> None:
        weight = validate_quant_format(self.weight_quant_format)
        validate_quant_format(self.activation_quant_format)
        if weight not in ("fp8_e4m3fn", "fp4_e2m1", "int4", "int6", "int8"):
            raise ValueError(
                "OmniQuant requires FP8 E4M3FN, FP4 E2M1, INT4, INT6, or INT8 weights."
            )
        if self.weight_quant_scheme not in ("symmetric", "asymmetric"):
            raise ValueError("weight_quant_scheme must be 'symmetric' or 'asymmetric'.")
        if weight in ("fp8_e4m3fn", "fp4_e2m1") and self.weight_quant_scheme != "symmetric":
            raise ValueError(
                f"{weight} OmniQuant requires weight_quant_scheme='symmetric'; "
                "floating-point codebooks have no integer zero point for asymmetric LWC."
            )
        if self.let_init not in ("smoothquant", "ones"):
            raise ValueError("let_init must be 'smoothquant' or 'ones'.")
        if (
            not math.isfinite(self.smoothquant_alpha)
            or not 0.0 <= self.smoothquant_alpha <= 1.0
        ):
            raise ValueError("smoothquant_alpha must be finite and in [0, 1].")
        if self.final_objective not in ("mse", "lfq_ce"):
            raise ValueError("final_objective must be 'mse' or 'lfq_ce'.")
        if (self.lfq_token_scope, self.lfq_vocab_scope) != ("sid_slots", "s_abc"):
            raise ValueError(
                "LFQ requires lfq_token_scope='sid_slots' and "
                "lfq_vocab_scope='s_abc'."
            )
        if len(self.lfq_slot_weights) != len(LFQ_SLOT_NAMES):
            raise ValueError("lfq_slot_weights must contain weights for SID_a, SID_b, and SID_c.")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in self.lfq_slot_weights):
            raise ValueError("lfq_slot_weights must be finite and non-negative.")
        if sum(self.lfq_slot_weights) <= 0.0:
            raise ValueError("At least one LFQ slot weight must be positive.")
        if not math.isfinite(self.lfq_loss_weight) or self.lfq_loss_weight < 0.0:
            raise ValueError("lfq_loss_weight must be finite and non-negative.")
        if self.final_objective == "lfq_ce" and self.lfq_loss_weight == 0.0:
            raise ValueError("LFQ requires a positive lfq_loss_weight.")
        if self.epochs <= 0:
            raise ValueError("omni_epochs must be positive.")
        if self.validation_sample_size < 0:
            raise ValueError("omni_validation_sample_size must be non-negative.")
        if self.train_sample_size < 0:
            raise ValueError("omni_train_sample_size must be non-negative.")
        if self.train_sample_size > 0 and self.validation_sample_size == 0:
            raise ValueError(
                "omni_train_sample_size requires a positive validation_sample_size."
            )
        if self.validation_sample_size > 0 and self.final_objective != "lfq_ce":
            raise ValueError(
                "omni_validation_sample_size currently requires final_objective='lfq_ce'."
            )
        if self.epoch_eval_interval < 0:
            raise ValueError("omni_epoch_eval_interval must be non-negative.")
        if self.lwc_lr <= 0 or self.let_lr <= 0:
            raise ValueError("OmniQuant learning rates must be positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be None or finite and positive.")
        if not self.use_lwc and not self.use_let:
            raise ValueError("At least one of LWC or LET must be enabled.")
        if self.learn_let and not self.use_let:
            raise ValueError("learn_let=True requires use_let=True.")


@dataclass(frozen=True)
class OmniQuantEpochMetric:
    epoch: int
    train_loss: float | None
    mean_grad_norm: float | None
    max_grad_norm: float | None
    eval_loss: float | None = None
    eval_mse_loss: float | None = None
    validation_loss: float | None = None
    validation_lfq_slot_losses: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class OmniQuantSummary:
    replaced_linears: int
    final_loss: float
    initial_loss: float
    let_scales: tuple[str, ...]
    objective: OmniQuantObjective = "mse"
    initial_lfq_slot_losses: tuple[tuple[str, float], ...] = ()
    final_lfq_slot_losses: tuple[tuple[str, float], ...] = ()
    lfq_slot_weights: tuple[float, float, float] | None = None
    lfq_loss_weight: float = 1.0
    initial_mse_loss: float | None = None
    final_mse_loss: float | None = None
    best_epoch: int | None = None
    epoch_metrics: tuple[OmniQuantEpochMetric, ...] = ()
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _total_grad_norm(parameters: Sequence[nn.Parameter]) -> torch.Tensor:
    """Report the global L2 gradient norm without modifying gradients."""
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    device = gradients[0].device
    return torch.norm(
        torch.stack([torch.norm(gradient, 2).to(device) for gradient in gradients]),
        2,
    )


def _fp8_cast_ste(x: torch.Tensor) -> torch.Tensor:
    """Cast to E4M3FN in forward while keeping an identity STE backward."""
    casted = x.to(require_fp8()).float()
    return x + (casted - x).detach()


def _fp4_e2m1_ste(x: torch.Tensor) -> torch.Tensor:
    """Map to the E2M1 codebook in forward with an identity STE backward."""
    quantized = fp4_e2m1_quantize(x).float()
    return x + (quantized - x).detach()


def _signed_qmax(quant_format: QuantFormat) -> float:
    if quant_format == "int4":
        return 7.0
    if quant_format == "int6":
        return 31.0
    if quant_format == "int8":
        return 127.0
    raise ValueError(f"Symmetric LWC needs an integer format, got {quant_format!r}")


def _unsigned_qmax(quant_format: QuantFormat) -> float:
    if quant_format == "int4":
        return 15.0
    if quant_format == "int6":
        return 63.0
    if quant_format == "int8":
        return 255.0
    raise ValueError(f"Asymmetric LWC needs an integer format, got {quant_format!r}")


def _trainable_activation_qdq(
    x: torch.Tensor,
    *,
    quant_format: QuantFormat,
    eps: float,
) -> torch.Tensor:
    """Activation QDQ with gradients through its dynamic absmax scale.

    The ordinary inference helper intentionally detaches calibration
    statistics.  Doing that while optimizing LET makes the QDQ backward look
    like an identity, so the reciprocal input/weight LET factors cancel and
    every LET gradient becomes exactly zero.  OmniQuant needs the quantizer
    scale to remain in the training graph.
    """

    normalized = validate_quant_format(quant_format)
    if normalized == "none":
        return x
    qmax = FP8_MAX if normalized == "fp8_e4m3fn" else quant_format_qmax(normalized)
    x_fp32 = x.float()
    threshold = x_fp32.abs().amax(dim=-1, keepdim=True)
    scale = threshold.clamp_min(eps) / float(qmax)
    normalized_x = (x_fp32 / scale).clamp(min=-float(qmax), max=float(qmax))
    if normalized == "fp8_e4m3fn":
        quantized = _fp8_cast_ste(normalized_x)
    elif normalized == "fp4_e2m1":
        quantized = _fp4_e2m1_ste(normalized_x)
    else:
        quantized = _round_ste(normalized_x)
    return (quantized * scale).to(x.dtype)


class _TrainableSymmetricLinear(nn.Module):
    """Frozen Linear with trainable per-channel LWC and optional LET scales.

    The historical class name is retained to avoid breaking local probes that
    import it directly.  ``config.weight_quant_scheme`` selects zero-centered
    symmetric INT/FP LWC or paper-style asymmetric integer LWC.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        config: OmniQuantConfig,
        let_parameters: nn.ParameterDict,
        input_let_name: str | None = None,
        weight_col_let_name: str | None = None,
        weight_col_repeat_let_name: str | None = None,
        weight_col_repeat_head_dim: int | None = None,
        weight_row_let_name: str | None = None,
        weight_row_let_inverse: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.execution_dtype = linear.weight.dtype
        self.config = config
        self.input_let_name = input_let_name
        self.weight_col_let_name = weight_col_let_name
        self.weight_col_repeat_let_name = weight_col_repeat_let_name
        self.weight_col_repeat_head_dim = weight_col_repeat_head_dim
        self.weight_row_let_name = weight_row_let_name
        self.weight_row_let_inverse = weight_row_let_inverse
        # The ParameterDict is owned by _TrainableOmniBlock.  Avoid registering
        # it again here while still allowing every wrapper to share a scale.
        object.__setattr__(self, "_let_parameters", let_parameters)
        # Keep a full-precision master copy for LET/LWC and QDQ, but cast the
        # dequantized tensor back to execution_dtype before F.linear.
        self.register_buffer("weight_fp", linear.weight.detach().float().clone(), persistent=False)
        self.register_buffer(
            "bias_fp",
            None if linear.bias is None else linear.bias.detach().float().clone(),
            persistent=False,
        )
        parameter_shape = (linear.out_features, 1)
        if config.weight_quant_scheme == "symmetric":
            self.clip_logits = nn.Parameter(
                torch.full(parameter_shape, float(config.init_lwc_logit), device=linear.weight.device)
            )
            self.clip_logits.requires_grad_(config.use_lwc)
            self.register_parameter("upper_clip_logits", None)
            self.register_parameter("lower_clip_logits", None)
        else:
            self.register_parameter("clip_logits", None)
            self.upper_clip_logits = nn.Parameter(
                torch.full(parameter_shape, float(config.init_lwc_logit), device=linear.weight.device)
            )
            self.lower_clip_logits = nn.Parameter(
                torch.full(parameter_shape, float(config.init_lwc_logit), device=linear.weight.device)
            )
            self.upper_clip_logits.requires_grad_(config.use_lwc)
            self.lower_clip_logits.requires_grad_(config.use_lwc)

    def lwc_parameters(self) -> list[nn.Parameter]:
        if self.config.weight_quant_scheme == "symmetric":
            assert self.clip_logits is not None
            return [self.clip_logits]
        assert self.upper_clip_logits is not None and self.lower_clip_logits is not None
        return [self.upper_clip_logits, self.lower_clip_logits]

    def lwc_state(self) -> dict[str, torch.Tensor]:
        if self.config.weight_quant_scheme == "symmetric":
            assert self.clip_logits is not None
            return {"clip_logits": self.clip_logits.detach().cpu()}
        assert self.upper_clip_logits is not None and self.lower_clip_logits is not None
        return {
            "upper_clip_logits": self.upper_clip_logits.detach().cpu(),
            "lower_clip_logits": self.lower_clip_logits.detach().cpu(),
        }


    def _let_scale(self, name: str | None, *, columns: bool) -> torch.Tensor | None:
        if name is None:
            return None
        scale = _positive_let_scale(self._let_parameters[name])
        expected = self.in_features if columns else self.out_features
        if scale.numel() != expected:
            raise ValueError(
                f"LET scale {name!r} has {scale.numel()} entries; expected {expected} for this Linear."
            )
        return scale

    def transformed_weight(self) -> torch.Tensor:
        weight = self.weight_fp
        col_scale = self._let_scale(self.weight_col_let_name, columns=True)
        if col_scale is not None:
            weight = weight * col_scale.to(weight.dtype).view(1, -1)
        if self.weight_col_repeat_let_name is not None:
            scale = _positive_let_scale(self._let_parameters[self.weight_col_repeat_let_name])
            if self.in_features % scale.numel() != 0:
                raise ValueError("GQA V->O LET scale cannot be repeated into o_proj input features.")
            scale = _repeat_gqa_scale(
                scale,
                output_features=self.in_features,
                head_dim=self.weight_col_repeat_head_dim,
            )
            weight = weight * scale.to(weight.dtype).view(1, -1)
        row_scale = self._let_scale(self.weight_row_let_name, columns=False)
        if row_scale is not None:
            row_scale = row_scale.to(weight.dtype).view(-1, 1)
            weight = weight / row_scale if self.weight_row_let_inverse else weight * row_scale
        return weight

    def _qdq_weight_tensor(self, weight: torch.Tensor, *, use_ste: bool) -> torch.Tensor:
        round_fn = _round_ste if use_ste else torch.round
        weight_fp32 = weight.float()
        if self.config.weight_quant_scheme == "asymmetric":
            assert self.upper_clip_logits is not None and self.lower_clip_logits is not None
            if self.config.use_lwc:
                upper_ratio = torch.sigmoid(self.upper_clip_logits)
                lower_ratio = torch.sigmoid(self.lower_clip_logits)
            else:
                upper_ratio = torch.ones_like(self.upper_clip_logits)
                lower_ratio = torch.ones_like(self.lower_clip_logits)
            upper = weight_fp32.amax(dim=1, keepdim=True) * upper_ratio
            lower = weight_fp32.amin(dim=1, keepdim=True) * lower_ratio
            qmax = _unsigned_qmax(self.config.weight_quant_format)
            scale = ((upper - lower) / qmax).clamp_min(self.config.eps)
            # OmniQuant's asymmetric MinMax/LWC equation:
            # z = -round(beta * min(W) / h), Q in [0, 2^N - 1].
            zero_point = -round_fn(lower / scale)
            q = (round_fn(weight_fp32 / scale) + zero_point).clamp(min=0.0, max=qmax)
            return ((q - zero_point) * scale).to(weight.dtype)

        assert self.clip_logits is not None
        if not self.config.use_lwc:
            ratio = torch.ones_like(self.clip_logits)
        else:
            ratio = torch.sigmoid(self.clip_logits)
        # Keep the transformed-weight absmax in the graph.  Detaching it makes
        # the STE path locally identical to the unquantized Linear and causes
        # the reciprocal LET factors to cancel to an exact zero gradient.
        threshold = weight_fp32.abs().amax(dim=1, keepdim=True) * ratio
        if self.config.weight_quant_format == "fp8_e4m3fn":
            scale = threshold.clamp_min(self.config.eps) / FP8_MAX
            normalized = (weight_fp32 / scale).clamp(min=-FP8_MAX, max=FP8_MAX)
            q = _fp8_cast_ste(normalized) if use_ste else normalized.to(require_fp8()).float()
            return (q * scale).to(weight.dtype)
        if self.config.weight_quant_format == "fp4_e2m1":
            scale = threshold.clamp_min(self.config.eps) / FP4_E2M1_MAX
            normalized = (weight_fp32 / scale).clamp(
                min=-FP4_E2M1_MAX,
                max=FP4_E2M1_MAX,
            )
            q = _fp4_e2m1_ste(normalized) if use_ste else fp4_e2m1_quantize(normalized)
            return (q * scale).to(weight.dtype)

        qmax = _signed_qmax(self.config.weight_quant_format)
        scale = threshold.clamp_min(self.config.eps) / qmax
        q = round_fn(weight_fp32 / scale).clamp(min=-qmax, max=qmax)
        return (q * scale).to(weight.dtype)

    def qdq_weight(self) -> torch.Tensor:
        return self._qdq_weight_tensor(self.transformed_weight(), use_ste=True)

    def finalize_qdq_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self._qdq_weight_tensor(weight, use_ste=False)

    def transformed_bias(self) -> torch.Tensor | None:
        if self.bias_fp is None:
            return None
        bias = self.bias_fp
        row_scale = self._let_scale(self.weight_row_let_name, columns=False)
        if row_scale is not None:
            row_scale = row_scale.to(bias.dtype)
            bias = bias / row_scale if self.weight_row_let_inverse else bias * row_scale
        return bias

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._let_scale(self.input_let_name, columns=True)
        if scale is not None:
            x = x / scale.to(device=x.device, dtype=x.dtype).view((1,) * (x.ndim - 1) + (-1,))
        if self.config.activation_quant_format != "none":
            x = _trainable_activation_qdq(
                x,
                quant_format=self.config.activation_quant_format,
                eps=self.config.eps,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare_input(x).to(dtype=self.execution_dtype)
        weight = self.qdq_weight().to(dtype=self.execution_dtype)
        bias = self.transformed_bias()
        if bias is not None:
            bias = bias.to(dtype=self.execution_dtype)
        return F.linear(prepared, weight, bias)


class _TrainableScaledNorm(nn.Module):
    """Execute a folded LET norm in the deployment model dtype."""

    def __init__(
        self,
        norm: nn.Module,
        *,
        let_parameters: nn.ParameterDict,
        scale_name: str,
    ) -> None:
        super().__init__()
        weight = getattr(norm, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 1:
            raise TypeError("Deployment-matched LET requires a norm with a 1D weight.")
        self.base_norm = copy.deepcopy(norm)
        self.base_norm.requires_grad_(False)
        self.execution_dtype = weight.dtype
        self.scale_name = scale_name
        object.__setattr__(self, "_let_parameters", let_parameters)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scale = _positive_let_scale(self._let_parameters[self.scale_name])
        weight = getattr(self.base_norm, "weight")
        replacements: dict[str, torch.Tensor] = {
            "weight": (weight.float() / scale).to(dtype=self.execution_dtype)
        }
        bias = getattr(self.base_norm, "bias", None)
        if torch.is_tensor(bias):
            replacements["bias"] = (bias.float() / scale).to(dtype=self.execution_dtype)
        return torch.func.functional_call(
            self.base_norm,
            replacements,
            (hidden_states.to(dtype=self.execution_dtype),),
            strict=False,
        )


class _TrainableOmniBlock(nn.Module):
    """Copied decoder block plus the shared LET parameter bank."""

    def __init__(self, block: nn.Module, *, config: OmniQuantConfig, init_scales: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        q_proj = block.get_submodule("self_attn.q_proj")
        if not isinstance(q_proj, nn.Linear):
            raise TypeError("Expected self_attn.q_proj to be nn.Linear.")
        if not q_proj.weight.is_floating_point():
            raise TypeError("OmniQuant calibration requires floating-point model weights.")
        # Execute the copied block in the installed model dtype. Master
        # weights, LET/LWC parameters, QDQ arithmetic, and losses remain FP32.
        self.inference_dtype = q_proj.weight.dtype
        self.calibration_dtype = self.inference_dtype
        self.block = block.to(dtype=self.inference_dtype)
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)
        self.config = config
        self.gqa_head_dim = self._head_dim()
        self.let_parameters = nn.ParameterDict()
        if config.use_let:
            qkv_initial = None if config.let_init == "ones" else init_scales.get("self_attn.q_proj")
            mlp_initial = None if config.let_init == "ones" else init_scales.get("mlp.gate_proj")
            vo_initial = (
                None
                if config.let_init == "ones"
                else self._gqa_vo_initial(init_scales.get("self_attn.o_proj"))
            )
            self._add_let_parameter("qkv", qkv_initial, self._hidden_size())
            self._add_let_parameter("mlp", mlp_initial, self._hidden_size())
            self._add_let_parameter(
                "vo",
                vo_initial,
                self._kv_hidden_size(),
            )
            self._replace_norms()
        self._replace_linears()

    def _hidden_size(self) -> int:
        q_proj = self.block.get_submodule("self_attn.q_proj")
        if not isinstance(q_proj, nn.Linear):
            raise TypeError("Expected self_attn.q_proj to be nn.Linear.")
        return q_proj.in_features

    def _kv_hidden_size(self) -> int:
        v_proj = self.block.get_submodule("self_attn.v_proj")
        if not isinstance(v_proj, nn.Linear):
            raise TypeError("Expected self_attn.v_proj to be nn.Linear.")
        return v_proj.out_features

    def _add_let_parameter(self, name: str, initial: torch.Tensor | None, size: int) -> None:
        if initial is None:
            initial = torch.ones(size, device=_module_device(self.block), dtype=torch.float32)
        initial = initial.detach().float().reshape(-1)
        if initial.numel() != size:
            initial = torch.ones(size, device=initial.device, dtype=torch.float32)
        initial = initial.to(device=_module_device(self.block))
        if not torch.isfinite(initial).all() or torch.any(initial <= 0):
            raise ValueError(
                f"LET scale {name!r} must contain finite, strictly positive values."
            )
        self.let_parameters[name] = nn.Parameter(torch.log(initial))
        self.let_parameters[name].requires_grad_(self.config.learn_let)

    def _head_dim(self) -> int:
        attention = self.block.get_submodule("self_attn")
        head_dim = int(getattr(attention, "head_dim", self._kv_hidden_size()))
        if head_dim <= 0 or self._kv_hidden_size() % head_dim != 0:
            raise ValueError("Cannot infer a valid Qwen3 attention head dimension for GQA LET.")
        return head_dim

    def _gqa_vo_initial(self, o_scale: torch.Tensor | None) -> torch.Tensor | None:
        if o_scale is None:
            return None
        o_proj = self.block.get_submodule("self_attn.o_proj")
        if not isinstance(o_proj, nn.Linear):
            return None
        scale = o_scale.detach().float().reshape(-1)
        kv_features = self._kv_hidden_size()
        head_dim = self.gqa_head_dim
        if scale.numel() != o_proj.in_features or o_proj.in_features % kv_features != 0:
            return None
        kv_heads = kv_features // head_dim
        repeats = o_proj.in_features // kv_features
        # Independent SmoothQuant scales on O-proj are tied across the query
        # heads sharing one KV head.  The geometric mean is the least-squares
        # projection in log-scale space and preserves positivity.
        return torch.exp(
            torch.log(scale.clamp_min(self.config.eps))
            .view(kv_heads, repeats, head_dim)
            .mean(dim=1)
        ).reshape(-1)

    def _replace(self, name: str, **kwargs: Any) -> None:
        linear = self.block.get_submodule(name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected {name} to be nn.Linear, got {type(linear)!r}.")
        parent_name, child_name = name.rsplit(".", 1)
        setattr(
            self.block.get_submodule(parent_name),
            child_name,
            _TrainableSymmetricLinear(linear, config=self.config, let_parameters=self.let_parameters, **kwargs),
        )

    def _replace_norms(self) -> None:
        for norm_name, scale_name in (
            ("input_layernorm", "qkv"),
            ("post_attention_layernorm", "mlp"),
        ):
            norm = self.block.get_submodule(norm_name)
            setattr(
                self.block,
                norm_name,
                _TrainableScaledNorm(
                    norm,
                    let_parameters=self.let_parameters,
                    scale_name=scale_name,
                ),
            )

    def _replace_linears(self) -> None:
        qkv = "qkv" if self.config.use_let else None
        mlp = "mlp" if self.config.use_let else None
        vo = "vo" if self.config.use_let else None
        # Norm-side inverse LET scales are already applied by
        # _TrainableScaledNorm in the model execution dtype.
        self._replace("self_attn.q_proj", weight_col_let_name=qkv)
        self._replace("self_attn.k_proj", weight_col_let_name=qkv)
        self._replace(
            "self_attn.v_proj",
            weight_col_let_name=qkv,
            weight_row_let_name=vo,
            weight_row_let_inverse=True,
        )
        self._replace(
            "self_attn.o_proj",
            weight_col_repeat_let_name=vo,
            weight_col_repeat_head_dim=self.gqa_head_dim if vo is not None else None,
        )
        self._replace("mlp.gate_proj", weight_col_let_name=mlp)
        self._replace("mlp.up_proj", weight_col_let_name=mlp)
        self._replace("mlp.down_proj")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        args = _floating_tree_to_dtype(args, self.calibration_dtype)
        kwargs = _floating_tree_to_dtype(kwargs, self.calibration_dtype)
        return self.block(*args, **kwargs)


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"Could not find a tensor in {type(value)!r}")


def _positive_let_scale(log_scale: torch.Tensor) -> torch.Tensor:
    """Map the unconstrained log parameter to a strictly positive LET scale."""
    return torch.exp(log_scale)


def _repeat_gqa_scale(
    kv_scale: torch.Tensor,
    *,
    output_features: int,
    head_dim: int | None,
) -> torch.Tensor:
    """Repeat complete KV-head scale vectors in Qwen3 query-head order."""
    scale = kv_scale.reshape(-1)
    if output_features % scale.numel() != 0:
        raise ValueError("GQA scale size does not divide O-proj input features.")
    resolved_head_dim = scale.numel() if head_dim is None else int(head_dim)
    if resolved_head_dim <= 0 or scale.numel() % resolved_head_dim != 0:
        raise ValueError("Invalid head_dim for GQA LET scale expansion.")
    repeats = output_features // scale.numel()
    kv_heads = scale.numel() // resolved_head_dim
    return (
        scale.view(kv_heads, 1, resolved_head_dim)
        .expand(kv_heads, repeats, resolved_head_dim)
        .reshape(output_features)
    )


def _tree_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_tree_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_cpu(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _tree_cpu(item) for key, item in value.items()}
    return value


def _floating_tree_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    """Cast only floating tensors in a nested model input tree."""
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_floating_tree_to_dtype(item, dtype) for item in value)
    if isinstance(value, list):
        return [_floating_tree_to_dtype(item, dtype) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _floating_tree_to_dtype(item, dtype)
            for key, item in value.items()
        }
    return value


def _advance_cpu(layer: nn.Module, batches: Sequence[Batch]) -> list[Batch]:
    result: list[Batch] = []
    device = _module_device(layer)
    was_training = layer.training
    layer.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, device)
                kwargs = _move_tree_to_device(kwargs, device)
                output = layer(*args, **kwargs)
                hidden = _first_tensor(output).detach()
                next_args: tuple[Any, ...] = (hidden, *args[1:]) if args else ()
                next_kwargs = dict(kwargs)
                if not next_args:
                    if "hidden_states" in next_kwargs:
                        next_kwargs["hidden_states"] = hidden
                    else:
                        next_args = (hidden,)
                result.append(_tree_cpu((next_args, next_kwargs)))
    finally:
        layer.train(was_training)
    return result


class _LFQOutputProjector(nn.Module):
    """Frozen final norm and slot-specific LM-head rows for SID-chain LFQ."""

    def __init__(
        self,
        *,
        final_norm: nn.Module,
        output_head: nn.Module,
        token_ids: Mapping[str, Sequence[int]],
    ) -> None:
        super().__init__()
        if set(token_ids) != set(LFQ_SLOT_NAMES):
            raise ValueError("LFQ token IDs must contain exactly the SID_a, SID_b, and SID_c slots.")
        weight = getattr(output_head, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise TypeError("LFQ requires an output head with a rank-2 weight tensor.")

        # Match deployment: the frozen final norm and LM-head rows execute in
        # the model dtype. LFQ logits are promoted to FP32 only by the loss.
        self.execution_dtype = weight.dtype
        self.final_norm = copy.deepcopy(final_norm).to(dtype=self.execution_dtype)
        self.final_norm.requires_grad_(False)
        self.final_norm.eval()
        bias = getattr(output_head, "bias", None)
        for slot in LFQ_SLOT_NAMES:
            ids = torch.as_tensor(tuple(token_ids[slot]), device=weight.device, dtype=torch.long)
            if ids.numel() == 0:
                raise ValueError(f"LFQ SID_{slot} vocabulary must not be empty.")
            if torch.unique(ids).numel() != ids.numel():
                raise ValueError(f"LFQ SID_{slot} vocabulary token IDs must be unique.")
            if int(ids.min()) < 0 or int(ids.max()) >= weight.shape[0]:
                raise ValueError(f"LFQ SID_{slot} token IDs are outside the LM-head vocabulary.")
            self.register_buffer(
                f"weight_{slot}",
                weight.detach().index_select(0, ids).to(dtype=self.execution_dtype).clone(),
                persistent=False,
            )
            self.register_buffer(
                f"bias_{slot}",
                None
                if bias is None
                else bias.detach().index_select(0, ids).to(dtype=self.execution_dtype).clone(),
                persistent=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden_states.ndim < 2 or hidden_states.shape[-2] < len(LFQ_SLOT_NAMES):
            raise ValueError("LFQ expects hidden states with at least three sequence positions.")
        # Inputs end in <|sid_begin|>, a_gt, b_gt. Their last three hidden
        # states predict SID_a, SID_b, and SID_c respectively.
        selected = hidden_states.narrow(
            dim=-2,
            start=hidden_states.shape[-2] - len(LFQ_SLOT_NAMES),
            length=len(LFQ_SLOT_NAMES),
        )
        normalized = self.final_norm(selected.to(dtype=self.execution_dtype))
        return {
            slot: F.linear(
                normalized.select(dim=-2, index=slot_idx),
                getattr(self, f"weight_{slot}"),
                getattr(self, f"bias_{slot}"),
            )
            for slot_idx, slot in enumerate(LFQ_SLOT_NAMES)
        }




def _normalized_lfq_slot_weights(config: OmniQuantConfig) -> dict[str, float]:
    total = float(sum(config.lfq_slot_weights))
    return {
        slot: float(weight) / total
        for slot, weight in zip(LFQ_SLOT_NAMES, config.lfq_slot_weights)
    }


def _lfq_soft_cross_entropy(
    prediction: torch.Tensor,
    teacher_probabilities: Mapping[str, torch.Tensor],
    projector: _LFQOutputProjector,
    slot_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_logits = {
        slot: logits.float() for slot, logits in projector(prediction).items()
    }
    if set(teacher_probabilities) != set(LFQ_SLOT_NAMES):
        raise ValueError("LFQ teacher probabilities must contain SID_a, SID_b, and SID_c.")
    slot_losses: dict[str, torch.Tensor] = {}
    for slot in LFQ_SLOT_NAMES:
        teacher = teacher_probabilities[slot].to(
            device=student_logits[slot].device,
            dtype=torch.float32,
        )
        if student_logits[slot].shape != teacher.shape:
            raise ValueError(
                f"LFQ SID_{slot} teacher/student logit shapes differ: "
                f"{tuple(teacher.shape)} vs {tuple(student_logits[slot].shape)}."
            )
        slot_losses[slot] = -(
            teacher * F.log_softmax(student_logits[slot], dim=-1)
        ).sum(dim=-1).mean()
    total_loss = sum(slot_weights[slot] * slot_losses[slot] for slot in LFQ_SLOT_NAMES)
    return total_loss, slot_losses




def _train_block(
    *,
    teacher_block: nn.Module,
    train_block: _TrainableOmniBlock,
    fp_inputs: Sequence[Batch],
    quant_inputs: Sequence[Batch],
    config: OmniQuantConfig,
    lfq_projector: _LFQOutputProjector | None = None,
    layer_idx: int,
) -> tuple[
    float,
    float,
    dict[str, float],
    dict[str, float],
    float,
    float,
    int | None,
    tuple[OmniQuantEpochMetric, ...],
]:
    if len(fp_inputs) != len(quant_inputs):
        raise ValueError("FP and quantized calibration streams must contain the same number of batches.")
    validation_fp_inputs: Sequence[Batch] = ()
    validation_quant_inputs: Sequence[Batch] = ()
    if config.validation_sample_size > 0:
        if lfq_projector is None:
            raise ValueError("LFQ validation requires the SID-slot output projector.")
        if config.validation_sample_size >= len(fp_inputs):
            raise ValueError(
                "omni_validation_sample_size must be smaller than the number of "
                "loaded calibration samples."
            )
        validation_start = len(fp_inputs) - config.validation_sample_size
        validation_fp_inputs = fp_inputs[validation_start:]
        validation_quant_inputs = quant_inputs[validation_start:]
        train_count = config.train_sample_size or validation_start
        if train_count > validation_start:
            raise ValueError(
                "omni_train_sample_size cannot exceed the number of samples "
                "before the held-out validation tail."
            )
        fp_inputs = fp_inputs[:train_count]
        quant_inputs = quant_inputs[:train_count]
        print(
            f"[omniquant][validation] layer={layer_idx} "
            f"train_samples={len(fp_inputs)} "
            f"unused_samples={validation_start - train_count} "
            f"validation_samples={len(validation_fp_inputs)} split=fixed_tail"
        )

    lwc_parameters = [
        parameter
        for wrapper in train_block.modules()
        if isinstance(wrapper, _TrainableSymmetricLinear) and config.use_lwc
        for parameter in wrapper.lwc_parameters()
    ]
    let_parameters = list(train_block.let_parameters.parameters()) if config.learn_let else []
    groups = []
    if lwc_parameters:
        groups.append({"params": lwc_parameters, "lr": config.lwc_lr})
    if let_parameters:
        groups.append({"params": let_parameters, "lr": config.let_lr})
    optimizer = (
        torch.optim.AdamW(groups, weight_decay=config.weight_decay)
        if groups
        else None
    )
    device = _module_device(train_block)
    teacher_block.eval()
    train_block.eval()
    lfq_teacher_probabilities: list[dict[str, torch.Tensor]] | None = None
    validation_lfq_teacher_probabilities: list[dict[str, torch.Tensor]] | None = None
    slot_weights = (
        _normalized_lfq_slot_weights(config)
        if lfq_projector is not None
        else {}
    )
    if lfq_projector is not None:
        lfq_projector.eval()
        lfq_teacher_probabilities = []
        # Three 8192-way float32 distributions use about 12 MiB for 128 samples.
        with torch.no_grad():
            for fp_batch in fp_inputs:
                fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
                fp_args = _move_tree_to_device(fp_args, device)
                fp_kwargs = _move_tree_to_device(fp_kwargs, device)
                target = _first_tensor(teacher_block(*fp_args, **fp_kwargs))
                projected = lfq_projector(target)
                probabilities = {
                    slot: F.softmax(projected[slot].float(), dim=-1).detach().cpu()
                    for slot in LFQ_SLOT_NAMES
                }
                lfq_teacher_probabilities.append(probabilities)
        if validation_fp_inputs:
            validation_lfq_teacher_probabilities = []
            with torch.no_grad():
                for fp_batch in validation_fp_inputs:
                    fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
                    fp_args = _move_tree_to_device(fp_args, device)
                    fp_kwargs = _move_tree_to_device(fp_kwargs, device)
                    target = _first_tensor(teacher_block(*fp_args, **fp_kwargs))
                    projected = lfq_projector(target)
                    validation_lfq_teacher_probabilities.append(
                        {
                            slot: F.softmax(projected[slot].float(), dim=-1)
                            .detach()
                            .cpu()
                            for slot in LFQ_SLOT_NAMES
                        }
                    )

    def compute_batch_loss(
        batch_idx: int,
        fp_batch: Batch,
        quant_batch: Batch,
        *,
        evaluate_lfq: bool = True,
        evaluate_mse: bool = False,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor | None,
    ]:
        slot_losses: dict[str, torch.Tensor] = {}
        prediction: torch.Tensor | None = None
        mse_loss: torch.Tensor | None = None
        if lfq_teacher_probabilities is None:
            quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
            quant_args = _move_tree_to_device(quant_args, device)
            quant_kwargs = _move_tree_to_device(quant_kwargs, device)
            prediction = _first_tensor(train_block(*quant_args, **quant_kwargs))
            fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
            fp_args = _move_tree_to_device(fp_args, device)
            fp_kwargs = _move_tree_to_device(fp_kwargs, device)
            with torch.no_grad():
                target = _first_tensor(teacher_block(*fp_args, **fp_kwargs)).detach()
            mse_loss = F.mse_loss(prediction.float(), target.float())
            total_loss = mse_loss
        else:
            if not evaluate_lfq:
                raise RuntimeError("LFQ loss cannot be disabled for an LFQ objective.")
            quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
            quant_args = _move_tree_to_device(quant_args, device)
            quant_kwargs = _move_tree_to_device(quant_kwargs, device)
            prediction = _first_tensor(train_block(*quant_args, **quant_kwargs))
            assert lfq_projector is not None
            base_loss, slot_losses = _lfq_soft_cross_entropy(
                prediction,
                lfq_teacher_probabilities[batch_idx],
                lfq_projector,
                slot_weights,
            )
            total_loss = config.lfq_loss_weight * base_loss

            # Reconstruction MSE is diagnostic only for LFQ and is never added
            # to the loss used for backpropagation.
            if evaluate_mse:
                fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
                fp_args = _move_tree_to_device(fp_args, device)
                fp_kwargs = _move_tree_to_device(fp_kwargs, device)
                with torch.no_grad():
                    target = _first_tensor(teacher_block(*fp_args, **fp_kwargs)).detach()
                mse_loss = F.mse_loss(prediction.float(), target.float())

        return total_loss, slot_losses, mse_loss

    def evaluate_parameters() -> tuple[
        float,
        dict[str, float],
        float,
    ]:
        """Evaluate one fixed parameter state over the full calibration set."""
        was_training = train_block.training
        train_block.eval()
        total = 0.0
        count = 0
        slot_totals = {slot: 0.0 for slot in LFQ_SLOT_NAMES}
        mse_total = 0.0
        with torch.no_grad():
            for batch_idx, (fp_batch, quant_batch) in enumerate(
                zip(fp_inputs, quant_inputs)
            ):
                loss, slot_losses, mse_loss = compute_batch_loss(
                    batch_idx,
                    fp_batch,
                    quant_batch,
                    evaluate_mse=True,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite OmniQuant loss during fixed-parameter evaluation."
                    )
                total += float(loss)
                count += 1
                if lfq_teacher_probabilities is not None:
                    for slot in LFQ_SLOT_NAMES:
                        slot_totals[slot] += float(slot_losses[slot])
                if mse_loss is None or not torch.isfinite(mse_loss):
                    raise FloatingPointError(
                        "Non-finite OmniQuant MSE during fixed-parameter evaluation."
                    )
                mse_total += float(mse_loss)
        if was_training:
            train_block.train()
        evaluated_slot_losses = (
            {
                slot: slot_totals[slot] / max(1, count)
                for slot in LFQ_SLOT_NAMES
            }
            if lfq_teacher_probabilities is not None
            else {}
        )
        return (
            total / max(1, count),
            evaluated_slot_losses,
            mse_total / max(1, count),
        )

    def evaluate_validation_parameters() -> tuple[float, dict[str, float]] | None:
        """Evaluate held-out LFQ samples without changing or selecting parameters."""
        if validation_lfq_teacher_probabilities is None:
            return None
        assert lfq_projector is not None
        was_training = train_block.training
        train_block.eval()
        total = 0.0
        slot_totals = {slot: 0.0 for slot in LFQ_SLOT_NAMES}
        with torch.no_grad():
            for quant_batch, teacher_probabilities in zip(
                validation_quant_inputs,
                validation_lfq_teacher_probabilities,
            ):
                quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
                quant_args = _move_tree_to_device(quant_args, device)
                quant_kwargs = _move_tree_to_device(quant_kwargs, device)
                prediction = _first_tensor(train_block(*quant_args, **quant_kwargs))
                loss, slot_losses = _lfq_soft_cross_entropy(
                    prediction,
                    teacher_probabilities,
                    lfq_projector,
                    slot_weights,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite held-out LFQ validation loss."
                    )
                total += float(loss)
                for slot in LFQ_SLOT_NAMES:
                    slot_totals[slot] += float(slot_losses[slot])
        if was_training:
            train_block.train()
        count = len(validation_quant_inputs)
        return (
            total / max(1, count),
            {
                slot: slot_totals[slot] / max(1, count)
                for slot in LFQ_SLOT_NAMES
            },
        )

    # Epoch zero is a fixed-state full-calibration measurement. It also gives
    # best-state tracking a safe fallback if every optimizer update is worse.
    initial_loss, initial_slot_losses, initial_mse_loss = evaluate_parameters()
    initial_validation = evaluate_validation_parameters()
    epoch_metrics: list[OmniQuantEpochMetric] = [
        OmniQuantEpochMetric(
            epoch=0,
            train_loss=None,
            mean_grad_norm=None,
            max_grad_norm=None,
            eval_loss=initial_loss,
            eval_mse_loss=initial_mse_loss,
            validation_loss=(
                initial_validation[0] if initial_validation is not None else None
            ),
            validation_lfq_slot_losses=(
                tuple(
                    (slot, initial_validation[1][slot])
                    for slot in LFQ_SLOT_NAMES
                )
                if initial_validation is not None
                else ()
            ),
        )
    ]
    if initial_validation is not None:
        slot_text = ",".join(
            f"{slot}:{initial_validation[1][slot]:.6e}"
            for slot in LFQ_SLOT_NAMES
        )
        print(
            f"[omniquant][validation] layer={layer_idx} epoch=0/{config.epochs} "
            f"loss={initial_validation[0]:.6e} slot_loss={slot_text}"
        )

    if optimizer is None:
        return (
            initial_loss,
            initial_loss,
            initial_slot_losses,
            dict(initial_slot_losses),
            initial_mse_loss,
            initial_mse_loss,
            0,
            tuple(epoch_metrics),
        )

    trainable_parameters = lwc_parameters + let_parameters

    def snapshot_trainable_parameters() -> tuple[torch.Tensor, ...]:
        return tuple(
            parameter.detach().cpu().clone()
            for parameter in trainable_parameters
        )

    def restore_trainable_parameters(snapshot: Sequence[torch.Tensor]) -> None:
        if len(snapshot) != len(trainable_parameters):
            raise ValueError("OmniQuant best-state parameter count mismatch.")
        with torch.no_grad():
            for parameter, saved in zip(trainable_parameters, snapshot):
                if tuple(parameter.shape) != tuple(saved.shape):
                    raise ValueError("OmniQuant best-state parameter shape mismatch.")
                parameter.copy_(
                    saved.to(device=parameter.device, dtype=parameter.dtype)
                )

    best_epoch = 0
    best_loss = initial_loss
    best_evaluation = (
        initial_loss,
        dict(initial_slot_losses),
        initial_mse_loss,
    )
    best_parameters = snapshot_trainable_parameters()
    checked_let_gradient = not let_parameters
    train_block.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_grad_norm_total = 0.0
        epoch_grad_norm_max = 0.0
        for batch_idx, (fp_batch, quant_batch) in enumerate(zip(fp_inputs, quant_inputs)):
            loss, _slot_losses, _mse_loss = compute_batch_loss(
                batch_idx,
                fp_batch,
                quant_batch,
                evaluate_lfq=config.lfq_loss_weight > 0.0,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite OmniQuant reconstruction loss before optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable_parameters
            ):
                raise FloatingPointError("Non-finite OmniQuant gradient encountered.")
            if not checked_let_gradient:
                checked_let_gradient = True
                if not any(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                    for parameter in let_parameters
                ):
                    raise RuntimeError(
                        "All LET gradients are zero; the trainable QDQ path is disconnected."
                    )
            grad_norm_tensor = (
                _total_grad_norm(trainable_parameters)
                if config.max_grad_norm is None
                else torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=config.max_grad_norm,
                )
            )
            grad_norm = float(grad_norm_tensor.detach())
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Non-finite OmniQuant gradient norm.")
            epoch_grad_norm_total += grad_norm
            epoch_grad_norm_max = max(epoch_grad_norm_max, grad_norm)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in trainable_parameters):
                raise FloatingPointError("Non-finite OmniQuant parameter encountered after optimizer step.")
            epoch_loss += float(loss.detach())

        train_loss = epoch_loss / max(1, len(fp_inputs))
        mean_grad_norm = epoch_grad_norm_total / max(1, len(fp_inputs))
        should_evaluate = (
            config.epoch_eval_interval > 0
            and (
                epoch % config.epoch_eval_interval == 0
                or epoch == config.epochs
            )
        )
        evaluated = evaluate_parameters() if should_evaluate else None
        validation_evaluated = evaluate_validation_parameters()
        is_best = False
        if evaluated is not None and evaluated[0] < best_loss:
            best_epoch = epoch
            best_loss = evaluated[0]
            best_evaluation = (
                evaluated[0],
                dict(evaluated[1]),
                evaluated[2],
            )
            best_parameters = snapshot_trainable_parameters()
            is_best = True
        epoch_metrics.append(
            OmniQuantEpochMetric(
                epoch=epoch,
                train_loss=train_loss,
                mean_grad_norm=mean_grad_norm,
                max_grad_norm=epoch_grad_norm_max,
                eval_loss=evaluated[0] if evaluated is not None else None,
                eval_mse_loss=evaluated[2] if evaluated is not None else None,
                validation_loss=(
                    validation_evaluated[0]
                    if validation_evaluated is not None
                    else None
                ),
                validation_lfq_slot_losses=(
                    tuple(
                        (slot, validation_evaluated[1][slot])
                        for slot in LFQ_SLOT_NAMES
                    )
                    if validation_evaluated is not None
                    else ()
                ),
            )
        )
        evaluation_text = (
            (
                f" eval_loss={evaluated[0]:.6e} "
                f"eval_mse={evaluated[2]:.6e}"
            )
            if evaluated is not None
            else ""
        )
        best_marker = " best" if is_best else ""
        validation_text = ""
        if validation_evaluated is not None:
            validation_slot_text = ",".join(
                f"{slot}:{validation_evaluated[1][slot]:.6e}"
                for slot in LFQ_SLOT_NAMES
            )
            validation_text = (
                f" val_loss={validation_evaluated[0]:.6e} "
                f"val_slot={validation_slot_text}"
            )
        print(
            f"[omniquant][epoch] layer={layer_idx} "
            f"epoch={epoch}/{config.epochs} train_loss={train_loss:.6e} "
            f"grad_norm_mean={mean_grad_norm:.6e} "
            f"grad_norm_max={epoch_grad_norm_max:.6e}"
            f"{evaluation_text}{validation_text}{best_marker}"
        )
        train_block.train()

    if config.epoch_eval_interval > 0:
        restore_trainable_parameters(best_parameters)
        final_loss, final_slot_losses, final_mse_loss = best_evaluation
    else:
        # Preserve legacy compute cost when epoch evaluation is disabled:
        # only evaluate the final fixed parameter state once.
        final_loss, final_slot_losses, final_mse_loss = evaluate_parameters()
        last_metric = epoch_metrics[-1]
        epoch_metrics[-1] = OmniQuantEpochMetric(
            epoch=last_metric.epoch,
            train_loss=last_metric.train_loss,
            mean_grad_norm=last_metric.mean_grad_norm,
            max_grad_norm=last_metric.max_grad_norm,
            eval_loss=final_loss,
            eval_mse_loss=final_mse_loss,
            validation_loss=last_metric.validation_loss,
            validation_lfq_slot_losses=last_metric.validation_lfq_slot_losses,
        )
        best_epoch = None
    train_block.eval()

    if config.epoch_eval_interval > 0:
        print(
            f"[omniquant][best] layer={layer_idx} epoch={best_epoch} "
            f"eval_loss={final_loss:.6e} eval_mse={final_mse_loss:.6e}"
        )

    return (
        initial_loss,
        final_loss,
        initial_slot_losses,
        final_slot_losses,
        initial_mse_loss,
        final_mse_loss,
        best_epoch,
        tuple(epoch_metrics),
    )


def _get_linear(module: nn.Module, name: str) -> nn.Linear:
    linear = module.get_submodule(name)
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected {name} to remain nn.Linear during finalization.")
    return linear


def _scale(train_block: _TrainableOmniBlock, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _positive_let_scale(train_block.let_parameters[name]).to(device, dtype)



def _finalize_block(train_block: _TrainableOmniBlock) -> tuple[nn.Module, int]:
    """Fold/QDQ from FP32 masters and materialize deployment-dtype execution."""
    source = copy.deepcopy(train_block.block)
    config = train_block.config
    device = _module_device(source) or torch.device("cpu")
    names = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    trained_linears = {name: train_block.block.get_submodule(name) for name in names}
    if config.use_let:
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            wrapped_norm = source.get_submodule(norm_name)
            if not isinstance(wrapped_norm, _TrainableScaledNorm):
                raise TypeError(f"Expected trainable LET norm wrapper for {norm_name}.")
            setattr(source, norm_name, copy.deepcopy(wrapped_norm.base_norm))
    # Recover ordinary FP linears in the copied block.  The LET folding below
    # operates on their frozen source weights, while learned LWC parameters are
    # read from the training wrappers retained above.
    for name in names:
        wrapped = source.get_submodule(name)
        if not isinstance(wrapped, _TrainableSymmetricLinear):
            raise TypeError(f"Expected trainable wrapper for {name}.")
        linear = nn.Linear(wrapped.in_features, wrapped.out_features, bias=wrapped.bias_fp is not None)
        linear = linear.to(device=wrapped.weight_fp.device, dtype=wrapped.weight_fp.dtype)
        with torch.no_grad():
            linear.weight.copy_(wrapped.weight_fp)
            if linear.bias is not None and wrapped.bias_fp is not None:
                linear.bias.copy_(wrapped.bias_fp)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(source.get_submodule(parent_name), child_name, linear)

    if config.use_let:
        qkv_scale = _scale(train_block, "qkv", device, _get_linear(source, "self_attn.q_proj").weight.dtype)
        mlp_scale = _scale(train_block, "mlp", device, _get_linear(source, "mlp.gate_proj").weight.dtype)
        vo_scale = _scale(train_block, "vo", device, _get_linear(source, "self_attn.v_proj").weight.dtype)
        _fold_norm_input(source, "input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), qkv_scale)
        _fold_norm_input(source, "post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), mlp_scale)
        _fold_gqa_v_to_o(source, vo_scale)

    source_linears: dict[str, nn.Linear] = {name: _get_linear(source, name) for name in names}
    replaced = 0
    for name, linear in source_linears.items():
        trained = trained_linears[name]
        if not isinstance(trained, _TrainableSymmetricLinear):
            raise TypeError(f"Expected trainable wrapper for {name}.")
        # Re-evaluate LWC on the final folded weight.  For qkv/gate/up/v this
        # is exactly the training transform; for O it includes the folded GQA
        # transform and is the correct static inference representation.
        weight = linear.weight.detach()
        with torch.no_grad():
            qdq = trained.finalize_qdq_weight(weight)
        parent_name, child_name = name.rsplit(".", 1)
        act_quant: ActQuant = "none" if config.activation_quant_format == "none" else "per_token"
        setattr(
            source.get_submodule(parent_name), child_name,
            OmniQuantFakeQuantLinear(
                weight_qdq=qdq,
                bias=linear.bias,
                act_quant=act_quant,
                weight_quant_format=config.weight_quant_format,
                activation_quant_format=config.activation_quant_format,
                eps=config.eps,
            ),
        )
        replaced += 1
    source.to(dtype=train_block.inference_dtype)
    return source, replaced


def _fold_norm_input(module: nn.Module, norm_name: str, linear_names: Sequence[str], scale: torch.Tensor) -> None:
    norm = module.get_submodule(norm_name)
    weight = getattr(norm, "weight", None)
    if not torch.is_tensor(weight) or weight.numel() != scale.numel():
        raise ValueError(f"LET scale cannot fold into {norm_name}.")
    with torch.no_grad():
        folded_weight = (weight.float() / scale.float().to(weight.device)).to(weight.dtype)
        weight.copy_(folded_weight.reshape_as(weight))
        bias = getattr(norm, "bias", None)
        if torch.is_tensor(bias):
            folded_bias = (bias.float() / scale.float().to(bias.device)).to(bias.dtype)
            bias.copy_(folded_bias.reshape_as(bias))
        for name in linear_names:
            linear = _get_linear(module, name)
            linear.weight.mul_(scale.to(linear.weight.device, linear.weight.dtype).view(1, -1))


def _fold_gqa_v_to_o(module: nn.Module, kv_scale: torch.Tensor) -> None:
    v_proj = _get_linear(module, "self_attn.v_proj")
    o_proj = _get_linear(module, "self_attn.o_proj")
    if v_proj.out_features != kv_scale.numel():
        raise ValueError("V LET scale does not match v_proj output dimension.")
    if o_proj.in_features % v_proj.out_features != 0:
        raise ValueError("Qwen3 GQA dimensions do not permit an exact V->O LET fold.")
    attention = module.get_submodule("self_attn")
    head_dim = int(getattr(attention, "head_dim", v_proj.out_features))
    repeated = _repeat_gqa_scale(
        kv_scale,
        output_features=o_proj.in_features,
        head_dim=head_dim,
    )
    with torch.no_grad():
        v_proj.weight.div_(kv_scale.to(v_proj.weight.device, v_proj.weight.dtype).view(-1, 1))
        if v_proj.bias is not None:
            v_proj.bias.div_(kv_scale.to(v_proj.bias.device, v_proj.bias.dtype))
        o_proj.weight.mul_(repeated.to(o_proj.weight.device, o_proj.weight.dtype).view(1, -1))


def _layer_objective(
    config: OmniQuantConfig,
    *,
    layer_idx: int,
    final_layer_idx: int,
) -> OmniQuantObjective:
    return (
        config.final_objective
        if config.final_objective != "mse" and layer_idx == final_layer_idx
        else "mse"
    )




def _load_omniquant_checkpoint(
    checkpoint_path: Path,
    *,
    layer_idx: int,
) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing OmniQuant checkpoint for layer {layer_idx}: {checkpoint_path}"
        )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Unexpected checkpoint payload type for layer {layer_idx}: {type(state)!r}"
        )
    if int(state.get("layer_idx", -1)) != layer_idx:
        raise ValueError(f"Checkpoint layer index mismatch in {checkpoint_path}.")
    return state


def _checkpoint_epoch_metrics(
    state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> tuple[OmniQuantEpochMetric, ...]:
    raw_metrics = state.get("epoch_metrics", ())
    if not isinstance(raw_metrics, Sequence) or isinstance(
        raw_metrics,
        (str, bytes),
    ):
        raise TypeError(f"Checkpoint epoch metrics are invalid in {checkpoint_path}.")
    metrics: list[OmniQuantEpochMetric] = []
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, Mapping):
            raise TypeError(f"Checkpoint epoch metric is invalid in {checkpoint_path}.")

        def optional_float(name: str) -> float | None:
            value = raw_metric.get(name)
            return None if value is None else float(value)

        raw_validation_slots = raw_metric.get(
            "validation_lfq_slot_losses", ()
        )
        validation_slot_items = (
            raw_validation_slots.items()
            if isinstance(raw_validation_slots, Mapping)
            else raw_validation_slots
        )

        metrics.append(
            OmniQuantEpochMetric(
                epoch=int(raw_metric["epoch"]),
                train_loss=optional_float("train_loss"),
                mean_grad_norm=optional_float("mean_grad_norm"),
                max_grad_norm=optional_float("max_grad_norm"),
                eval_loss=optional_float("eval_loss"),
                eval_mse_loss=optional_float("eval_mse_loss"),
                validation_loss=optional_float("validation_loss"),
                validation_lfq_slot_losses=tuple(
                    (str(slot), float(loss))
                    for slot, loss in validation_slot_items
                ),
            )
        )
    return tuple(metrics)


def _epoch_metric_state(metric: OmniQuantEpochMetric) -> dict[str, Any]:
    return {
        "epoch": metric.epoch,
        "train_loss": metric.train_loss,
        "mean_grad_norm": metric.mean_grad_norm,
        "max_grad_norm": metric.max_grad_norm,
        "eval_loss": metric.eval_loss,
        "eval_mse_loss": metric.eval_mse_loss,
        "validation_loss": metric.validation_loss,
        "validation_lfq_slot_losses": dict(
            metric.validation_lfq_slot_losses
        ),
    }


def _validate_omniquant_checkpoint(
    state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    config: OmniQuantConfig,
    expected_objective: OmniQuantObjective,
    require_run_objective: bool,
) -> None:
    saved_config = state.get("config")
    if not isinstance(saved_config, Mapping):
        raise TypeError(f"Checkpoint config is missing or invalid in {checkpoint_path}.")
    if bool(saved_config.get("use_lac", False)):
        raise ValueError(f"LAC checkpoints are no longer supported: {checkpoint_path}")
    expected_config: dict[str, Any] = {
        "weight_quant_format": config.weight_quant_format,
        "activation_quant_format": config.activation_quant_format,
        "weight_quant_scheme": config.weight_quant_scheme,
        "use_lwc": config.use_lwc,
        "use_let": config.use_let,
        "learn_let": config.learn_let,
        "calibration_forward_mode": OMNIQUANT_CALIBRATION_FORWARD_MODE,
        "calibration_compute_dtype": OMNIQUANT_CALIBRATION_COMPUTE_DTYPE,
        "quantization_compute_dtype": OMNIQUANT_QUANTIZATION_COMPUTE_DTYPE,
        "loss_compute_dtype": OMNIQUANT_LOSS_COMPUTE_DTYPE,
    }
    # LET initialization does not affect a no-LET checkpoint. Legacy
    # SmoothQuant-initialized checkpoints predate this explicit field.
    if config.use_let:
        expected_config["let_init"] = config.let_init

        if config.let_init == "smoothquant":
            expected_config["smoothquant_alpha"] = config.smoothquant_alpha
    if require_run_objective:
        expected_config["final_objective"] = config.final_objective
        if config.final_objective == "lfq_ce":
            expected_config.update(
                {
                    "lfq_token_scope": config.lfq_token_scope,
                    "lfq_vocab_scope": config.lfq_vocab_scope,
                    "lfq_slot_weights": tuple(config.lfq_slot_weights),
                    "lfq_loss_weight": config.lfq_loss_weight,
                }
            )
    legacy_defaults = {
        "let_init": "smoothquant",
        "smoothquant_alpha": DEFAULT_SMOOTHQUANT_ALPHA,
        "final_objective": "mse",
        "lfq_loss_weight": 1.0,
    }
    for key, expected in expected_config.items():
        saved_value = saved_config.get(key, legacy_defaults.get(key))
        if saved_value != expected:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has {key}={saved_value!r}, "
                f"but the requested configuration requires {expected!r}."
            )
    saved_objective = state.get("objective", "mse")
    if saved_objective != expected_objective:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has objective={saved_objective!r}, "
            f"but this layer requires {expected_objective!r}."
        )


def _restore_train_block_parameters(
    train_block: _TrainableOmniBlock,
    state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    layer_idx: int,
) -> None:
    saved_lwc = state.get("lwc_parameters")
    if not isinstance(saved_lwc, Mapping):
        raise TypeError(f"Checkpoint LWC parameters are missing or invalid in {checkpoint_path}.")
    wrappers = {
        name: child
        for name, child in train_block.block.named_modules()
        if isinstance(child, _TrainableSymmetricLinear)
    }
    if set(saved_lwc) != set(wrappers):
        raise ValueError(f"Checkpoint LWC modules do not match layer {layer_idx}.")
    saved_let = state.get("let_log_scales")
    if not isinstance(saved_let, Mapping) or set(saved_let) != set(train_block.let_parameters):
        raise ValueError(f"Checkpoint LET parameters do not match layer {layer_idx}.")

    with torch.no_grad():
        for name, wrapper in wrappers.items():
            saved_wrapper = saved_lwc[name]
            if not isinstance(saved_wrapper, Mapping):
                raise TypeError(f"Invalid LWC state for {name!r} in {checkpoint_path}.")
            current_state = wrapper.lwc_state()
            if set(saved_wrapper) != set(current_state):
                raise ValueError(
                    f"Checkpoint LWC tensors do not match {name!r} in layer {layer_idx}."
                )
            for parameter_name, current in current_state.items():
                saved = saved_wrapper[parameter_name]
                if not torch.is_tensor(saved) or tuple(saved.shape) != tuple(current.shape):
                    raise ValueError(
                        f"Checkpoint tensor shape mismatch for "
                        f"{name}.{parameter_name} in layer {layer_idx}."
                    )
                parameter = getattr(wrapper, parameter_name)
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        f"Missing trainable parameter "
                        f"{name}.{parameter_name} in layer {layer_idx}."
                    )
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
        saved_config = state.get("config")
        legacy_let_bounds: tuple[float, float] | None = None
        if isinstance(saved_config, Mapping):
            legacy_min = saved_config.get("min_let_scale")
            legacy_max = saved_config.get("max_let_scale")
            if (legacy_min is None) != (legacy_max is None):
                raise ValueError(
                    f"Incomplete legacy LET scale bounds in {checkpoint_path}."
                )
            if legacy_min is not None and legacy_max is not None:
                legacy_min = float(legacy_min)
                legacy_max = float(legacy_max)
                if legacy_min <= 0.0 or legacy_max <= legacy_min:
                    raise ValueError(
                        f"Invalid legacy LET scale bounds in {checkpoint_path}."
                    )
                legacy_let_bounds = (math.log(legacy_min), math.log(legacy_max))
        for name, parameter in train_block.let_parameters.items():
            saved = saved_let[name]
            if not torch.is_tensor(saved) or tuple(saved.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Checkpoint LET tensor shape mismatch for {name!r} in layer {layer_idx}."
                )
            if legacy_let_bounds is not None:
                saved = saved.clamp(min=legacy_let_bounds[0], max=legacy_let_bounds[1])
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def apply_omniquant_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    config: OmniQuantConfig,
    capture_layer_input_batches: Any,
    act_quant_mode: str = "per_linear",
    checkpoint_dir: str | Path | None = None,
    prefix_checkpoint_dir: str | Path | None = None,
    lfq_token_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[int, OmniQuantSummary]:
    """Calibrate and replace selected Qwen3 blocks sequentially.

    ``capture_layer_input_batches`` is injected by the runner to avoid a
    circular import.  Captured and propagated streams are moved to CPU between
    blocks, which keeps variable-length OneRec calibration prompts from
    occupying the entire GPU.
    """
    config.validate()
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.layers
    final_layer_idx = len(layers) - 1
    selected_layer_indices = sorted(layer_indices)
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    prefix_checkpoint_root = (
        None if prefix_checkpoint_dir is None else Path(prefix_checkpoint_dir)
    )
    if prefix_checkpoint_root is not None:
        if not prefix_checkpoint_root.is_dir():
            raise FileNotFoundError(
                "OmniQuant prefix checkpoint directory does not exist: "
                f"{prefix_checkpoint_root}"
            )
        if selected_layer_indices != list(range(len(layers))):
            raise ValueError(
                "omni_prefix_checkpoint_dir requires all Transformer layers so the "
                "FP and quantized streams remain aligned through the final block."
            )
        if checkpoint_root is not None and checkpoint_root.resolve() == prefix_checkpoint_root.resolve():
            raise ValueError("Prefix and output checkpoint directories must be different.")
        print(
            f"[omniquant] prefix checkpoint enabled layers=0-{final_layer_idx - 1} "
            f"source={prefix_checkpoint_root}"
        )
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    lfq_projector: _LFQOutputProjector | None = None
    if config.final_objective == "lfq_ce":
        if final_layer_idx not in selected_layer_indices:
            raise ValueError(
                f"omni_final_objective='lfq_ce' requires the actual final "
                f"Transformer layer ({final_layer_idx}) to be included in --layers."
            )
        backbone = getattr(model, "model", None)
        final_norm = getattr(backbone, "norm", None)
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        output_head = get_output_embeddings() if callable(get_output_embeddings) else None
        if not isinstance(final_norm, nn.Module) or not isinstance(output_head, nn.Module):
            raise TypeError("LFQ requires model.model.norm and model.get_output_embeddings().")
        if lfq_token_ids is None or set(lfq_token_ids) != set(LFQ_SLOT_NAMES):
            raise ValueError(
                "SID-slot LFQ requires token IDs for the SID_a, SID_b, and "
                "SID_c vocabularies."
            )
        lfq_projector = _LFQOutputProjector(
            final_norm=final_norm,
            output_head=output_head,
            token_ids=lfq_token_ids,
        )
        token_counts = ",".join(
            f"{slot}:{len(lfq_token_ids[slot])}" for slot in LFQ_SLOT_NAMES
        )
        normalized_weights = _normalized_lfq_slot_weights(config)
        weight_text = ",".join(
            f"{slot}:{normalized_weights[slot]:.6g}" for slot in LFQ_SLOT_NAMES
        )
        print(
            f"[omniquant] LFQ enabled final_layer={final_layer_idx} "
            f"token_scope={config.lfq_token_scope} "
            f"vocab_scope={config.lfq_vocab_scope} tokens={token_counts} "
            f"weights={weight_text} lfq_weight={config.lfq_loss_weight:.6g}"
        )
    summaries: dict[int, OmniQuantSummary] = {}
    fp_inputs: list[Batch] | None = None
    quant_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    for layer_idx in selected_layer_indices:
        if fp_inputs is None:
            captured = capture_layer_input_batches(model=model, layer=layers[layer_idx], model_batches=model_batches)
            fp_inputs = [_tree_cpu(batch) for batch in captured]
            quant_inputs = [_tree_cpu(batch) for batch in captured]
            stream_layer_idx = layer_idx
        else:
            assert quant_inputs is not None and stream_layer_idx is not None
            while stream_layer_idx < layer_idx:
                fp_inputs = _advance_cpu(layers[stream_layer_idx], fp_inputs)
                quant_inputs = _advance_cpu(layers[stream_layer_idx], quant_inputs)
                stream_layer_idx += 1

        assert fp_inputs is not None and quant_inputs is not None
        teacher_block = layers[layer_idx]
        objective = _layer_objective(
            config,
            layer_idx=layer_idx,
            final_layer_idx=final_layer_idx,
        )
        layer_config = config
        source_checkpoint: Path | None = None
        initial_lfq_slot_losses: dict[str, float] = {}
        final_lfq_slot_losses: dict[str, float] = {}
        initial_mse_loss: float | None = None
        final_mse_loss: float | None = None
        best_epoch: int | None = None
        epoch_metrics: tuple[OmniQuantEpochMetric, ...] = ()
        if prefix_checkpoint_root is not None and layer_idx < final_layer_idx:
            source_checkpoint = prefix_checkpoint_root / f"layer_{layer_idx:02d}.pt"
            state = _load_omniquant_checkpoint(source_checkpoint, layer_idx=layer_idx)
            _validate_omniquant_checkpoint(
                state,
                checkpoint_path=source_checkpoint,
                config=layer_config,
                expected_objective="mse",
                require_run_objective=False,
            )
            train_block = _TrainableOmniBlock(
                copy.deepcopy(teacher_block),
                config=layer_config,
                init_scales={},
            )
            _restore_train_block_parameters(
                train_block,
                state,
                checkpoint_path=source_checkpoint,
                layer_idx=layer_idx,
            )
            initial_loss = float(state["initial_loss"])
            final_loss = float(state["final_loss"])
            initial_mse_loss = float(state.get("initial_mse_loss", initial_loss))
            final_mse_loss = float(state.get("final_mse_loss", final_loss))
            best_epoch_value = state.get("best_epoch")
            best_epoch = (
                None if best_epoch_value is None else int(best_epoch_value)
            )
            epoch_metrics = _checkpoint_epoch_metrics(state, checkpoint_path=source_checkpoint)
        else:
            init_scales = (
                collect_smoothquant_scales(
                    teacher_block,
                    fp_inputs,
                    alpha=layer_config.smoothquant_alpha,
                )
                if layer_config.use_let and layer_config.let_init == "smoothquant"
                else {}
            )
            train_block = _TrainableOmniBlock(
                copy.deepcopy(teacher_block),
                config=layer_config,
                init_scales=init_scales,
            )
            (
                initial_loss,
                final_loss,
                initial_lfq_slot_losses,
                final_lfq_slot_losses,
                initial_mse_loss,
                final_mse_loss,
                best_epoch,
                epoch_metrics,
            ) = _train_block(
                teacher_block=teacher_block,
                train_block=train_block,
                fp_inputs=fp_inputs,
                quant_inputs=quant_inputs,
                config=layer_config,
                lfq_projector=lfq_projector if layer_idx == final_layer_idx else None,
                layer_idx=layer_idx,
            )
        next_fp_inputs = _advance_cpu(teacher_block, fp_inputs)
        final_block, replaced = _finalize_block(train_block)
        shared_attention_modules = 0
        shared_mlp_modules = 0
        if (
            layer_config.activation_quant_format != "none"
            and act_quant_mode == "shared_input"
        ):
            shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(final_block)
        layers[layer_idx] = final_block
        quant_inputs = _advance_cpu(final_block, quant_inputs)
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        summaries[layer_idx] = OmniQuantSummary(
            replaced_linears=replaced,
            initial_loss=initial_loss,
            final_loss=final_loss,
            let_scales=tuple(train_block.let_parameters.keys()),
            objective=objective,
            initial_lfq_slot_losses=tuple(
                (slot, initial_lfq_slot_losses[slot])
                for slot in LFQ_SLOT_NAMES
                if slot in initial_lfq_slot_losses
            ),
            final_lfq_slot_losses=tuple(
                (slot, final_lfq_slot_losses[slot])
                for slot in LFQ_SLOT_NAMES
                if slot in final_lfq_slot_losses
            ),
            lfq_slot_weights=(
                tuple(config.lfq_slot_weights) if objective == "lfq_ce" else None
            ),
            lfq_loss_weight=(
                config.lfq_loss_weight if objective != "mse" else 1.0
            ),
            initial_mse_loss=initial_mse_loss,
            final_mse_loss=final_mse_loss,
            best_epoch=best_epoch,
            epoch_metrics=epoch_metrics,
            shared_attention_modules=shared_attention_modules,
            shared_mlp_modules=shared_mlp_modules,
        )
        if checkpoint_root is not None:
            learned_state = {
                "layer_idx": layer_idx,
                "config": {
                    "weight_quant_format": config.weight_quant_format,
                    "activation_quant_format": config.activation_quant_format,
                    "weight_quant_scheme": config.weight_quant_scheme,
                    "use_lwc": config.use_lwc,
                    "use_let": config.use_let,
                    "learn_let": config.learn_let,
                    "let_init": config.let_init,
                    "let_scale_parameterization": "unbounded_log",
                    "calibration_forward_mode": OMNIQUANT_CALIBRATION_FORWARD_MODE,
                    "calibration_compute_dtype": OMNIQUANT_CALIBRATION_COMPUTE_DTYPE,
                    "quantization_compute_dtype": OMNIQUANT_QUANTIZATION_COMPUTE_DTYPE,
                    "loss_compute_dtype": OMNIQUANT_LOSS_COMPUTE_DTYPE,
                    "smoothquant_alpha": config.smoothquant_alpha,
                    "final_objective": config.final_objective,
                    "lfq_token_scope": config.lfq_token_scope,
                    "lfq_vocab_scope": config.lfq_vocab_scope,
                    "lfq_slot_weights": tuple(config.lfq_slot_weights),
                    "lfq_loss_weight": config.lfq_loss_weight,
                    "epochs": config.epochs,
                    "validation_sample_size": config.validation_sample_size,
                    "train_sample_size": config.train_sample_size,
                    "epoch_eval_interval": config.epoch_eval_interval,
                    "lwc_lr": config.lwc_lr,
                    "let_lr": config.let_lr,
                    "weight_decay": config.weight_decay,
                    "max_grad_norm": config.max_grad_norm,
                },
                "lwc_parameters": {
                    name: child.lwc_state()
                    for name, child in train_block.block.named_modules()
                    if isinstance(child, _TrainableSymmetricLinear)
                },
                "let_log_scales": {
                    name: parameter.detach().cpu()
                    for name, parameter in train_block.let_parameters.items()
                },
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "initial_lfq_slot_losses": dict(initial_lfq_slot_losses),
                "final_lfq_slot_losses": dict(final_lfq_slot_losses),
                "initial_mse_loss": initial_mse_loss,
                "final_mse_loss": final_mse_loss,
                "best_epoch": best_epoch,
                "epoch_metrics": [_epoch_metric_state(metric) for metric in epoch_metrics],
                "objective": objective,
            }
            if source_checkpoint is not None:
                learned_state["source_checkpoint"] = str(source_checkpoint)
            torch.save(learned_state, checkpoint_root / f"layer_{layer_idx:02d}.pt")
        lfq_loss_text = ""
        if initial_lfq_slot_losses:
            lfq_loss_text = "slot_loss=" + ",".join(
                f"{slot}:{initial_lfq_slot_losses[slot]:.6e}->{final_lfq_slot_losses[slot]:.6e}"
                for slot in LFQ_SLOT_NAMES
            ) + f" lfq_weight={config.lfq_loss_weight:.6g} "
        mse_loss_text = ""
        if (
            objective != "mse"
            and initial_mse_loss is not None
            and final_mse_loss is not None
        ):
            mse_loss_text = (
                f"mse={initial_mse_loss:.6e}->{final_mse_loss:.6e} "
            )
        print(
            f"[omniquant] layer={layer_idx} replaced_linears={replaced} "
            f"objective={objective} loss={initial_loss:.6e}->{final_loss:.6e} "
            f"{lfq_loss_text}"
            f"{mse_loss_text}"
            f"source={'prefix_checkpoint' if source_checkpoint is not None else 'optimized'} "
            f"let_mode={'learned' if layer_config.learn_let else ('fixed' if layer_config.use_let else 'none')} "
            f"let={','.join(train_block.let_parameters.keys()) or 'off'}"
        )
    return summaries


def restore_omniquant_layers_from_checkpoints(
    *,
    model: nn.Module,
    layer_indices: Sequence[int],
    config: OmniQuantConfig,
    checkpoint_dir: str | Path,
    act_quant_mode: str = "per_linear",
) -> dict[int, OmniQuantSummary]:
    """Restore statically quantized OmniQuant blocks from saved calibration state.

    Checkpoints store the learned LWC logits and LET log-scales rather than
    quantized weights. This function reconstructs each static QDQ block from
    the original FP model, verifies its compatible checkpoint configuration,
    and installs it without running calibration.
    """

    config.validate()
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"OmniQuant checkpoint directory does not exist: {checkpoint_root}")
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.layers
    summaries: dict[int, OmniQuantSummary] = {}

    for layer_idx in sorted(layer_indices):
        checkpoint_path = checkpoint_root / f"layer_{layer_idx:02d}.pt"
        state = _load_omniquant_checkpoint(checkpoint_path, layer_idx=layer_idx)
        expected_objective = _layer_objective(
            config,
            layer_idx=layer_idx,
            final_layer_idx=len(layers) - 1,
        )
        layer_config = config
        _validate_omniquant_checkpoint(
            state,
            checkpoint_path=checkpoint_path,
            config=layer_config,
            expected_objective=expected_objective,
            require_run_objective=True,
        )

        train_block = _TrainableOmniBlock(
            copy.deepcopy(layers[layer_idx]),
            config=layer_config,
            init_scales={},
        )
        _restore_train_block_parameters(
            train_block,
            state,
            checkpoint_path=checkpoint_path,
            layer_idx=layer_idx,
        )

        final_block, replaced = _finalize_block(train_block)
        shared_attention_modules = 0
        shared_mlp_modules = 0
        if (
            layer_config.activation_quant_format != "none"
            and act_quant_mode == "shared_input"
        ):
            shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(final_block)
        layers[layer_idx] = final_block
        initial_lfq_slot_losses = state.get("initial_lfq_slot_losses", {})
        final_lfq_slot_losses = state.get("final_lfq_slot_losses", {})
        if not isinstance(initial_lfq_slot_losses, Mapping) or not isinstance(
            final_lfq_slot_losses, Mapping
        ):
            raise TypeError(f"Checkpoint LFQ slot losses are invalid in {checkpoint_path}.")
        best_epoch_value = state.get("best_epoch")
        best_epoch = None if best_epoch_value is None else int(best_epoch_value)
        epoch_metrics = _checkpoint_epoch_metrics(
            state, checkpoint_path=checkpoint_path
        )
        summaries[layer_idx] = OmniQuantSummary(
            replaced_linears=replaced,
            initial_loss=float(state["initial_loss"]),
            final_loss=float(state["final_loss"]),
            let_scales=tuple(train_block.let_parameters.keys()),
            objective=expected_objective,
            initial_lfq_slot_losses=tuple(
                (slot, float(initial_lfq_slot_losses[slot]))
                for slot in LFQ_SLOT_NAMES
                if slot in initial_lfq_slot_losses
            ),
            final_lfq_slot_losses=tuple(
                (slot, float(final_lfq_slot_losses[slot]))
                for slot in LFQ_SLOT_NAMES
                if slot in final_lfq_slot_losses
            ),
            lfq_slot_weights=(
                tuple(config.lfq_slot_weights)
                if expected_objective == "lfq_ce"
                else None
            ),
            lfq_loss_weight=(
                config.lfq_loss_weight if expected_objective != "mse" else 1.0
            ),
            initial_mse_loss=(
                float(state["initial_mse_loss"])
                if state.get("initial_mse_loss") is not None
                else (
                    float(state["initial_loss"])
                    if expected_objective == "mse"
                    else None
                )
            ),
            final_mse_loss=(
                float(state["final_mse_loss"])
                if state.get("final_mse_loss") is not None
                else (
                    float(state["final_loss"])
                    if expected_objective == "mse"
                    else None
                )
            ),
            best_epoch=best_epoch,
            epoch_metrics=epoch_metrics,
            shared_attention_modules=shared_attention_modules,
            shared_mlp_modules=shared_mlp_modules,
        )
        del train_block
        print(f"[omniquant] restored layer={layer_idx} replaced_linears={replaced}")

    return summaries
