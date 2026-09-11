"""Official-topology FlatQuant fake-QDQ calibration for Qwen3 blocks.

The implementation jointly optimizes SVD/Cayley transforms, three diagonal
scales, per-output-channel LWC, and shared-site LAC with cumulative block MSE.
It uses the repository deployment-matched finalization and per-layer checkpoint
framework. KV-cache quantization and fused real-quant kernels are out of scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..apply import install_shared_input_activation_quantization
from ..omniquant.runtime import (
    LFQ_SLOT_NAMES,
    _LFQBoundaryTarget,
    _LFQOutputProjector,
    _TrainableSymmetricLinear,
    _advance_cpu,
    _build_lfq_boundary_targets,
    _first_tensor,
    _floating_tree_to_dtype,
    _fp4_e2m1_ste,
    _fp8_cast_ste,
    _lfq_boundary_loss,
    _lfq_soft_cross_entropy_from_logits,
    _round_ste,
    _signed_qmax,
    _total_grad_norm,
    _trainable_activation_qdq,
    _tree_cpu,
)
from ..quant import (
    ActQuant,
    ActQuantMode,
    FP4_E2M1_MAX,
    FP8_MAX,
    QuantFormat,
    WeightQuantScheme,
    activation_per_token_qdq_by_format,
    fp4_e2m1_quantize,
    normalize_weight_group_size,
    quant_format_qmax,
    resolve_weight_quant_scheme,
    validate_quant_format,
)
from ..support.runtime_utils import _module_device, _move_tree_to_device
from ..support.smoothquant_runtime import (
    Batch,
    DEFAULT_SMOOTHQUANT_ALPHA,
    _batch_to_args_kwargs,
    collect_smoothquant_scales,
)
from .transforms import (
    FixedSmoothQuantTransform,
    FlatTransformInit,
    FlatTransformKind,
    FrozenKroneckerTransform,
    FrozenSingleTransform,
    KroneckerSVDTransform,
    SingleSVDTransform,
    kronecker_matmul,
)


DEFAULT_FLATQUANT_EPOCHS = 15
DEFAULT_FLATQUANT_TRANSFORM_LR = 5e-3
DEFAULT_FLATQUANT_LWC_LR = 5e-2
DEFAULT_FLATQUANT_LAC_LR = 5e-2
DEFAULT_FLATQUANT_WEIGHT_DECAY = 0.01
DEFAULT_FLATQUANT_INIT_LWC_LOGIT = 4.0
DEFAULT_FLATQUANT_INIT_LAC_LOGIT = 4.0
DEFAULT_FLATQUANT_MIN_LR_FACTOR = 1e-3

FLATQUANT_TRANSFORM_NAMES = ("attn_in", "head", "head_dim", "mlp_in", "down")
FLATQUANT_LINEAR_ROLES = {
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
}
FLATQUANT_SMOOTHQUANT_SCALE_SOURCES = {
    "qkv": "self_attn.q_proj",
    "o": "self_attn.o_proj",
    "mlp": "mlp.gate_proj",
    "down": "mlp.down_proj",
}
TrainableFlatQuantTransform = KroneckerSVDTransform | SingleSVDTransform
FrozenFlatQuantTransform = FrozenKroneckerTransform | FrozenSingleTransform
FlatQuantTransform = (
    TrainableFlatQuantTransform
    | FrozenFlatQuantTransform
    | FixedSmoothQuantTransform
)


@dataclass(frozen=True)
class FlatQuantCoreConfig:
    weight_quant_format: QuantFormat = "fp4_e2m1"
    activation_quant_format: QuantFormat = "fp8_e4m3fn"
    weight_quant_scheme: WeightQuantScheme = "symmetric"
    weight_group_size: int = 0
    use_lwc: bool = True
    use_lac: bool = True
    learn_lac: bool = True
    learn_transform: bool = True
    transform_kind: FlatTransformKind = "kronecker"
    transform_init: FlatTransformInit = "random_orthogonal"
    smoothquant_alpha: float = DEFAULT_SMOOTHQUANT_ALPHA
    diag_alpha: float = 0.5
    final_objective: Literal["mse", "lfq_ce"] = "mse"
    lfq_token_scope: Literal["sid_slots"] = "sid_slots"
    lfq_vocab_scope: Literal["s_abc"] = "s_abc"
    lfq_slot_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lfq_loss_weight: float = 1.0
    lfq_boundary_loss_weight: float = 0.0
    lfq_boundary_topk: int = 32
    lfq_boundary_negative_count: int = 32
    lfq_boundary_tie_threshold: float = 1e-2
    lfq_boundary_gap_scale: float = 1.0
    epochs: int = DEFAULT_FLATQUANT_EPOCHS
    train_sample_size: int = 0
    validation_sample_size: int = 0
    epoch_eval_interval: int = 0
    transform_lr: float = DEFAULT_FLATQUANT_TRANSFORM_LR
    lwc_lr: float = DEFAULT_FLATQUANT_LWC_LR
    lac_lr: float = DEFAULT_FLATQUANT_LAC_LR
    weight_decay: float = DEFAULT_FLATQUANT_WEIGHT_DECAY
    init_lwc_logit: float = DEFAULT_FLATQUANT_INIT_LWC_LOGIT
    init_lac_logit: float = DEFAULT_FLATQUANT_INIT_LAC_LOGIT
    min_lr_factor: float = DEFAULT_FLATQUANT_MIN_LR_FACTOR
    normalize_mse_gradient: bool = True
    max_grad_norm: float | None = None
    eps: float = 1e-12

    def validate(self) -> None:
        weight_format = validate_quant_format(self.weight_quant_format)
        activation_format = validate_quant_format(self.activation_quant_format)
        if weight_format not in ("fp8_e4m3fn", "fp4_e2m1", "int4", "int6", "int8"):
            raise ValueError("FlatQuant-core requires a supported quantized weight format.")
        if activation_format == "none":
            raise ValueError("FlatQuant-core currently requires activation quantization.")
        resolved_scheme = resolve_weight_quant_scheme(
            weight_format,
            self.weight_quant_scheme,
        )
        if resolved_scheme != self.weight_quant_scheme:
            raise ValueError(
                f"Invalid weight scheme {self.weight_quant_scheme!r} for {weight_format}."
            )
        normalize_weight_group_size(self.weight_group_size)
        if self.transform_kind not in ("kronecker", "smoothquant"):
            raise ValueError(f"Unsupported transform_kind: {self.transform_kind!r}")
        if self.transform_init not in ("identity", "random_orthogonal"):
            raise ValueError(f"Unsupported transform_init: {self.transform_init!r}")
        if self.transform_kind == "smoothquant" and self.learn_transform:
            raise ValueError(
                "The SmoothQuant control is fixed; set learn_transform=False."
            )
        if (
            not math.isfinite(self.smoothquant_alpha)
            or not 0.0 <= self.smoothquant_alpha <= 1.0
        ):
            raise ValueError("smoothquant_alpha must be finite and in [0, 1].")
        if not math.isfinite(self.diag_alpha) or not 0.0 <= self.diag_alpha <= 1.0:
            raise ValueError("diag_alpha must be finite and in [0, 1].")
        if self.final_objective not in ("mse", "lfq_ce"):
            raise ValueError("final_objective must be 'mse' or 'lfq_ce'.")
        if (self.lfq_token_scope, self.lfq_vocab_scope) != ("sid_slots", "s_abc"):
            raise ValueError(
                "FlatQuant LFQ requires sid_slots token scope and s_abc vocabulary scope."
            )
        if len(self.lfq_slot_weights) != len(LFQ_SLOT_NAMES):
            raise ValueError("lfq_slot_weights must contain A, B, and C weights.")
        if any(
            not math.isfinite(weight) or weight < 0.0
            for weight in self.lfq_slot_weights
        ) or sum(self.lfq_slot_weights) <= 0.0:
            raise ValueError("lfq_slot_weights must be finite, non-negative, and not all zero.")
        if not math.isfinite(self.lfq_loss_weight) or self.lfq_loss_weight < 0.0:
            raise ValueError("lfq_loss_weight must be finite and non-negative.")
        if (
            not math.isfinite(self.lfq_boundary_loss_weight)
            or self.lfq_boundary_loss_weight < 0.0
        ):
            raise ValueError("lfq_boundary_loss_weight must be finite and non-negative.")
        if (
            self.final_objective == "lfq_ce"
            and self.lfq_loss_weight == 0.0
            and self.lfq_boundary_loss_weight == 0.0
        ):
            raise ValueError("LFQ requires a positive CE or boundary loss weight.")
        if self.lfq_boundary_topk <= 0 or self.lfq_boundary_negative_count <= 0:
            raise ValueError("LFQ boundary top-k and negative count must be positive.")
        if (
            not math.isfinite(self.lfq_boundary_tie_threshold)
            or self.lfq_boundary_tie_threshold < 0.0
        ):
            raise ValueError("LFQ boundary tie threshold must be finite and non-negative.")
        if (
            not math.isfinite(self.lfq_boundary_gap_scale)
            or self.lfq_boundary_gap_scale <= 0.0
        ):
            raise ValueError("LFQ boundary gap scale must be finite and positive.")
        if self.epochs <= 0:
            raise ValueError("FlatQuant-core epochs must be positive.")
        if self.train_sample_size < 0:
            raise ValueError("train_sample_size must be non-negative.")
        if self.validation_sample_size < 0:
            raise ValueError("validation_sample_size must be non-negative.")
        if self.epoch_eval_interval < 0:
            raise ValueError("epoch_eval_interval must be non-negative.")
        for name, value in (
            ("transform_lr", self.transform_lr),
            ("lwc_lr", self.lwc_lr),
            ("lac_lr", self.lac_lr),
            ("min_lr_factor", self.min_lr_factor),
            ("eps", self.eps),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.min_lr_factor > 1.0:
            raise ValueError("min_lr_factor must be at most one.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be None or finite and positive.")


@dataclass(frozen=True)
class FlatQuantEpochMetric:
    epoch: int
    train_mse: float | None
    eval_mse: float | None
    mean_grad_norm: float | None
    max_grad_norm: float | None
    transform_lr: float | None
    lwc_lr: float | None
    lac_lr: float | None
    validation_mse: float | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    validation_loss: float | None = None
    validation_lfq_slot_losses: tuple[tuple[str, float], ...] = ()
    train_lfq_boundary_loss: float | None = None
    validation_lfq_boundary_loss: float | None = None


@dataclass(frozen=True)
class FlatQuantCoreSummary:
    replaced_linears: int
    initial_mse_loss: float
    final_mse_loss: float
    transform_factors: tuple[tuple[str, int, int], ...]
    effective_transform_parameters: int
    trainable_transform_parameters: int
    trainable_lwc_parameters: int
    trainable_lac_parameters: int
    epoch_metrics: tuple[FlatQuantEpochMetric, ...]
    initial_validation_mse_loss: float | None = None
    final_validation_mse_loss: float | None = None
    best_epoch: int | None = None
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0
    objective: Literal["mse", "lfq_ce"] = "mse"
    initial_loss: float | None = None
    final_loss: float | None = None
    initial_lfq_slot_losses: tuple[tuple[str, float], ...] = ()
    final_lfq_slot_losses: tuple[tuple[str, float], ...] = ()
    lfq_slot_weights: tuple[float, float, float] | None = None
    lfq_loss_weight: float = 1.0
    lfq_boundary_loss_weight: float = 0.0
    initial_lfq_boundary_loss: float | None = None
    final_lfq_boundary_loss: float | None = None
    initial_lfq_boundary_slot_losses: tuple[tuple[str, float], ...] = ()
    final_lfq_boundary_slot_losses: tuple[tuple[str, float], ...] = ()
    initial_validation_loss: float | None = None
    final_validation_loss: float | None = None
    initial_validation_lfq_slot_losses: tuple[tuple[str, float], ...] = ()
    final_validation_lfq_slot_losses: tuple[tuple[str, float], ...] = ()


def _formal_transform_bank(transform_bank: nn.ModuleDict) -> bool:
    return "attn_in" in transform_bank


def _require_kronecker_transform(
    transform_bank: nn.ModuleDict,
    name: str,
) -> KroneckerSVDTransform | FrozenKroneckerTransform:
    transform = transform_bank[name]
    if not isinstance(
        transform,
        (KroneckerSVDTransform, FrozenKroneckerTransform),
    ):
        raise TypeError(
            f"FlatQuant transform {name!r} must be Kronecker, got {type(transform)!r}."
        )
    return transform


def _require_single_transform(
    transform_bank: nn.ModuleDict,
    name: str,
) -> SingleSVDTransform | FrozenSingleTransform:
    transform = transform_bank[name]
    if not isinstance(
        transform,
        (SingleSVDTransform, FrozenSingleTransform),
    ):
        raise TypeError(
            f"FlatQuant transform {name!r} must be single-matrix, got {type(transform)!r}."
        )
    return transform


def _transform_matrix_only(
    transform: KroneckerSVDTransform | FrozenKroneckerTransform,
    x: torch.Tensor,
    *,
    inverse_transpose: bool = False,
) -> torch.Tensor:
    return transform(
        x,
        inverse_transpose=inverse_transpose,
        include_diag=False,
    )


def _apply_head_axis_transform(
    x: torch.Tensor,
    transform: SingleSVDTransform | FrozenSingleTransform,
    *,
    head_dim: int,
    inverse_transpose: bool = False,
) -> torch.Tensor:
    expected = transform.size * head_dim
    if x.shape[-1] != expected:
        raise ValueError(
            f"FlatQuant head transform expected last dimension {expected}, "
            f"got {x.shape[-1]}."
        )
    matrix = transform.matrix(inverse_transpose=inverse_transpose).to(
        device=x.device,
        dtype=x.dtype,
    )
    original_shape = x.shape
    viewed = x.reshape(-1, transform.size, head_dim)
    return torch.matmul(matrix.T, viewed).reshape(original_shape)


def _legacy_transform_name(role: str) -> str:
    if role in ("q", "k", "v"):
        return "qkv"
    if role in ("gate", "up"):
        return "mlp"
    if role in ("o", "down"):
        return role
    raise ValueError(f"Unknown FlatQuant linear role: {role!r}")


def _official_transformed_weight(
    weight: torch.Tensor,
    *,
    role: str,
    transform_bank: nn.ModuleDict,
    head_dim: int,
) -> torch.Tensor:
    if not _formal_transform_bank(transform_bank):
        transform = transform_bank[_legacy_transform_name(role)]
        if not isinstance(transform, (KroneckerSVDTransform, FixedSmoothQuantTransform)):
            raise TypeError(f"Unexpected legacy FlatQuant transform: {type(transform)!r}")
        return transform.transform_weight(weight)

    if role in ("q", "k", "v"):
        attn_in = _require_kronecker_transform(transform_bank, "attn_in")
        transformed = attn_in.transform_weight(weight)
        if role == "v":
            head_dim_transform = _require_single_transform(
                transform_bank,
                "head_dim",
            )
            transformed = head_dim_transform(transformed.T).T
        return transformed

    if role == "o":
        head = _require_single_transform(transform_bank, "head")
        head_dim_transform = _require_single_transform(
            transform_bank,
            "head_dim",
        )
        return kronecker_matmul(
            weight,
            head.matrix(inverse_transpose=True).to(weight),
            head_dim_transform.matrix(inverse_transpose=True).to(weight),
        )

    if role in ("gate", "up"):
        mlp_in = _require_kronecker_transform(transform_bank, "mlp_in")
        transformed = mlp_in.transform_weight(weight)
        if role == "up":
            down = _require_kronecker_transform(transform_bank, "down")
            if down.diag_scale is None:
                raise ValueError("Official FlatQuant down transform requires a diagonal.")
            transformed = transformed * down.diag_scale.to(
                device=transformed.device,
                dtype=transformed.dtype,
            ).view(-1, 1)
        return transformed

    if role == "down":
        down = _require_kronecker_transform(transform_bank, "down")
        return down.transform_weight(weight)

    raise ValueError(f"Unknown FlatQuant linear role: {role!r}")


def _official_transformed_bias(
    bias: torch.Tensor | None,
    *,
    role: str,
    transform_bank: nn.ModuleDict,
) -> torch.Tensor | None:
    if bias is None or not _formal_transform_bank(transform_bank):
        return bias
    if role == "v":
        return _require_single_transform(transform_bank, "head_dim")(bias)
    if role == "up":
        down = _require_kronecker_transform(transform_bank, "down")
        if down.diag_scale is None:
            raise ValueError("Official FlatQuant down transform requires a diagonal.")
        return bias * down.diag_scale.to(device=bias.device, dtype=bias.dtype)
    return bias


def _official_transform_activation(
    x: torch.Tensor,
    *,
    role: str,
    transform_bank: nn.ModuleDict,
    head_dim: int,
) -> torch.Tensor:
    if not _formal_transform_bank(transform_bank):
        transform = transform_bank[_legacy_transform_name(role)]
        if not isinstance(transform, (KroneckerSVDTransform, FixedSmoothQuantTransform)):
            raise TypeError(f"Unexpected legacy FlatQuant transform: {type(transform)!r}")
        return transform(x)

    if role in ("q", "k", "v"):
        return _transform_matrix_only(
            _require_kronecker_transform(transform_bank, "attn_in"),
            x,
        )
    if role == "o":
        return _apply_head_axis_transform(
            x,
            _require_single_transform(transform_bank, "head"),
            head_dim=head_dim,
        )
    if role in ("gate", "up"):
        return _transform_matrix_only(
            _require_kronecker_transform(transform_bank, "mlp_in"),
            x,
        )
    if role == "down":
        return _transform_matrix_only(
            _require_kronecker_transform(transform_bank, "down"),
            x,
        )
    raise ValueError(f"Unknown FlatQuant linear role: {role!r}")


class _TrainableFlatQuantScaledNorm(nn.Module):
    """Execute an official FlatQuant diagonal already folded into RMSNorm."""

    def __init__(
        self,
        norm: nn.Module,
        *,
        transform_bank: nn.ModuleDict,
        transform_name: str,
    ) -> None:
        super().__init__()
        self.base_norm = copy.deepcopy(norm)
        self.base_norm.requires_grad_(False)
        self.transform_name = transform_name
        object.__setattr__(self, "_flat_transform_bank", transform_bank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        transform = _require_kronecker_transform(
            self._flat_transform_bank,
            self.transform_name,
        )
        if transform.diag_scale is None:
            raise ValueError(
                f"FlatQuant transform {self.transform_name!r} has no diagonal."
            )
        output = self.base_norm(hidden_states)
        return output * transform.diag_scale.to(
            device=output.device,
            dtype=output.dtype,
        )



def _activation_site_for_role(role: str) -> str:
    if role in ("q", "k", "v"):
        return "attn_in"
    if role == "o":
        return "o"
    if role in ("gate", "up"):
        return "mlp_in"
    if role == "down":
        return "down"
    raise ValueError(f"Unknown FlatQuant linear role: {role!r}")


class _LearnableActivationClip(nn.Module):
    """Official two-sided scalar LAC followed by per-token symmetric QDQ."""

    def __init__(self, config: FlatQuantCoreConfig) -> None:
        super().__init__()
        initial = float(config.init_lac_logit)
        self.upper_clip_logit = nn.Parameter(torch.tensor([initial], dtype=torch.float32))
        self.lower_clip_logit = nn.Parameter(torch.tensor([initial], dtype=torch.float32))
        learn_lac = config.use_lac and config.learn_lac
        self.upper_clip_logit.requires_grad_(learn_lac)
        self.lower_clip_logit.requires_grad_(learn_lac)
        self.quant_format = config.activation_quant_format
        self.eps = float(config.eps)
        self.use_lac = bool(config.use_lac)

    def clip_parameters(self) -> list[nn.Parameter]:
        return [self.upper_clip_logit, self.lower_clip_logit]

    def clipped(self, x: torch.Tensor) -> torch.Tensor:
        x_fp = x.float()
        reshaped = x_fp.reshape(-1, x_fp.shape[-1])
        zeros = torch.zeros(
            (reshaped.shape[0], 1),
            device=reshaped.device,
            dtype=reshaped.dtype,
        )
        upper = torch.maximum(reshaped.amax(dim=1, keepdim=True), zeros)
        lower = torch.minimum(reshaped.amin(dim=1, keepdim=True), zeros)
        if self.use_lac:
            upper = upper * torch.sigmoid(self.upper_clip_logit)
            lower = lower * torch.sigmoid(self.lower_clip_logit)
        clipped = torch.minimum(torch.maximum(reshaped, lower), upper)
        return clipped.reshape_as(x_fp).to(dtype=x.dtype)

    def qdq(self, x: torch.Tensor, *, use_ste: bool) -> torch.Tensor:
        clipped = self.clipped(x)
        if use_ste:
            return _trainable_activation_qdq(
                clipped,
                quant_format=self.quant_format,
                eps=self.eps,
            )
        qmax = (
            FP8_MAX
            if self.quant_format == "fp8_e4m3fn"
            else quant_format_qmax(self.quant_format)
        )
        return activation_per_token_qdq_by_format(
            clipped,
            quant_format=self.quant_format,
            eps=self.eps,
            fp8_qmax=qmax,
        )


class _TrainableFlatQuantLinear(_TrainableSymmetricLinear):
    """Deployment-matched LWC with the official FlatQuant transform topology."""

    is_flatquant_linear = True

    def __init__(
        self,
        linear: nn.Linear,
        *,
        config: FlatQuantCoreConfig,
        transform_bank: nn.ModuleDict,
        activation_quantizer_bank: nn.ModuleDict,
        role: str,
        head_dim: int,
    ) -> None:
        super().__init__(
            linear,
            config=config,  # type: ignore[arg-type]
            let_parameters=nn.ParameterDict(),
        )
        self.role = role
        self.head_dim = int(head_dim)
        object.__setattr__(self, "_flat_transform_bank", transform_bank)
        object.__setattr__(
            self,
            "_flat_activation_quantizer_bank",
            activation_quantizer_bank,
        )
        self.act_quant: ActQuant = "per_token"
        self.activation_quant_format = config.activation_quant_format
        self.weight_quant_format = config.weight_quant_format
        self.weight_quant_scheme = config.weight_quant_scheme
        self.qmax = (
            FP8_MAX
            if config.activation_quant_format == "fp8_e4m3fn"
            else quant_format_qmax(config.activation_quant_format)
        )
        # Official FlatQuant keeps independent upper/lower LWC logits even
        # when the following FP/INT quantizer is symmetric.
        if config.weight_quant_scheme == "symmetric":
            assert self.clip_logits is not None
            parameter_shape = tuple(self.clip_logits.shape)
            del self.clip_logits
            self.register_parameter("clip_logits", None)
            self.upper_clip_logits = nn.Parameter(
                torch.full(
                    parameter_shape,
                    float(config.init_lwc_logit),
                    device=linear.weight.device,
                )
            )
            self.lower_clip_logits = nn.Parameter(
                torch.full(
                    parameter_shape,
                    float(config.init_lwc_logit),
                    device=linear.weight.device,
                )
            )
            self.upper_clip_logits.requires_grad_(config.use_lwc)
            self.lower_clip_logits.requires_grad_(config.use_lwc)

    def lwc_parameters(self) -> list[nn.Parameter]:
        if self.config.weight_quant_scheme != "symmetric":
            return super().lwc_parameters()
        assert self.upper_clip_logits is not None
        assert self.lower_clip_logits is not None
        return [self.upper_clip_logits, self.lower_clip_logits]

    def lwc_state(self) -> dict[str, torch.Tensor]:
        if self.config.weight_quant_scheme != "symmetric":
            return super().lwc_state()
        assert self.upper_clip_logits is not None
        assert self.lower_clip_logits is not None
        return {
            "upper_clip_logits": self.upper_clip_logits.detach().cpu(),
            "lower_clip_logits": self.lower_clip_logits.detach().cpu(),
        }

    def _qdq_weight_tensor(
        self,
        weight: torch.Tensor,
        *,
        use_ste: bool,
    ) -> torch.Tensor:
        if self.config.weight_quant_scheme != "symmetric":
            return super()._qdq_weight_tensor(weight, use_ste=use_ste)

        assert self.upper_clip_logits is not None
        assert self.lower_clip_logits is not None
        round_fn = _round_ste if use_ste else torch.round
        grouped_weight, valid_mask, padded_in_features = self._group_weight(
            weight.float()
        )
        if valid_mask is None:
            upper = grouped_weight.amax(dim=-1, keepdim=True)
            lower = grouped_weight.amin(dim=-1, keepdim=True)
        else:
            upper = grouped_weight.masked_fill(
                ~valid_mask,
                -torch.inf,
            ).amax(dim=-1, keepdim=True)
            lower = grouped_weight.masked_fill(
                ~valid_mask,
                torch.inf,
            ).amin(dim=-1, keepdim=True)
        if self.config.use_lwc:
            upper = upper * torch.sigmoid(self.upper_clip_logits)
            lower = lower * torch.sigmoid(self.lower_clip_logits)
        clipped = torch.minimum(torch.maximum(grouped_weight, lower), upper)
        threshold = clipped.abs().amax(dim=-1, keepdim=True)
        threshold = threshold.clamp_min(self.config.eps)

        if self.config.weight_quant_format == "fp8_e4m3fn":
            scale = threshold / FP8_MAX
            normalized = (clipped / scale).clamp(min=-FP8_MAX, max=FP8_MAX)
            q = (
                _fp8_cast_ste(normalized)
                if use_ste
                else normalized.to(torch.float8_e4m3fn).float()
            )
            qdq = q * scale
        elif self.config.weight_quant_format == "fp4_e2m1":
            scale = threshold / FP4_E2M1_MAX
            normalized = (clipped / scale).clamp(
                min=-FP4_E2M1_MAX,
                max=FP4_E2M1_MAX,
            )
            q = (
                _fp4_e2m1_ste(normalized)
                if use_ste
                else fp4_e2m1_quantize(normalized)
            )
            qdq = q * scale
        else:
            qmax = _signed_qmax(self.config.weight_quant_format)
            scale = threshold / qmax
            q = round_fn(clipped / scale).clamp(min=-qmax, max=qmax)
            qdq = q * scale
        return self._restore_grouped_weight(
            qdq,
            padded_in_features=padded_in_features,
        ).to(dtype=weight.dtype)

    def transformed_weight(self) -> torch.Tensor:
        return _official_transformed_weight(
            self.weight_fp,
            role=self.role,
            transform_bank=self._flat_transform_bank,
            head_dim=self.head_dim,
        )

    def transformed_bias(self) -> torch.Tensor | None:
        return _official_transformed_bias(
            self.bias_fp,
            role=self.role,
            transform_bank=self._flat_transform_bank,
        )

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        transformed = _official_transform_activation(
            x,
            role=self.role,
            transform_bank=self._flat_transform_bank,
            head_dim=self.head_dim,
        )
        quantizer = self._flat_activation_quantizer_bank[
            _activation_site_for_role(self.role)
        ]
        if not isinstance(quantizer, _LearnableActivationClip):
            raise TypeError(
                f"Unexpected FlatQuant activation quantizer: {type(quantizer)!r}"
            )
        return quantizer.qdq(transformed, use_ste=True)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize_activation(x)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qdq_weight().to(dtype=self.execution_dtype)
        bias = self.transformed_bias()
        if bias is not None:
            bias = bias.to(dtype=self.execution_dtype)
        return F.linear(x.to(dtype=self.execution_dtype), weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_prepared(self.quantize_activation(x))


class FlatQuantCoreLinear(nn.Module):
    """Static fake-QDQ Linear retaining only FlatQuant's online matrix."""

    is_flatquant_linear = True

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        transform_bank: nn.ModuleDict,
        activation_quantizer_bank: nn.ModuleDict,
        role: str,
        head_dim: int,
        config: FlatQuantCoreConfig,
    ) -> None:
        super().__init__()
        if weight_qdq.ndim != 2:
            raise ValueError(f"Expected 2D weight, got {tuple(weight_qdq.shape)}")
        self.in_features = int(weight_qdq.shape[1])
        self.out_features = int(weight_qdq.shape[0])
        self.role = role
        self.head_dim = int(head_dim)
        object.__setattr__(self, "_flat_transform_bank", transform_bank)
        object.__setattr__(
            self,
            "_flat_activation_quantizer_bank",
            activation_quantizer_bank,
        )
        self.act_quant: ActQuant = "per_token"
        self.activation_quant_format = config.activation_quant_format
        self.weight_quant_format = config.weight_quant_format
        self.weight_quant_scheme = config.weight_quant_scheme
        self.weight_group_size = normalize_weight_group_size(config.weight_group_size)
        self.eps = float(config.eps)
        self.qmax = (
            FP8_MAX
            if config.activation_quant_format == "fp8_e4m3fn"
            else quant_format_qmax(config.activation_quant_format)
        )
        self.register_buffer("weight_qdq", weight_qdq.detach().clone(), persistent=True)
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().clone(),
            persistent=True,
        )

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        transformed = _official_transform_activation(
            x,
            role=self.role,
            transform_bank=self._flat_transform_bank,
            head_dim=self.head_dim,
        )
        quantizer = self._flat_activation_quantizer_bank[
            _activation_site_for_role(self.role)
        ]
        if not isinstance(quantizer, _LearnableActivationClip):
            raise TypeError(
                f"Unexpected FlatQuant activation quantizer: {type(quantizer)!r}"
            )
        return quantizer.qdq(transformed, use_ste=False)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_prepared(self.quantize_activation(x))


class _TrainableFlatQuantBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        *,
        config: FlatQuantCoreConfig,
        act_quant_mode: ActQuantMode,
        init_scales: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        q_proj = block.get_submodule("self_attn.q_proj")
        if not isinstance(q_proj, nn.Linear):
            raise TypeError("FlatQuant expects Qwen-style self_attn.q_proj.")
        self.config = config
        self.act_quant_mode = act_quant_mode
        self.inference_dtype = q_proj.weight.dtype
        self.block = block.to(dtype=self.inference_dtype)
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

        attention = self.block.get_submodule("self_attn")
        attention_config = getattr(attention, "config", None)
        configured_heads = getattr(attention_config, "num_attention_heads", None)
        resolved_head_dim = getattr(attention, "head_dim", None)
        if resolved_head_dim is None:
            if configured_heads is None:
                raise ValueError(
                    "FlatQuant needs self_attn.head_dim or "
                    "self_attn.config.num_attention_heads."
                )
            resolved_head_dim = q_proj.out_features // int(configured_heads)
        self.head_dim = int(resolved_head_dim)
        self.num_attention_heads = int(
            configured_heads
            if configured_heads is not None
            else q_proj.out_features // self.head_dim
        )
        if q_proj.out_features != self.num_attention_heads * self.head_dim:
            raise ValueError("Qwen3 q_proj dimensions do not match attention heads.")
        o_proj = self._linear("self_attn.o_proj")
        if o_proj.in_features != self.num_attention_heads * self.head_dim:
            raise ValueError("Qwen3 o_proj dimensions do not match attention heads.")

        resolved_scales = {} if init_scales is None else dict(init_scales)
        if config.transform_kind == "kronecker":
            transforms: dict[str, nn.Module] = {
                "attn_in": KroneckerSVDTransform(
                    q_proj.in_features,
                    init=config.transform_init,
                    add_diag=True,
                    diag_init=resolved_scales.get("attn_in"),
                ),
                "head": SingleSVDTransform(
                    self.num_attention_heads,
                    init=config.transform_init,
                ),
                "head_dim": SingleSVDTransform(
                    self.head_dim,
                    init=config.transform_init,
                ),
                "mlp_in": KroneckerSVDTransform(
                    self._linear("mlp.gate_proj").in_features,
                    init=config.transform_init,
                    add_diag=True,
                    diag_init=resolved_scales.get("mlp_in"),
                ),
                "down": KroneckerSVDTransform(
                    self._linear("mlp.down_proj").in_features,
                    init=config.transform_init,
                    add_diag=True,
                    diag_init=resolved_scales.get("down"),
                ),
            }
        else:
            transform_sizes = {
                "qkv": q_proj.in_features,
                "o": o_proj.in_features,
                "mlp": self._linear("mlp.gate_proj").in_features,
                "down": self._linear("mlp.down_proj").in_features,
            }
            transforms = {}
            for name, size in transform_sizes.items():
                scale = resolved_scales.get(name)
                if scale is None:
                    scale = torch.ones(size, dtype=torch.float32)
                if scale.numel() != size:
                    raise ValueError(
                        f"SmoothQuant scale {name!r} has {scale.numel()} entries; "
                        f"expected {size}."
                    )
                transforms[name] = FixedSmoothQuantTransform(scale)
        self.transforms = nn.ModuleDict(transforms).to(device=q_proj.weight.device)
        self.activation_quantizers = nn.ModuleDict(
            {
                name: _LearnableActivationClip(config)
                for name in ("attn_in", "o", "mlp_in", "down")
            }
        ).to(device=q_proj.weight.device)

        if _formal_transform_bank(self.transforms):
            self._replace_norms()
        for linear_name, role in FLATQUANT_LINEAR_ROLES.items():
            self._replace_linear(linear_name, role)
        if act_quant_mode == "shared_input":
            install_shared_input_activation_quantization(self.block)
        elif act_quant_mode != "per_linear":
            raise ValueError(f"Unsupported activation quantization mode: {act_quant_mode!r}")

    def _linear(self, name: str) -> nn.Linear:
        linear = self.block.get_submodule(name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected {name} to be nn.Linear, got {type(linear)!r}.")
        return linear

    def _replace_norms(self) -> None:
        for norm_name, transform_name in (
            ("input_layernorm", "attn_in"),
            ("post_attention_layernorm", "mlp_in"),
        ):
            norm = self.block.get_submodule(norm_name)
            setattr(
                self.block,
                norm_name,
                _TrainableFlatQuantScaledNorm(
                    norm,
                    transform_bank=self.transforms,
                    transform_name=transform_name,
                ),
            )

    def _replace_linear(self, name: str, role: str) -> None:
        linear = self._linear(name)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(
            self.block.get_submodule(parent_name),
            child_name,
            _TrainableFlatQuantLinear(
                linear,
                config=self.config,
                transform_bank=self.transforms,
                activation_quantizer_bank=self.activation_quantizers,
                role=role,
                head_dim=self.head_dim,
            ),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        args = _floating_tree_to_dtype(args, self.inference_dtype)
        kwargs = _floating_tree_to_dtype(kwargs, self.inference_dtype)
        return self.block(*args, **kwargs)

def _flat_wrappers(module: nn.Module) -> dict[str, _TrainableFlatQuantLinear]:
    return {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, _TrainableFlatQuantLinear)
    }


def _flat_split_counts(
    sample_count: int,
    config: FlatQuantCoreConfig,
) -> tuple[int, int, int]:
    """Return train count, held-out start, and held-out count."""
    validation_count = config.validation_sample_size
    if validation_count >= sample_count:
        raise ValueError(
            "FlatQuant validation_sample_size must be smaller than the loaded "
            f"sample count; got validation={validation_count}, total={sample_count}."
        )
    validation_start = sample_count - validation_count
    train_count = config.train_sample_size or validation_start
    if train_count <= 0 or train_count > validation_start:
        raise ValueError(
            "FlatQuant train_sample_size must select at least one sample and "
            "must not overlap the held-out tail."
        )
    return train_count, validation_start, validation_count


def _collect_fixed_smoothquant_transform_scales(
    teacher_block: nn.Module,
    fp_inputs: Sequence[Batch],
    *,
    config: FlatQuantCoreConfig,
) -> dict[str, torch.Tensor]:
    """Collect diagonal scales from the training prefix only."""
    train_count, _validation_start, _validation_count = _flat_split_counts(
        len(fp_inputs),
        config,
    )
    raw_scales = collect_smoothquant_scales(
        teacher_block,
        fp_inputs[:train_count],
        alpha=config.smoothquant_alpha,
    )
    result: dict[str, torch.Tensor] = {}
    for transform_name, linear_name in FLATQUANT_SMOOTHQUANT_SCALE_SOURCES.items():
        scale = raw_scales.get(linear_name)
        if scale is None:
            raise KeyError(
                f"Missing SmoothQuant scale for {linear_name!r} "
                f"({transform_name!r} transform)."
            )
        result[transform_name] = scale
    return result



def _collect_flatquant_diag_initials(
    teacher_block: nn.Module,
    fp_inputs: Sequence[Batch],
    *,
    config: FlatQuantCoreConfig,
) -> dict[str, torch.Tensor]:
    """Initialize the three official diagonals from the training prefix only."""
    train_count, _validation_start, _validation_count = _flat_split_counts(
        len(fp_inputs),
        config,
    )
    smooth_scales = collect_smoothquant_scales(
        teacher_block,
        fp_inputs[:train_count],
        alpha=config.diag_alpha,
        # Official get_init_scale clamps D = W^(1-a) / X^a from below at
        # 1e-5. This helper returns its reciprocal SmoothQuant scale.
        max_scale=1e5,
    )
    source_names = {
        "attn_in": "self_attn.q_proj",
        "mlp_in": "mlp.gate_proj",
        "down": "mlp.down_proj",
    }
    diagonals: dict[str, torch.Tensor] = {}
    for transform_name, source_name in source_names.items():
        scale = smooth_scales.get(source_name)
        if scale is None:
            raise KeyError(
                f"Missing FlatQuant diagonal statistics for {source_name!r}."
            )
        # The repository SmoothQuant convention is X / s and W * s.
        # Official FlatQuant names the reciprocal D, so X * D and W / D.
        diagonal = scale.detach().float().reshape(-1).reciprocal()
        if not torch.isfinite(diagonal).all() or torch.any(diagonal == 0):
            raise FloatingPointError(
                f"Invalid FlatQuant diagonal initialization for {transform_name!r}."
            )
        diagonals[transform_name] = diagonal
    return diagonals


def _train_flat_block(
    *,
    teacher_block: nn.Module,
    train_block: _TrainableFlatQuantBlock,
    fp_inputs: Sequence[Batch],
    quant_inputs: Sequence[Batch],
    config: FlatQuantCoreConfig,
    layer_idx: int,
) -> tuple[
    float,
    float,
    float | None,
    float | None,
    int | None,
    tuple[FlatQuantEpochMetric, ...],
]:
    if len(fp_inputs) != len(quant_inputs) or not fp_inputs:
        raise ValueError("FlatQuant FP/quant calibration streams must be non-empty and aligned.")
    device = _module_device(train_block)
    teacher_block.eval()
    train_block.eval()

    teacher_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for fp_batch in fp_inputs:
            args, kwargs = _batch_to_args_kwargs(fp_batch)
            args = _move_tree_to_device(args, device)
            kwargs = _move_tree_to_device(kwargs, device)
            teacher_targets.append(
                _first_tensor(teacher_block(*args, **kwargs)).detach().cpu()
            )

    train_count, validation_start, validation_count = _flat_split_counts(
        len(quant_inputs),
        config,
    )
    train_quant_inputs = quant_inputs[:train_count]
    train_targets = teacher_targets[:train_count]
    validation_quant_inputs = (
        quant_inputs[validation_start:] if validation_count else ()
    )
    validation_targets = (
        teacher_targets[validation_start:] if validation_count else ()
    )
    print(
        f"[flatquant][split] layer={layer_idx} train_samples={train_count} "
        f"unused_samples={validation_start - train_count} "
        f"heldout_samples={validation_count}"
    )

    def evaluate(
        input_batches: Sequence[Batch],
        target_batches: Sequence[torch.Tensor],
    ) -> float:
        if not input_batches or len(input_batches) != len(target_batches):
            raise ValueError("FlatQuant evaluation batches must be non-empty and aligned.")
        was_training = train_block.training
        train_block.eval()
        total = 0.0
        with torch.no_grad():
            for quant_batch, target_cpu in zip(input_batches, target_batches):
                args, kwargs = _batch_to_args_kwargs(quant_batch)
                args = _move_tree_to_device(args, device)
                kwargs = _move_tree_to_device(kwargs, device)
                prediction = _first_tensor(train_block(*args, **kwargs))
                target = target_cpu.to(device=device, dtype=prediction.dtype)
                total += float(F.mse_loss(prediction.float(), target.float()))
        if was_training:
            train_block.train()
        return total / len(input_batches)

    wrappers = _flat_wrappers(train_block.block)
    all_transform_parameters = list(train_block.transforms.parameters())
    for parameter in all_transform_parameters:
        parameter.requires_grad_(config.learn_transform)
    transform_parameters = (
        all_transform_parameters if config.learn_transform else []
    )
    lwc_parameters = [
        parameter
        for wrapper in wrappers.values()
        for parameter in wrapper.lwc_parameters()
        if config.use_lwc
    ]
    lac_parameters = [
        parameter
        for quantizer in train_block.activation_quantizers.values()
        for parameter in quantizer.clip_parameters()
        if config.use_lac and config.learn_lac
    ]
    parameter_groups: list[dict[str, Any]] = []
    transform_group_index: int | None = None
    lwc_group_index: int | None = None
    lac_group_index: int | None = None
    if transform_parameters:
        transform_group_index = len(parameter_groups)
        parameter_groups.append(
            {"params": transform_parameters, "lr": config.transform_lr}
        )
    if lwc_parameters:
        lwc_group_index = len(parameter_groups)
        parameter_groups.append(
            {"params": lwc_parameters, "lr": config.lwc_lr}
        )
    if lac_parameters:
        lac_group_index = len(parameter_groups)
        parameter_groups.append(
            {"params": lac_parameters, "lr": config.lac_lr}
        )
    optimizer = (
        torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.weight_decay,
        )
        if parameter_groups
        else None
    )
    total_steps = config.epochs * len(train_quant_inputs)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            # Match official FlatQuant: one absolute floor based on flat_lr,
            # including the LWC/LAC groups initialized at 10x flat_lr.
            eta_min=config.transform_lr * config.min_lr_factor,
        )
        if optimizer is not None
        else None
    )
    trainable_parameters = transform_parameters + lwc_parameters + lac_parameters

    initial_mse = evaluate(train_quant_inputs, train_targets)
    initial_validation_mse = (
        evaluate(validation_quant_inputs, validation_targets)
        if validation_quant_inputs
        else None
    )
    metrics: list[FlatQuantEpochMetric] = [
        FlatQuantEpochMetric(
            epoch=0,
            train_mse=None,
            eval_mse=initial_mse,
            mean_grad_norm=None,
            max_grad_norm=None,
            transform_lr=(
                config.transform_lr if transform_parameters else None
            ),
            lwc_lr=config.lwc_lr if lwc_parameters else None,
            lac_lr=config.lac_lr if lac_parameters else None,
            validation_mse=initial_validation_mse,
        )
    ]
    validation_text = (
        ""
        if initial_validation_mse is None
        else f" heldout_mse={initial_validation_mse:.6e}"
    )
    print(
        f"[flatquant][initial] layer={layer_idx} train_mse={initial_mse:.6e}"
        f"{validation_text} learn_transform={config.learn_transform} "
        f"use_lwc={config.use_lwc} use_lac={config.use_lac} "
        f"learn_lac={config.learn_lac}"
    )

    if optimizer is None:
        print(f"[flatquant][no_train] layer={layer_idx} arm=RTN")
        return (
            initial_mse,
            initial_mse,
            initial_validation_mse,
            initial_validation_mse,
            0,
            tuple(metrics),
        )

    def snapshot_parameters() -> tuple[torch.Tensor, ...]:
        return tuple(
            parameter.detach().cpu().clone()
            for parameter in trainable_parameters
        )

    def restore_parameters(snapshot: Sequence[torch.Tensor]) -> None:
        if len(snapshot) != len(trainable_parameters):
            raise ValueError("FlatQuant best-state parameter count mismatch.")
        with torch.no_grad():
            for parameter, saved in zip(trainable_parameters, snapshot):
                parameter.copy_(
                    saved.to(device=parameter.device, dtype=parameter.dtype)
                )

    best_epoch = 0
    best_mse = initial_mse
    best_snapshot = snapshot_parameters()
    checked_trainable_gradient = False
    train_block.train()
    for epoch in range(1, config.epochs + 1):
        mse_total = 0.0
        grad_total = 0.0
        grad_max = 0.0
        for quant_batch, target_cpu in zip(train_quant_inputs, train_targets):
            args, kwargs = _batch_to_args_kwargs(quant_batch)
            args = _move_tree_to_device(args, device)
            kwargs = _move_tree_to_device(kwargs, device)
            prediction = _first_tensor(train_block(*args, **kwargs))
            target = target_cpu.to(device=device, dtype=prediction.dtype)
            mse = F.mse_loss(prediction.float(), target.float())
            if not torch.isfinite(mse):
                raise FloatingPointError("Non-finite FlatQuant block MSE before optimizer step.")
            optimization_loss = (
                mse / mse.detach().clamp_min(config.eps)
                if config.normalize_mse_gradient
                else mse
            )
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            optimization_loss.backward()
            if not checked_trainable_gradient:
                checked_trainable_gradient = True
                if not any(
                    parameter.grad is not None
                    and torch.count_nonzero(parameter.grad).item() > 0
                    for parameter in trainable_parameters
                ):
                    raise RuntimeError("All FlatQuant trainable gradients are zero.")
                for group_name, group_parameters in (
                    ("transform", transform_parameters),
                    ("LWC", lwc_parameters),
                    ("LAC", lac_parameters),
                ):
                    if group_parameters and not any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in group_parameters
                    ):
                        raise RuntimeError(
                            f"All FlatQuant {group_name} gradients are zero."
                        )
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable_parameters
            ):
                raise FloatingPointError("Non-finite FlatQuant gradient encountered.")
            grad_tensor = (
                _total_grad_norm(trainable_parameters)
                if config.max_grad_norm is None
                else torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config.max_grad_norm,
                )
            )
            grad_norm = float(grad_tensor.detach())
            optimizer.step()
            assert scheduler is not None
            scheduler.step()
            if any(not torch.isfinite(parameter).all() for parameter in trainable_parameters):
                raise FloatingPointError("Non-finite FlatQuant parameter after optimizer step.")
            mse_total += float(mse.detach())
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)

        transform_lr = (
            float(optimizer.param_groups[transform_group_index]["lr"])
            if transform_group_index is not None
            else None
        )
        current_lwc_lr = (
            float(optimizer.param_groups[lwc_group_index]["lr"])
            if lwc_group_index is not None
            else None
        )
        current_lac_lr = (
            float(optimizer.param_groups[lac_group_index]["lr"])
            if lac_group_index is not None
            else None
        )
        should_evaluate = config.epoch_eval_interval > 0 and (
            epoch % config.epoch_eval_interval == 0
            or epoch == config.epochs
        )
        evaluated_mse = (
            evaluate(train_quant_inputs, train_targets)
            if should_evaluate
            else None
        )
        validation_mse = (
            evaluate(validation_quant_inputs, validation_targets)
            if should_evaluate and validation_quant_inputs
            else None
        )
        if evaluated_mse is not None and evaluated_mse < best_mse:
            best_epoch = epoch
            best_mse = evaluated_mse
            best_snapshot = snapshot_parameters()
        metric = FlatQuantEpochMetric(
            epoch=epoch,
            train_mse=mse_total / len(train_quant_inputs),
            eval_mse=evaluated_mse,
            mean_grad_norm=grad_total / len(train_quant_inputs),
            max_grad_norm=grad_max,
            transform_lr=transform_lr,
            lwc_lr=current_lwc_lr,
            lac_lr=current_lac_lr,
            validation_mse=validation_mse,
        )
        metrics.append(metric)
        eval_text = (
            ""
            if evaluated_mse is None
            else f" eval_mse={evaluated_mse:.6e}"
        )
        heldout_text = (
            ""
            if validation_mse is None
            else f" heldout_mse={validation_mse:.6e}"
        )
        print(
            f"[flatquant][epoch] layer={layer_idx} epoch={epoch}/{config.epochs} "
            f"train_mse={metric.train_mse:.6e} "
            f"grad_norm_mean={metric.mean_grad_norm:.6e} "
            f"grad_norm_max={grad_max:.6e}"
            f"{eval_text}{heldout_text}"
        )
        train_block.train()

    train_block.eval()
    selected_epoch: int | None = None
    if config.epoch_eval_interval > 0:
        restore_parameters(best_snapshot)
        selected_epoch = best_epoch
    final_mse = evaluate(train_quant_inputs, train_targets)
    final_validation_mse = (
        evaluate(validation_quant_inputs, validation_targets)
        if validation_quant_inputs
        else None
    )
    if config.epoch_eval_interval == 0:
        last = metrics[-1]
        metrics[-1] = FlatQuantEpochMetric(
            epoch=last.epoch,
            train_mse=last.train_mse,
            eval_mse=final_mse,
            mean_grad_norm=last.mean_grad_norm,
            max_grad_norm=last.max_grad_norm,
            transform_lr=last.transform_lr,
            lwc_lr=last.lwc_lr,
            lac_lr=last.lac_lr,
            validation_mse=final_validation_mse,
        )
    final_validation_text = (
        ""
        if final_validation_mse is None
        else f" heldout_mse={final_validation_mse:.6e}"
    )
    print(
        f"[flatquant][final] layer={layer_idx} train_mse={final_mse:.6e}"
        f"{final_validation_text} best_epoch={selected_epoch}"
    )
    return (
        initial_mse,
        final_mse,
        initial_validation_mse,
        final_validation_mse,
        selected_epoch,
        tuple(metrics),
    )


@dataclass(frozen=True)
class _FlatQuantLFQTrainResult:
    initial_loss: float
    final_loss: float
    initial_slot_losses: dict[str, float]
    final_slot_losses: dict[str, float]
    initial_boundary_loss: float | None
    final_boundary_loss: float | None
    initial_boundary_slot_losses: dict[str, float]
    final_boundary_slot_losses: dict[str, float]
    initial_mse: float
    final_mse: float
    initial_validation_loss: float | None
    final_validation_loss: float | None
    initial_validation_slot_losses: dict[str, float]
    final_validation_slot_losses: dict[str, float]
    initial_validation_mse: float | None
    final_validation_mse: float | None
    best_epoch: int | None
    metrics: tuple[FlatQuantEpochMetric, ...]


def _normalized_flat_lfq_slot_weights(
    config: FlatQuantCoreConfig,
) -> dict[str, float]:
    total = float(sum(config.lfq_slot_weights))
    return {
        slot: float(weight) / total
        for slot, weight in zip(LFQ_SLOT_NAMES, config.lfq_slot_weights)
    }


def _train_flat_block_lfq(
    *,
    teacher_block: nn.Module,
    train_block: _TrainableFlatQuantBlock,
    fp_inputs: Sequence[Batch],
    quant_inputs: Sequence[Batch],
    config: FlatQuantCoreConfig,
    lfq_projector: _LFQOutputProjector,
    layer_idx: int,
) -> _FlatQuantLFQTrainResult:
    """Fine-tune one FlatQuant block with the existing ABC CE/boundary target."""

    if len(fp_inputs) != len(quant_inputs) or not fp_inputs:
        raise ValueError("FlatQuant LFQ streams must be non-empty and aligned.")
    if config.final_objective != "lfq_ce":
        raise ValueError("FlatQuant LFQ training requires final_objective='lfq_ce'.")

    train_count, validation_start, validation_count = _flat_split_counts(
        len(quant_inputs),
        config,
    )
    train_fp_inputs = fp_inputs[:train_count]
    train_quant_inputs = quant_inputs[:train_count]
    validation_fp_inputs = (
        fp_inputs[validation_start:] if validation_count else ()
    )
    validation_quant_inputs = (
        quant_inputs[validation_start:] if validation_count else ()
    )
    print(
        f"[flatquant][lfq_split] layer={layer_idx} train_samples={train_count} "
        f"unused_samples={validation_start - train_count} "
        f"heldout_samples={validation_count}"
    )

    device = _module_device(train_block)
    teacher_block.eval()
    train_block.eval()
    lfq_projector.eval()
    slot_weights = _normalized_flat_lfq_slot_weights(config)
    use_boundary = config.lfq_boundary_loss_weight > 0.0

    def cache_teacher_targets(
        input_batches: Sequence[Batch],
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        list[dict[str, _LFQBoundaryTarget]] | None,
    ]:
        probabilities: list[dict[str, torch.Tensor]] = []
        boundaries: list[dict[str, _LFQBoundaryTarget]] | None = (
            [] if use_boundary else None
        )
        with torch.no_grad():
            for fp_batch in input_batches:
                args, kwargs = _batch_to_args_kwargs(fp_batch)
                args = _move_tree_to_device(args, device)
                kwargs = _move_tree_to_device(kwargs, device)
                hidden = _first_tensor(teacher_block(*args, **kwargs))
                logits = {
                    slot: value.float()
                    for slot, value in lfq_projector(hidden).items()
                }
                probabilities.append(
                    {
                        slot: F.softmax(logits[slot], dim=-1).detach().cpu()
                        for slot in LFQ_SLOT_NAMES
                    }
                )
                if boundaries is not None:
                    boundaries.append(
                        _build_lfq_boundary_targets(
                            logits,
                            topk=config.lfq_boundary_topk,
                            negative_count=config.lfq_boundary_negative_count,
                            tie_threshold=config.lfq_boundary_tie_threshold,
                            gap_scale=config.lfq_boundary_gap_scale,
                        )
                    )
        return probabilities, boundaries

    train_probabilities, train_boundaries = cache_teacher_targets(train_fp_inputs)
    validation_probabilities, validation_boundaries = (
        cache_teacher_targets(validation_fp_inputs)
        if validation_fp_inputs
        else ([], None)
    )

    def student_objective(
        quant_batch: Batch,
        teacher_probabilities: Mapping[str, torch.Tensor],
        boundary_targets: Mapping[str, _LFQBoundaryTarget] | None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor | None,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        args, kwargs = _batch_to_args_kwargs(quant_batch)
        args = _move_tree_to_device(args, device)
        kwargs = _move_tree_to_device(kwargs, device)
        prediction = _first_tensor(train_block(*args, **kwargs))
        logits = {
            slot: value.float()
            for slot, value in lfq_projector(prediction).items()
        }
        ce, slot_losses = _lfq_soft_cross_entropy_from_logits(
            logits,
            teacher_probabilities,
            slot_weights,
        )
        loss = config.lfq_loss_weight * ce
        boundary_loss: torch.Tensor | None = None
        boundary_slot_losses: dict[str, torch.Tensor] = {}
        if boundary_targets is not None:
            boundary_loss, boundary_slot_losses = _lfq_boundary_loss(
                logits,
                boundary_targets,
                slot_weights,
            )
            loss = loss + config.lfq_boundary_loss_weight * boundary_loss
        return loss, slot_losses, boundary_loss, boundary_slot_losses, prediction

    def evaluate(
        fp_batches: Sequence[Batch],
        quant_batches: Sequence[Batch],
        probabilities: Sequence[Mapping[str, torch.Tensor]],
        boundaries: Sequence[Mapping[str, _LFQBoundaryTarget]] | None,
    ) -> tuple[float, dict[str, float], float | None, dict[str, float], float]:
        if not fp_batches or not (
            len(fp_batches) == len(quant_batches) == len(probabilities)
        ):
            raise ValueError("FlatQuant LFQ evaluation streams must be aligned.")
        was_training = train_block.training
        train_block.eval()
        total = 0.0
        slot_totals = {slot: 0.0 for slot in LFQ_SLOT_NAMES}
        boundary_total = 0.0
        boundary_slot_totals = {slot: 0.0 for slot in LFQ_SLOT_NAMES}
        mse_total = 0.0
        with torch.no_grad():
            for batch_idx, (fp_batch, quant_batch, teacher_probabilities) in enumerate(
                zip(fp_batches, quant_batches, probabilities)
            ):
                boundary_targets = (
                    None if boundaries is None else boundaries[batch_idx]
                )
                (
                    loss,
                    slot_losses,
                    boundary_loss,
                    boundary_slot_losses,
                    prediction,
                ) = student_objective(
                    quant_batch,
                    teacher_probabilities,
                    boundary_targets,
                )
                fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
                fp_args = _move_tree_to_device(fp_args, device)
                fp_kwargs = _move_tree_to_device(fp_kwargs, device)
                target = _first_tensor(teacher_block(*fp_args, **fp_kwargs))
                mse = F.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(loss) or not torch.isfinite(mse):
                    raise FloatingPointError(
                        "Non-finite FlatQuant LFQ objective during evaluation."
                    )
                total += float(loss)
                mse_total += float(mse)
                for slot in LFQ_SLOT_NAMES:
                    slot_totals[slot] += float(slot_losses[slot])
                if boundary_loss is not None:
                    boundary_total += float(boundary_loss)
                    for slot in LFQ_SLOT_NAMES:
                        boundary_slot_totals[slot] += float(
                            boundary_slot_losses[slot]
                        )
        if was_training:
            train_block.train()
        count = len(quant_batches)
        return (
            total / count,
            {slot: slot_totals[slot] / count for slot in LFQ_SLOT_NAMES},
            boundary_total / count if boundaries is not None else None,
            (
                {
                    slot: boundary_slot_totals[slot] / count
                    for slot in LFQ_SLOT_NAMES
                }
                if boundaries is not None
                else {}
            ),
            mse_total / count,
        )

    wrappers = _flat_wrappers(train_block.block)
    all_transform_parameters = list(train_block.transforms.parameters())
    for parameter in all_transform_parameters:
        parameter.requires_grad_(config.learn_transform)
    transform_parameters = (
        all_transform_parameters if config.learn_transform else []
    )
    lwc_parameters = [
        parameter
        for wrapper in wrappers.values()
        for parameter in wrapper.lwc_parameters()
        if config.use_lwc
    ]
    lac_parameters = [
        parameter
        for quantizer in train_block.activation_quantizers.values()
        for parameter in quantizer.clip_parameters()
        if config.use_lac and config.learn_lac
    ]
    parameter_groups: list[dict[str, Any]] = []
    transform_group_index: int | None = None
    lwc_group_index: int | None = None
    lac_group_index: int | None = None
    if transform_parameters:
        transform_group_index = len(parameter_groups)
        parameter_groups.append(
            {"params": transform_parameters, "lr": config.transform_lr}
        )
    if lwc_parameters:
        lwc_group_index = len(parameter_groups)
        parameter_groups.append({"params": lwc_parameters, "lr": config.lwc_lr})
    if lac_parameters:
        lac_group_index = len(parameter_groups)
        parameter_groups.append({"params": lac_parameters, "lr": config.lac_lr})
    if not parameter_groups:
        raise ValueError("FlatQuant LFQ requires at least one trainable parameter group.")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs * len(train_quant_inputs),
        eta_min=config.transform_lr * config.min_lr_factor,
    )
    trainable_parameters = transform_parameters + lwc_parameters + lac_parameters

    initial = evaluate(
        train_fp_inputs,
        train_quant_inputs,
        train_probabilities,
        train_boundaries,
    )
    initial_validation = (
        evaluate(
            validation_fp_inputs,
            validation_quant_inputs,
            validation_probabilities,
            validation_boundaries,
        )
        if validation_quant_inputs
        else None
    )
    initial_slot_text = ",".join(
        f"{slot}:{initial[1][slot]:.6e}" for slot in LFQ_SLOT_NAMES
    )
    validation_text = ""
    if initial_validation is not None:
        validation_slot_text = ",".join(
            f"{slot}:{initial_validation[1][slot]:.6e}"
            for slot in LFQ_SLOT_NAMES
        )
        validation_text = (
            f" heldout_loss={initial_validation[0]:.6e}"
            f" heldout_slot={validation_slot_text}"
            f" heldout_mse={initial_validation[4]:.6e}"
        )
    print(
        f"[flatquant][lfq_initial] layer={layer_idx} loss={initial[0]:.6e} "
        f"slot={initial_slot_text} mse={initial[4]:.6e}"
        f"{validation_text} learn_transform={config.learn_transform}"
    )

    def snapshot_parameters() -> tuple[torch.Tensor, ...]:
        return tuple(parameter.detach().cpu().clone() for parameter in trainable_parameters)

    def restore_parameters(snapshot: Sequence[torch.Tensor]) -> None:
        if len(snapshot) != len(trainable_parameters):
            raise ValueError("FlatQuant LFQ best-state parameter count mismatch.")
        with torch.no_grad():
            for parameter, saved in zip(trainable_parameters, snapshot):
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))

    metrics: list[FlatQuantEpochMetric] = [
        FlatQuantEpochMetric(
            epoch=0,
            train_mse=None,
            eval_mse=initial[4],
            mean_grad_norm=None,
            max_grad_norm=None,
            transform_lr=config.transform_lr if transform_parameters else None,
            lwc_lr=config.lwc_lr if lwc_parameters else None,
            lac_lr=config.lac_lr if lac_parameters else None,
            validation_mse=(
                initial_validation[4] if initial_validation is not None else None
            ),
            train_loss=None,
            eval_loss=initial[0],
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
            train_lfq_boundary_loss=initial[2],
            validation_lfq_boundary_loss=(
                initial_validation[2] if initial_validation is not None else None
            ),
        )
    ]
    best_epoch = 0
    best_loss = initial[0]
    best_snapshot = snapshot_parameters()
    checked_gradients = False
    train_block.train()
    for epoch in range(1, config.epochs + 1):
        loss_total = 0.0
        boundary_total = 0.0
        grad_total = 0.0
        grad_max = 0.0
        for batch_idx, (quant_batch, teacher_probabilities) in enumerate(
            zip(train_quant_inputs, train_probabilities)
        ):
            boundary_targets = (
                None if train_boundaries is None else train_boundaries[batch_idx]
            )
            loss, _slot_losses, boundary_loss, _boundary_slots, _prediction = (
                student_objective(
                    quant_batch,
                    teacher_probabilities,
                    boundary_targets,
                )
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite FlatQuant LFQ loss before optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not checked_gradients:
                checked_gradients = True
                for group_name, group_parameters in (
                    ("transform", transform_parameters),
                    ("LWC", lwc_parameters),
                    ("LAC", lac_parameters),
                ):
                    if group_parameters and not any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in group_parameters
                    ):
                        raise RuntimeError(
                            f"All FlatQuant LFQ {group_name} gradients are zero."
                        )
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable_parameters
            ):
                raise FloatingPointError("Non-finite FlatQuant LFQ gradient.")
            grad_tensor = (
                _total_grad_norm(trainable_parameters)
                if config.max_grad_norm is None
                else torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config.max_grad_norm,
                )
            )
            grad_norm = float(grad_tensor.detach())
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Non-finite FlatQuant LFQ gradient norm.")
            optimizer.step()
            scheduler.step()
            if any(not torch.isfinite(parameter).all() for parameter in trainable_parameters):
                raise FloatingPointError("Non-finite FlatQuant LFQ parameter.")
            loss_total += float(loss.detach())
            if boundary_loss is not None:
                boundary_total += float(boundary_loss.detach())
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)

        should_evaluate = config.epoch_eval_interval > 0 and (
            epoch % config.epoch_eval_interval == 0 or epoch == config.epochs
        )
        evaluated = (
            evaluate(
                train_fp_inputs,
                train_quant_inputs,
                train_probabilities,
                train_boundaries,
            )
            if should_evaluate
            else None
        )
        validation_evaluated = (
            evaluate(
                validation_fp_inputs,
                validation_quant_inputs,
                validation_probabilities,
                validation_boundaries,
            )
            if should_evaluate and validation_quant_inputs
            else None
        )
        if evaluated is not None and evaluated[0] < best_loss:
            best_epoch = epoch
            best_loss = evaluated[0]
            best_snapshot = snapshot_parameters()
        transform_lr = (
            float(optimizer.param_groups[transform_group_index]["lr"])
            if transform_group_index is not None
            else None
        )
        current_lwc_lr = (
            float(optimizer.param_groups[lwc_group_index]["lr"])
            if lwc_group_index is not None
            else None
        )
        current_lac_lr = (
            float(optimizer.param_groups[lac_group_index]["lr"])
            if lac_group_index is not None
            else None
        )
        metric = FlatQuantEpochMetric(
            epoch=epoch,
            train_mse=None,
            eval_mse=evaluated[4] if evaluated is not None else None,
            mean_grad_norm=grad_total / len(train_quant_inputs),
            max_grad_norm=grad_max,
            transform_lr=transform_lr,
            lwc_lr=current_lwc_lr,
            lac_lr=current_lac_lr,
            validation_mse=(
                validation_evaluated[4]
                if validation_evaluated is not None
                else None
            ),
            train_loss=loss_total / len(train_quant_inputs),
            eval_loss=evaluated[0] if evaluated is not None else None,
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
            train_lfq_boundary_loss=(
                boundary_total / len(train_quant_inputs)
                if train_boundaries is not None
                else None
            ),
            validation_lfq_boundary_loss=(
                validation_evaluated[2]
                if validation_evaluated is not None
                else None
            ),
        )
        metrics.append(metric)
        eval_text = (
            ""
            if evaluated is None
            else f" eval_loss={evaluated[0]:.6e} eval_mse={evaluated[4]:.6e}"
        )
        heldout_text = (
            ""
            if validation_evaluated is None
            else (
                f" heldout_loss={validation_evaluated[0]:.6e}"
                f" heldout_mse={validation_evaluated[4]:.6e}"
            )
        )
        print(
            f"[flatquant][lfq_epoch] layer={layer_idx} "
            f"epoch={epoch}/{config.epochs} train_loss={metric.train_loss:.6e} "
            f"grad_norm_mean={metric.mean_grad_norm:.6e} "
            f"grad_norm_max={metric.max_grad_norm:.6e}"
            f"{eval_text}{heldout_text}"
        )
        train_block.train()

    if config.epoch_eval_interval > 0:
        restore_parameters(best_snapshot)
        selected_epoch: int | None = best_epoch
    else:
        selected_epoch = None
    train_block.eval()
    final = evaluate(
        train_fp_inputs,
        train_quant_inputs,
        train_probabilities,
        train_boundaries,
    )
    final_validation = (
        evaluate(
            validation_fp_inputs,
            validation_quant_inputs,
            validation_probabilities,
            validation_boundaries,
        )
        if validation_quant_inputs
        else None
    )
    if config.epoch_eval_interval == 0:
        last = metrics[-1]
        metrics[-1] = FlatQuantEpochMetric(
            epoch=last.epoch,
            train_mse=last.train_mse,
            eval_mse=final[4],
            mean_grad_norm=last.mean_grad_norm,
            max_grad_norm=last.max_grad_norm,
            transform_lr=last.transform_lr,
            lwc_lr=last.lwc_lr,
            lac_lr=last.lac_lr,
            validation_mse=(
                final_validation[4] if final_validation is not None else None
            ),
            train_loss=last.train_loss,
            eval_loss=final[0],
            validation_loss=(
                final_validation[0] if final_validation is not None else None
            ),
            validation_lfq_slot_losses=(
                tuple(
                    (slot, final_validation[1][slot])
                    for slot in LFQ_SLOT_NAMES
                )
                if final_validation is not None
                else ()
            ),
            train_lfq_boundary_loss=last.train_lfq_boundary_loss,
            validation_lfq_boundary_loss=(
                final_validation[2] if final_validation is not None else None
            ),
        )
    final_slot_text = ",".join(
        f"{slot}:{final[1][slot]:.6e}" for slot in LFQ_SLOT_NAMES
    )
    final_validation_text = ""
    if final_validation is not None:
        final_validation_text = (
            f" heldout_loss={final_validation[0]:.6e}"
            f" heldout_mse={final_validation[4]:.6e}"
        )
    print(
        f"[flatquant][lfq_final] layer={layer_idx} loss={final[0]:.6e} "
        f"slot={final_slot_text} mse={final[4]:.6e}"
        f"{final_validation_text} best_epoch={selected_epoch}"
    )
    return _FlatQuantLFQTrainResult(
        initial_loss=initial[0],
        final_loss=final[0],
        initial_slot_losses=dict(initial[1]),
        final_slot_losses=dict(final[1]),
        initial_boundary_loss=initial[2],
        final_boundary_loss=final[2],
        initial_boundary_slot_losses=dict(initial[3]),
        final_boundary_slot_losses=dict(final[3]),
        initial_mse=initial[4],
        final_mse=final[4],
        initial_validation_loss=(
            initial_validation[0] if initial_validation is not None else None
        ),
        final_validation_loss=(
            final_validation[0] if final_validation is not None else None
        ),
        initial_validation_slot_losses=(
            dict(initial_validation[1]) if initial_validation is not None else {}
        ),
        final_validation_slot_losses=(
            dict(final_validation[1]) if final_validation is not None else {}
        ),
        initial_validation_mse=(
            initial_validation[4] if initial_validation is not None else None
        ),
        final_validation_mse=(
            final_validation[4] if final_validation is not None else None
        ),
        best_epoch=selected_epoch,
        metrics=tuple(metrics),
    )


def _materialize_transform_bank(
    train_block: _TrainableFlatQuantBlock,
) -> nn.ModuleDict:
    if not _formal_transform_bank(train_block.transforms):
        frozen = copy.deepcopy(train_block.transforms)
        frozen.requires_grad_(False)
        return frozen

    materialized: dict[str, nn.Module] = {}
    for name, transform in train_block.transforms.items():
        if isinstance(transform, KroneckerSVDTransform):
            # All three diagonals are folded into adjacent modules before the
            # finalized block runs, so online transforms retain matrices only.
            materialized[name] = transform.materialize(use_diag=False)
        elif isinstance(transform, SingleSVDTransform):
            materialized[name] = transform.materialize()
        else:
            raise TypeError(
                f"Cannot materialize FlatQuant transform {name!r}: {type(transform)!r}"
            )
    frozen = nn.ModuleDict(materialized)
    frozen.requires_grad_(False)
    return frozen


def _fold_flatquant_norm_diagonal(
    source: nn.Module,
    train_block: _TrainableFlatQuantBlock,
    *,
    norm_name: str,
    transform_name: str,
) -> None:
    wrapped = source.get_submodule(norm_name)
    if not isinstance(wrapped, _TrainableFlatQuantScaledNorm):
        raise TypeError(f"Expected FlatQuant scaled norm wrapper for {norm_name}.")
    norm = copy.deepcopy(wrapped.base_norm)
    transform = _require_kronecker_transform(
        train_block.transforms,
        transform_name,
    )
    if transform.diag_scale is None:
        raise ValueError(f"FlatQuant transform {transform_name!r} has no diagonal.")
    scale = transform.diag_scale.detach()
    with torch.no_grad():
        weight = getattr(norm, "weight", None)
        if not torch.is_tensor(weight) or weight.numel() != scale.numel():
            raise ValueError(f"FlatQuant diagonal cannot fold into {norm_name}.")
        weight.mul_(scale.to(device=weight.device, dtype=weight.dtype))
        bias = getattr(norm, "bias", None)
        if torch.is_tensor(bias):
            bias.mul_(scale.to(device=bias.device, dtype=bias.dtype))
    setattr(source, norm_name, norm)


def _finalize_flat_block(
    train_block: _TrainableFlatQuantBlock,
) -> tuple[nn.Module, int, int, int]:
    """Materialize official transforms and static QDQ in the current framework."""
    source = copy.deepcopy(train_block.block)
    final_transforms = _materialize_transform_bank(train_block)
    final_activation_quantizers = copy.deepcopy(train_block.activation_quantizers)
    final_activation_quantizers.requires_grad_(False)
    source_wrappers = _flat_wrappers(source)
    train_wrappers = _flat_wrappers(train_block.block)
    if set(source_wrappers) != set(train_wrappers):
        raise ValueError("FlatQuant train/final wrapper sets do not match.")

    if _formal_transform_bank(train_block.transforms):
        _fold_flatquant_norm_diagonal(
            source,
            train_block,
            norm_name="input_layernorm",
            transform_name="attn_in",
        )
        _fold_flatquant_norm_diagonal(
            source,
            train_block,
            norm_name="post_attention_layernorm",
            transform_name="mlp_in",
        )

    # Register one shared frozen bank on the finalized block. Static Linear
    # wrappers keep non-registering references to avoid seven duplicated copies.
    source.add_module("flatquant_transforms", final_transforms)
    source.add_module(
        "flatquant_activation_quantizers",
        final_activation_quantizers,
    )

    replaced = 0
    with torch.no_grad():
        for name, source_wrapper in source_wrappers.items():
            train_wrapper = train_wrappers[name]
            weight_qdq = train_wrapper.finalize_qdq_weight(
                train_wrapper.transformed_weight()
            ).to(dtype=train_wrapper.execution_dtype)
            bias = train_wrapper.transformed_bias()
            if bias is not None:
                bias = bias.to(dtype=train_wrapper.execution_dtype)
            parent_name, child_name = name.rsplit(".", 1)
            setattr(
                source.get_submodule(parent_name),
                child_name,
                FlatQuantCoreLinear(
                    weight_qdq=weight_qdq,
                    bias=bias,
                    transform_bank=final_transforms,
                    activation_quantizer_bank=final_activation_quantizers,
                    role=train_wrapper.role,
                    head_dim=train_wrapper.head_dim,
                    config=train_block.config,
                ),
            )
            replaced += 1
    # The copied block and static QDQ weights already use inference_dtype.
    # Keep frozen transform/LAC state in FP32; their forwards cast matrices or
    # thresholds at the numerical boundary instead of storing low-precision state.
    shared_attention = 0
    shared_mlp = 0
    if train_block.act_quant_mode == "shared_input":
        shared_attention, shared_mlp = install_shared_input_activation_quantization(source)
    return source, replaced, shared_attention, shared_mlp


def _restore_lwc_state(
    train_block: _TrainableFlatQuantBlock,
    saved_state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    wrappers = _flat_wrappers(train_block.block)
    if set(saved_state) != set(wrappers):
        raise ValueError(f"FlatQuant LWC wrapper names do not match {checkpoint_path}.")
    with torch.no_grad():
        for name, wrapper in wrappers.items():
            item = saved_state[name]
            if not isinstance(item, Mapping):
                raise TypeError(f"Invalid LWC state for {name!r} in {checkpoint_path}.")
            current = wrapper.lwc_state()
            if set(item) != set(current):
                raise ValueError(f"FlatQuant LWC keys do not match for {name!r}.")
            for parameter_name, expected in current.items():
                parameter = getattr(wrapper, parameter_name)
                saved = item[parameter_name]
                if not torch.is_tensor(saved) or tuple(saved.shape) != tuple(expected.shape):
                    raise ValueError(f"FlatQuant LWC tensor shape mismatch for {name!r}.")
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def _validate_checkpoint_config(
    state: Mapping[str, Any],
    *,
    config: FlatQuantCoreConfig,
    checkpoint_path: Path,
) -> None:
    saved = state.get("config")
    if not isinstance(saved, Mapping):
        raise TypeError(f"Missing FlatQuant config in {checkpoint_path}.")
    expected = asdict(config)
    for key in (
        "weight_quant_format",
        "activation_quant_format",
        "weight_quant_scheme",
        "weight_group_size",
        "use_lwc",
        "use_lac",
        "init_lwc_logit",
        "init_lac_logit",
        "eps",
    ):
        if saved.get(key) != expected[key]:
            raise ValueError(
                f"FlatQuant checkpoint {checkpoint_path} has {key}={saved.get(key)!r}, "
                f"expected {expected[key]!r}."
            )
    if saved.get("transform_init") != expected["transform_init"]:
        raise ValueError(
            f"FlatQuant checkpoint {checkpoint_path} has "
            f"transform_init={saved.get('transform_init')!r}, "
            f"expected {expected['transform_init']!r}."
        )
    saved_transform_kind = saved.get("transform_kind", "kronecker")
    if saved_transform_kind != config.transform_kind:
        raise ValueError(
            f"FlatQuant checkpoint {checkpoint_path} has "
            f"transform_kind={saved_transform_kind!r}, "
            f"expected {config.transform_kind!r}."
        )
    if saved.get("diag_alpha", 0.5) != config.diag_alpha:
        raise ValueError(
            f"FlatQuant checkpoint {checkpoint_path} has "
            f"diag_alpha={saved.get('diag_alpha')!r}, "
            f"expected {config.diag_alpha!r}."
        )
    if config.transform_kind == "smoothquant" and saved.get(
        "smoothquant_alpha",
        DEFAULT_SMOOTHQUANT_ALPHA,
    ) != config.smoothquant_alpha:
        raise ValueError(
            f"FlatQuant checkpoint {checkpoint_path} has "
            f"smoothquant_alpha={saved.get('smoothquant_alpha')!r}, "
            f"expected {config.smoothquant_alpha!r}."
        )
    # learn_transform is a calibration-time choice.  A checkpoint trained with
    # matrices enabled may be restored for an inference-only load or used to
    # initialize a branch that freezes those matrices.


def _load_flatquant_checkpoint(
    checkpoint_path: Path,
    *,
    layer_idx: int,
) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing FlatQuant checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or state.get("method") != "flatquant_core":
        raise TypeError(f"Invalid FlatQuant checkpoint: {checkpoint_path}")
    if int(state.get("checkpoint_schema", -1)) != 2:
        raise ValueError(
            f"Unsupported preliminary FlatQuant checkpoint schema in "
            f"{checkpoint_path}; expected schema 2."
        )
    if int(state.get("layer_idx", -1)) != layer_idx:
        raise ValueError(f"FlatQuant checkpoint layer mismatch: {checkpoint_path}")
    return state


def _restore_flatquant_train_state(
    train_block: _TrainableFlatQuantBlock,
    state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    transform_state = state.get("transform_state_dict")
    if not isinstance(transform_state, Mapping):
        raise TypeError(f"Missing transform state in {checkpoint_path}.")
    train_block.transforms.load_state_dict(transform_state, strict=True)
    activation_state = state.get("activation_quantizer_state_dict")
    if not isinstance(activation_state, Mapping):
        raise TypeError(f"Missing activation quantizer state in {checkpoint_path}.")
    train_block.activation_quantizers.load_state_dict(
        activation_state,
        strict=True,
    )
    lwc_state = state.get("lwc_parameters")
    if not isinstance(lwc_state, Mapping):
        raise TypeError(f"Missing LWC state in {checkpoint_path}.")
    _restore_lwc_state(train_block, lwc_state, checkpoint_path=checkpoint_path)


def _summary_from_block(
    train_block: _TrainableFlatQuantBlock,
    *,
    replaced: int,
    initial_mse: float,
    final_mse: float,
    initial_validation_mse: float | None,
    final_validation_mse: float | None,
    best_epoch: int | None,
    metrics: tuple[FlatQuantEpochMetric, ...],
    shared_attention: int,
    shared_mlp: int,
    objective: Literal["mse", "lfq_ce"] = "mse",
    initial_loss: float | None = None,
    final_loss: float | None = None,
    initial_slot_losses: Mapping[str, float] | None = None,
    final_slot_losses: Mapping[str, float] | None = None,
    initial_boundary_loss: float | None = None,
    final_boundary_loss: float | None = None,
    initial_boundary_slot_losses: Mapping[str, float] | None = None,
    final_boundary_slot_losses: Mapping[str, float] | None = None,
    initial_validation_loss: float | None = None,
    final_validation_loss: float | None = None,
    initial_validation_slot_losses: Mapping[str, float] | None = None,
    final_validation_slot_losses: Mapping[str, float] | None = None,
) -> FlatQuantCoreSummary:
    factor_items: list[tuple[str, int, int]] = []
    for name, transform in train_block.transforms.items():
        if isinstance(
            transform,
            (KroneckerSVDTransform, FixedSmoothQuantTransform),
        ):
            factor_items.append((name, transform.left_size, transform.right_size))
        elif isinstance(transform, SingleSVDTransform):
            factor_items.append((name, transform.size, 1))
    factors = tuple(factor_items)
    effective = sum(
        transform.effective_matrix_parameters
        for transform in train_block.transforms.values()
        if isinstance(
            transform,
            (
                KroneckerSVDTransform,
                SingleSVDTransform,
                FixedSmoothQuantTransform,
            ),
        )
    )
    transform_parameters = (
        sum(parameter.numel() for parameter in train_block.transforms.parameters())
        if train_block.config.learn_transform
        else 0
    )
    lwc_parameters = sum(
        parameter.numel()
        for wrapper in _flat_wrappers(train_block.block).values()
        for parameter in wrapper.lwc_parameters()
        if train_block.config.use_lwc
    )
    lac_parameters = sum(
        parameter.numel()
        for quantizer in train_block.activation_quantizers.values()
        for parameter in quantizer.clip_parameters()
        if train_block.config.use_lac and train_block.config.learn_lac
    )
    normalized_initial_loss = initial_mse if initial_loss is None else initial_loss
    normalized_final_loss = final_mse if final_loss is None else final_loss
    return FlatQuantCoreSummary(
        replaced_linears=replaced,
        initial_mse_loss=initial_mse,
        final_mse_loss=final_mse,
        transform_factors=factors,
        effective_transform_parameters=effective,
        trainable_transform_parameters=transform_parameters,
        trainable_lwc_parameters=lwc_parameters,
        trainable_lac_parameters=lac_parameters,
        epoch_metrics=metrics,
        initial_validation_mse_loss=initial_validation_mse,
        final_validation_mse_loss=final_validation_mse,
        best_epoch=best_epoch,
        shared_attention_modules=shared_attention,
        shared_mlp_modules=shared_mlp,
        objective=objective,
        initial_loss=normalized_initial_loss,
        final_loss=normalized_final_loss,
        initial_lfq_slot_losses=tuple((slot, float((initial_slot_losses or {})[slot])) for slot in LFQ_SLOT_NAMES if slot in (initial_slot_losses or {})),
        final_lfq_slot_losses=tuple((slot, float((final_slot_losses or {})[slot])) for slot in LFQ_SLOT_NAMES if slot in (final_slot_losses or {})),
        lfq_slot_weights=(
            tuple(train_block.config.lfq_slot_weights)
            if objective == "lfq_ce"
            else None
        ),
        lfq_loss_weight=(
            train_block.config.lfq_loss_weight if objective == "lfq_ce" else 1.0
        ),
        lfq_boundary_loss_weight=(
            train_block.config.lfq_boundary_loss_weight
            if objective == "lfq_ce"
            else 0.0
        ),
        initial_lfq_boundary_loss=initial_boundary_loss,
        final_lfq_boundary_loss=final_boundary_loss,
        initial_lfq_boundary_slot_losses=tuple((slot, float((initial_boundary_slot_losses or {})[slot])) for slot in LFQ_SLOT_NAMES if slot in (initial_boundary_slot_losses or {})),
        final_lfq_boundary_slot_losses=tuple((slot, float((final_boundary_slot_losses or {})[slot])) for slot in LFQ_SLOT_NAMES if slot in (final_boundary_slot_losses or {})),
        initial_validation_loss=initial_validation_loss,
        final_validation_loss=final_validation_loss,
        initial_validation_lfq_slot_losses=tuple((slot, float((initial_validation_slot_losses or {})[slot])) for slot in LFQ_SLOT_NAMES if slot in (initial_validation_slot_losses or {})),
        final_validation_lfq_slot_losses=tuple((slot, float((final_validation_slot_losses or {})[slot])) for slot in LFQ_SLOT_NAMES if slot in (final_validation_slot_losses or {})),
    )


def apply_flatquant_core_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    config: FlatQuantCoreConfig,
    capture_layer_input_batches: Any,
    act_quant_mode: ActQuantMode = "shared_input",
    checkpoint_dir: str | Path | None = None,
    prefix_checkpoint_dir: str | Path | None = None,
    finetune_checkpoint_dir: str | Path | None = None,
    lfq_token_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[int, FlatQuantCoreSummary]:
    """Train selected blocks or fine-tune the final block from one full checkpoint."""

    config.validate()
    layers = (
        model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model.layers
    )
    selected = sorted(layer_indices)
    final_layer_idx = len(layers) - 1
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    prefix_checkpoint_root = (
        None if prefix_checkpoint_dir is None else Path(prefix_checkpoint_dir)
    )
    finetune_checkpoint_root = (
        None if finetune_checkpoint_dir is None else Path(finetune_checkpoint_dir)
    )
    if prefix_checkpoint_root is not None and finetune_checkpoint_root is not None:
        raise ValueError(
            "FlatQuant prefix and full fine-tune checkpoint directories are mutually exclusive."
        )
    source_root = prefix_checkpoint_root or finetune_checkpoint_root
    if source_root is not None:
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"FlatQuant source checkpoint directory does not exist: {source_root}"
            )
        if selected != list(range(len(layers))):
            raise ValueError(
                "FlatQuant checkpoint-initialized training requires all Transformer "
                "layers so FP and quantized streams remain aligned."
            )
        if checkpoint_root is not None and checkpoint_root.resolve() == source_root.resolve():
            raise ValueError("FlatQuant source and output checkpoint directories must differ.")
    if prefix_checkpoint_root is not None:
        print(
            f"[flatquant] prefix checkpoint enabled layers=0-{final_layer_idx - 1} "
            f"source={prefix_checkpoint_root}"
        )
    if finetune_checkpoint_root is not None:
        print(
            f"[flatquant] full-checkpoint fine-tune enabled source={finetune_checkpoint_root} "
            f"train_layer={final_layer_idx} objective={config.final_objective}"
        )
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    lfq_projector: _LFQOutputProjector | None = None
    if config.final_objective == "lfq_ce":
        if final_layer_idx not in selected:
            raise ValueError(
                f"FlatQuant LFQ requires final Transformer layer {final_layer_idx}."
            )
        backbone = getattr(model, "model", None)
        final_norm = getattr(backbone, "norm", None)
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        output_head = get_output_embeddings() if callable(get_output_embeddings) else None
        if not isinstance(final_norm, nn.Module) or not isinstance(output_head, nn.Module):
            raise TypeError("FlatQuant LFQ requires model.model.norm and the output head.")
        if lfq_token_ids is None or set(lfq_token_ids) != set(LFQ_SLOT_NAMES):
            raise ValueError("FlatQuant LFQ requires SID_a/SID_b/SID_c token IDs.")
        lfq_projector = _LFQOutputProjector(
            final_norm=final_norm,
            output_head=output_head,
            token_ids=lfq_token_ids,
        )
        normalized_weights = _normalized_flat_lfq_slot_weights(config)
        print(
            f"[flatquant] LFQ enabled final_layer={final_layer_idx} "
            f"weights="
            + ",".join(
                f"{slot}:{normalized_weights[slot]:.6g}" for slot in LFQ_SLOT_NAMES
            )
            + f" ce_weight={config.lfq_loss_weight:.6g} "
            f"boundary_weight={config.lfq_boundary_loss_weight:.6g}"
        )

    summaries: dict[int, FlatQuantCoreSummary] = {}
    fp_inputs: list[Batch] | None = None
    quant_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    for layer_idx in selected:
        if fp_inputs is None:
            captured = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
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
        objective: Literal["mse", "lfq_ce"] = (
            config.final_objective if layer_idx == final_layer_idx else "mse"
        )
        source_checkpoint: Path | None = None
        source_state: Mapping[str, Any] | None = None
        restore_only = False
        initialize_from_full_checkpoint = False
        if prefix_checkpoint_root is not None and layer_idx < final_layer_idx:
            source_checkpoint = prefix_checkpoint_root / f"layer_{layer_idx:02d}.pt"
            restore_only = True
        elif finetune_checkpoint_root is not None:
            source_checkpoint = finetune_checkpoint_root / f"layer_{layer_idx:02d}.pt"
            restore_only = layer_idx < final_layer_idx
            initialize_from_full_checkpoint = layer_idx == final_layer_idx

        initial_slot_losses: dict[str, float] = {}
        final_slot_losses: dict[str, float] = {}
        initial_boundary_loss: float | None = None
        final_boundary_loss: float | None = None
        initial_boundary_slot_losses: dict[str, float] = {}
        final_boundary_slot_losses: dict[str, float] = {}
        initial_validation_loss: float | None = None
        final_validation_loss: float | None = None
        initial_validation_slot_losses: dict[str, float] = {}
        final_validation_slot_losses: dict[str, float] = {}

        if source_checkpoint is not None:
            source_state = _load_flatquant_checkpoint(source_checkpoint, layer_idx=layer_idx)
            _validate_checkpoint_config(
                source_state,
                config=config,
                checkpoint_path=source_checkpoint,
            )
            saved_mode = source_state.get("act_quant_mode", "shared_input")
            if saved_mode != act_quant_mode:
                raise ValueError(
                    f"FlatQuant source checkpoint uses act_quant_mode={saved_mode!r}, "
                    f"expected {act_quant_mode!r}."
                )
            train_block = _TrainableFlatQuantBlock(
                copy.deepcopy(teacher_block),
                config=config,
                act_quant_mode=act_quant_mode,
            )
            _restore_flatquant_train_state(
                train_block,
                source_state,
                checkpoint_path=source_checkpoint,
            )
            if initialize_from_full_checkpoint:
                saved_config = source_state.get("config", {})
                saved_objective = (
                    str(saved_config.get("final_objective", "mse"))
                    if isinstance(saved_config, Mapping)
                    else "mse"
                )
                if saved_objective != "mse":
                    raise ValueError(
                        "FlatQuant fine-tune initialization must be an MSE checkpoint."
                    )
                print(
                    f"[flatquant][finetune_init] layer={layer_idx} "
                    f"source={source_checkpoint} objective={objective}"
                )
        else:
            init_scales: dict[str, torch.Tensor] = {}
            if config.transform_kind == "smoothquant":
                init_scales = _collect_fixed_smoothquant_transform_scales(
                    teacher_block,
                    fp_inputs,
                    config=config,
                )
                label = "smoothquant"
            else:
                init_scales = _collect_flatquant_diag_initials(
                    teacher_block,
                    fp_inputs,
                    config=config,
                )
                label = "diag_init"
            train_count, _validation_start, _validation_count = _flat_split_counts(
                len(fp_inputs),
                config,
            )
            print(
                f"[flatquant][{label}] layer={layer_idx} "
                f"statistics_samples={train_count} heldout_leakage=false"
            )
            train_block = _TrainableFlatQuantBlock(
                copy.deepcopy(teacher_block),
                config=config,
                act_quant_mode=act_quant_mode,
                init_scales=init_scales,
            )

        if restore_only:
            assert source_state is not None and source_checkpoint is not None
            initial_mse = float(source_state["initial_mse_loss"])
            final_mse = float(source_state["final_mse_loss"])
            initial_validation_mse = (
                None
                if source_state.get("initial_validation_mse_loss") is None
                else float(source_state["initial_validation_mse_loss"])
            )
            final_validation_mse = (
                None
                if source_state.get("final_validation_mse_loss") is None
                else float(source_state["final_validation_mse_loss"])
            )
            initial_loss = float(source_state.get("initial_loss", initial_mse))
            final_loss = float(source_state.get("final_loss", final_mse))
            best_epoch = (
                None
                if source_state.get("best_epoch") is None
                else int(source_state["best_epoch"])
            )
            metrics = tuple(
                FlatQuantEpochMetric(**dict(metric))
                for metric in source_state.get("epoch_metrics", ())
                if isinstance(metric, Mapping)
            )
            print(f"[flatquant][source] layer={layer_idx} source={source_checkpoint}")
        elif objective == "lfq_ce":
            if lfq_projector is None:
                raise RuntimeError("FlatQuant LFQ projector was not initialized.")
            result = _train_flat_block_lfq(
                teacher_block=teacher_block,
                train_block=train_block,
                fp_inputs=fp_inputs,
                quant_inputs=quant_inputs,
                config=config,
                lfq_projector=lfq_projector,
                layer_idx=layer_idx,
            )
            initial_loss = result.initial_loss
            final_loss = result.final_loss
            initial_slot_losses = result.initial_slot_losses
            final_slot_losses = result.final_slot_losses
            initial_boundary_loss = result.initial_boundary_loss
            final_boundary_loss = result.final_boundary_loss
            initial_boundary_slot_losses = result.initial_boundary_slot_losses
            final_boundary_slot_losses = result.final_boundary_slot_losses
            initial_mse = result.initial_mse
            final_mse = result.final_mse
            initial_validation_loss = result.initial_validation_loss
            final_validation_loss = result.final_validation_loss
            initial_validation_slot_losses = result.initial_validation_slot_losses
            final_validation_slot_losses = result.final_validation_slot_losses
            initial_validation_mse = result.initial_validation_mse
            final_validation_mse = result.final_validation_mse
            best_epoch = result.best_epoch
            metrics = result.metrics
        else:
            (
                initial_mse,
                final_mse,
                initial_validation_mse,
                final_validation_mse,
                best_epoch,
                metrics,
            ) = _train_flat_block(
                teacher_block=teacher_block,
                train_block=train_block,
                fp_inputs=fp_inputs,
                quant_inputs=quant_inputs,
                config=config,
                layer_idx=layer_idx,
            )
            initial_loss = initial_mse
            final_loss = final_mse

        next_fp_inputs = _advance_cpu(teacher_block, fp_inputs)
        final_block, replaced, shared_attention, shared_mlp = _finalize_flat_block(
            train_block
        )
        layers[layer_idx] = final_block
        quant_inputs = _advance_cpu(final_block, quant_inputs)
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        summary = _summary_from_block(
            train_block,
            replaced=replaced,
            initial_mse=initial_mse,
            final_mse=final_mse,
            initial_validation_mse=initial_validation_mse,
            final_validation_mse=final_validation_mse,
            best_epoch=best_epoch,
            metrics=metrics,
            shared_attention=shared_attention,
            shared_mlp=shared_mlp,
            objective=objective,
            initial_loss=initial_loss,
            final_loss=final_loss,
            initial_slot_losses=initial_slot_losses,
            final_slot_losses=final_slot_losses,
            initial_boundary_loss=initial_boundary_loss,
            final_boundary_loss=final_boundary_loss,
            initial_boundary_slot_losses=initial_boundary_slot_losses,
            final_boundary_slot_losses=final_boundary_slot_losses,
            initial_validation_loss=initial_validation_loss,
            final_validation_loss=final_validation_loss,
            initial_validation_slot_losses=initial_validation_slot_losses,
            final_validation_slot_losses=final_validation_slot_losses,
        )
        summaries[layer_idx] = summary

        if checkpoint_root is not None:
            if restore_only:
                assert source_state is not None and source_checkpoint is not None
                checkpoint_state = dict(source_state)
                checkpoint_state["source_checkpoint"] = str(source_checkpoint.resolve())
            else:
                checkpoint_state = {
                    "method": "flatquant_core",
                    "checkpoint_schema": 2,
                    "layer_idx": layer_idx,
                    "config": asdict(config),
                    "act_quant_mode": act_quant_mode,
                    "transform_state_dict": {
                        key: value.detach().cpu()
                        for key, value in train_block.transforms.state_dict().items()
                    },
                    "activation_quantizer_state_dict": {
                        key: value.detach().cpu()
                        for key, value in train_block.activation_quantizers.state_dict().items()
                    },
                    "lwc_parameters": {
                        name: wrapper.lwc_state()
                        for name, wrapper in _flat_wrappers(train_block.block).items()
                    },
                    "objective": objective,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "initial_lfq_slot_losses": dict(initial_slot_losses),
                    "final_lfq_slot_losses": dict(final_slot_losses),
                    "initial_lfq_boundary_loss": initial_boundary_loss,
                    "final_lfq_boundary_loss": final_boundary_loss,
                    "initial_lfq_boundary_slot_losses": dict(initial_boundary_slot_losses),
                    "final_lfq_boundary_slot_losses": dict(final_boundary_slot_losses),
                    "initial_validation_loss": initial_validation_loss,
                    "final_validation_loss": final_validation_loss,
                    "initial_validation_lfq_slot_losses": dict(initial_validation_slot_losses),
                    "final_validation_lfq_slot_losses": dict(final_validation_slot_losses),
                    "initial_mse_loss": initial_mse,
                    "final_mse_loss": final_mse,
                    "initial_validation_mse_loss": initial_validation_mse,
                    "final_validation_mse_loss": final_validation_mse,
                    "best_epoch": best_epoch,
                    "epoch_metrics": [asdict(metric) for metric in metrics],
                    "transform_factors": summary.transform_factors,
                    "effective_transform_parameters": summary.effective_transform_parameters,
                    "trainable_transform_parameters": summary.trainable_transform_parameters,
                    "trainable_lwc_parameters": summary.trainable_lwc_parameters,
                    "trainable_lac_parameters": summary.trainable_lac_parameters,
                }
                if initialize_from_full_checkpoint and source_checkpoint is not None:
                    checkpoint_state["initialization_checkpoint"] = str(
                        source_checkpoint.resolve()
                    )
            torch.save(checkpoint_state, checkpoint_root / f"layer_{layer_idx:02d}.pt")

        factor_text = ",".join(
            f"{name}:{left}x{right}" for name, left, right in summary.transform_factors
        )
        objective_text = (
            f"loss={initial_loss:.6e}->{final_loss:.6e} "
            if objective == "lfq_ce"
            else ""
        )
        print(
            f"[flatquant] layer={layer_idx} replaced_linears={replaced} "
            f"objective={objective} {objective_text}"
            f"mse={initial_mse:.6e}->{final_mse:.6e} "
            f"best_epoch={best_epoch} transform_kind={config.transform_kind} "
            f"factors={factor_text} weight_group_size={config.weight_group_size or 'per_channel'}"
        )
    return summaries


def restore_flatquant_core_layers_from_checkpoints(
    *,
    model: nn.Module,
    layer_indices: Sequence[int],
    config: FlatQuantCoreConfig,
    checkpoint_dir: str | Path,
    act_quant_mode: ActQuantMode = "shared_input",
) -> dict[int, FlatQuantCoreSummary]:
    """Restore learned matrices/LWC parameters and materialize inference blocks."""
    config.validate()
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"FlatQuant checkpoint directory not found: {checkpoint_root}")
    layers = (
        model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else model.layers
    )
    summaries: dict[int, FlatQuantCoreSummary] = {}
    for layer_idx in sorted(layer_indices):
        checkpoint_path = checkpoint_root / f"layer_{layer_idx:02d}.pt"
        state = _load_flatquant_checkpoint(
            checkpoint_path,
            layer_idx=layer_idx,
        )
        _validate_checkpoint_config(state, config=config, checkpoint_path=checkpoint_path)
        saved_mode = state.get("act_quant_mode", "shared_input")
        if saved_mode != act_quant_mode:
            raise ValueError(
                f"FlatQuant checkpoint uses act_quant_mode={saved_mode!r}, "
                f"expected {act_quant_mode!r}."
            )
        train_block = _TrainableFlatQuantBlock(
            copy.deepcopy(layers[layer_idx]),
            config=config,
            act_quant_mode=act_quant_mode,
        )
        _restore_flatquant_train_state(
            train_block,
            state,
            checkpoint_path=checkpoint_path,
        )
        final_block, replaced, shared_attention, shared_mlp = _finalize_flat_block(
            train_block
        )
        layers[layer_idx] = final_block
        raw_metrics = state.get("epoch_metrics", ())
        metrics = tuple(
            FlatQuantEpochMetric(**dict(metric))
            for metric in raw_metrics
            if isinstance(metric, Mapping)
        )
        initial_mse = float(state["initial_mse_loss"])
        final_mse = float(state["final_mse_loss"])
        objective = str(state.get("objective", "mse"))
        if objective not in ("mse", "lfq_ce"):
            raise ValueError(f"Invalid FlatQuant checkpoint objective: {objective!r}.")
        summary = _summary_from_block(
            train_block,
            replaced=replaced,
            initial_mse=initial_mse,
            final_mse=final_mse,
            initial_validation_mse=(
                None
                if state.get("initial_validation_mse_loss") is None
                else float(state["initial_validation_mse_loss"])
            ),
            final_validation_mse=(
                None
                if state.get("final_validation_mse_loss") is None
                else float(state["final_validation_mse_loss"])
            ),
            best_epoch=(
                None
                if state.get("best_epoch") is None
                else int(state["best_epoch"])
            ),
            metrics=metrics,
            shared_attention=shared_attention,
            shared_mlp=shared_mlp,
            objective=objective,
            initial_loss=float(state.get("initial_loss", initial_mse)),
            final_loss=float(state.get("final_loss", final_mse)),
            initial_slot_losses=state.get("initial_lfq_slot_losses", {}),
            final_slot_losses=state.get("final_lfq_slot_losses", {}),
            initial_boundary_loss=(
                None
                if state.get("initial_lfq_boundary_loss") is None
                else float(state["initial_lfq_boundary_loss"])
            ),
            final_boundary_loss=(
                None
                if state.get("final_lfq_boundary_loss") is None
                else float(state["final_lfq_boundary_loss"])
            ),
            initial_boundary_slot_losses=state.get(
                "initial_lfq_boundary_slot_losses", {}
            ),
            final_boundary_slot_losses=state.get(
                "final_lfq_boundary_slot_losses", {}
            ),
            initial_validation_loss=(
                None
                if state.get("initial_validation_loss") is None
                else float(state["initial_validation_loss"])
            ),
            final_validation_loss=(
                None
                if state.get("final_validation_loss") is None
                else float(state["final_validation_loss"])
            ),
            initial_validation_slot_losses=state.get(
                "initial_validation_lfq_slot_losses", {}
            ),
            final_validation_slot_losses=state.get(
                "final_validation_lfq_slot_losses", {}
            ),
        )
        summaries[layer_idx] = summary
        print(
            f"[flatquant][restore] layer={layer_idx} replaced_linears={replaced} "
            f"mse={summary.initial_mse_loss:.6e}->{summary.final_mse_loss:.6e}"
        )
    return summaries
