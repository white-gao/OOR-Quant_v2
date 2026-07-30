"""Blockwise OmniQuant-style calibration for Qwen3 decoder blocks.

The weight path supports both the repository's legacy symmetric signed LWC and
the paper-style asymmetric LWC with independently learned upper/lower clipping
factors and an integer zero point.  Both variants operate per output channel.
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
    FP8_MAX,
    QuantFormat,
    quant_format_qmax,
    require_fp8,
    validate_quant_format,
)
from ..support.runtime_utils import _module_device, _move_tree_to_device
from ..support.smoothquant_runtime import (
    Batch,
    _batch_to_args_kwargs,
    collect_smoothquant_scales,
)


@dataclass(frozen=True)
class OmniQuantConfig:
    """Calibration parameters for per-channel LWC and scale-only LET."""

    weight_quant_format: QuantFormat = "int4"
    activation_quant_format: QuantFormat = "none"
    weight_quant_scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    use_lwc: bool = True
    use_let: bool = True
    learn_let: bool = True
    epochs: int = 10
    lwc_lr: float = 1e-2
    # Log-scale LET is more sensitive than OmniQuant's direct-scale
    # parameterization on Qwen3/GQA.  These bounded defaults are deliberately
    # conservative; the prior unbounded 5e-3 setup can explode within layers.
    let_lr: float = 1e-3
    init_lwc_logit: float = 4.0
    min_let_scale: float = 5e-2
    max_let_scale: float = 20.0
    max_grad_norm: float = 1.0
    eps: float = 1e-12

    def validate(self) -> None:
        weight = validate_quant_format(self.weight_quant_format)
        activation = validate_quant_format(self.activation_quant_format)
        if weight not in ("int4", "int8"):
            raise ValueError("OmniQuant LWC currently supports INT4/INT8 weights only.")
        if self.weight_quant_scheme not in ("symmetric", "asymmetric"):
            raise ValueError("weight_quant_scheme must be 'symmetric' or 'asymmetric'.")
        if self.epochs <= 0:
            raise ValueError("omni_epochs must be positive.")
        if self.lwc_lr <= 0 or self.let_lr <= 0:
            raise ValueError("OmniQuant learning rates must be positive.")
        if self.min_let_scale <= 0 or self.max_let_scale <= self.min_let_scale:
            raise ValueError("LET scales require 0 < min_let_scale < max_let_scale.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if not self.use_lwc and not self.use_let:
            raise ValueError("At least one of LWC or LET must be enabled.")
        if self.learn_let and not self.use_let:
            raise ValueError("learn_let=True requires use_let=True.")
        del activation


@dataclass(frozen=True)
class OmniQuantSummary:
    replaced_linears: int
    final_loss: float
    initial_loss: float
    let_scales: tuple[str, ...]
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _signed_qmax(quant_format: QuantFormat) -> float:
    if quant_format == "int4":
        return 7.0
    if quant_format == "int8":
        return 127.0
    raise ValueError(f"Symmetric LWC needs an integer format, got {quant_format!r}")


def _unsigned_qmax(quant_format: QuantFormat) -> float:
    if quant_format == "int4":
        return 15.0
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
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / float(qmax)
    normalized_x = (x.float() / scale).clamp(min=-float(qmax), max=float(qmax))
    if normalized == "fp8_e4m3fn":
        quantized = normalized_x.to(require_fp8()).float()
    else:
        quantized = _round_ste(normalized_x)
    return (quantized * scale).to(x.dtype)


class _TrainableSymmetricLinear(nn.Module):
    """Frozen Linear with trainable per-channel LWC and optional LET scales.

    The historical class name is retained to avoid breaking local probes that
    import it directly.  ``config.weight_quant_scheme`` selects symmetric or
    paper-style asymmetric LWC.
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
        self.register_buffer("weight_fp", linear.weight.detach().clone(), persistent=False)
        self.register_buffer(
            "bias_fp", None if linear.bias is None else linear.bias.detach().clone(), persistent=False
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
        scale = _bounded_let_scale(self._let_parameters[name], self.config)
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
            scale = _bounded_let_scale(
                self._let_parameters[self.weight_col_repeat_let_name], self.config
            )
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
        scale = threshold.clamp_min(self.config.eps) / _signed_qmax(self.config.weight_quant_format)
        q = round_fn(weight_fp32 / scale).clamp(
            min=-_signed_qmax(self.config.weight_quant_format),
            max=_signed_qmax(self.config.weight_quant_format),
        )
        return (q * scale).to(weight.dtype)

    def qdq_weight(self) -> torch.Tensor:
        return self._qdq_weight_tensor(self.transformed_weight(), use_ste=True)

    def finalize_qdq_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self._qdq_weight_tensor(weight, use_ste=False)

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
        return F.linear(self._prepare_input(x), self.qdq_weight(), self.bias_fp)


class _TrainableOmniBlock(nn.Module):
    """Copied decoder block plus the shared LET parameter bank."""

    def __init__(self, block: nn.Module, *, config: OmniQuantConfig, init_scales: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.block = block
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)
        self.config = config
        self.gqa_head_dim = self._head_dim()
        self.let_parameters = nn.ParameterDict()
        if config.use_let:
            self._add_let_parameter("qkv", init_scales.get("self_attn.q_proj"), self._hidden_size())
            self._add_let_parameter("mlp", init_scales.get("mlp.gate_proj"), self._hidden_size())
            self._add_let_parameter(
                "vo",
                self._gqa_vo_initial(init_scales.get("self_attn.o_proj")),
                self._kv_hidden_size(),
            )
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
        initial = initial.to(device=_module_device(self.block)).clamp(
            min=self.config.min_let_scale,
            max=self.config.max_let_scale,
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

    def _replace_linears(self) -> None:
        qkv = "qkv" if self.config.use_let else None
        mlp = "mlp" if self.config.use_let else None
        vo = "vo" if self.config.use_let else None
        self._replace("self_attn.q_proj", input_let_name=qkv, weight_col_let_name=qkv)
        self._replace("self_attn.k_proj", input_let_name=qkv, weight_col_let_name=qkv)
        self._replace(
            "self_attn.v_proj",
            input_let_name=qkv,
            weight_col_let_name=qkv,
            weight_row_let_name=vo,
            weight_row_let_inverse=True,
        )
        self._replace(
            "self_attn.o_proj",
            weight_col_repeat_let_name=vo,
            weight_col_repeat_head_dim=self.gqa_head_dim if vo is not None else None,
        )
        self._replace("mlp.gate_proj", input_let_name=mlp, weight_col_let_name=mlp)
        self._replace("mlp.up_proj", input_let_name=mlp, weight_col_let_name=mlp)
        self._replace("mlp.down_proj")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
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


def _bounded_let_scale(log_scale: torch.Tensor, config: OmniQuantConfig) -> torch.Tensor:
    """Exponentiate only after clamping to avoid overflow in failed updates."""
    return torch.exp(
        log_scale.clamp(
            min=math.log(config.min_let_scale),
            max=math.log(config.max_let_scale),
        )
    )


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


def _train_block(
    *,
    teacher_block: nn.Module,
    train_block: _TrainableOmniBlock,
    fp_inputs: Sequence[Batch],
    quant_inputs: Sequence[Batch],
    config: OmniQuantConfig,
) -> tuple[float, float]:
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
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0) if groups else None
    device = _module_device(train_block)
    initial_total = 0.0
    initial_count = 0
    final_loss = math.nan
    teacher_block.eval()
    train_block.eval()
    # A full pre-training mean makes the printed loss comparable to the final
    # epoch mean.  Recording only the first batch made small apparent
    # increases ambiguous for variable-length calibration prompts.
    with torch.no_grad():
        for fp_batch, quant_batch in zip(fp_inputs, quant_inputs):
            fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
            fp_args = _move_tree_to_device(fp_args, device)
            fp_kwargs = _move_tree_to_device(fp_kwargs, device)
            quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
            quant_args = _move_tree_to_device(quant_args, device)
            quant_kwargs = _move_tree_to_device(quant_kwargs, device)
            target = _first_tensor(teacher_block(*fp_args, **fp_kwargs)).detach()
            prediction = _first_tensor(train_block(*quant_args, **quant_kwargs))
            loss = F.mse_loss(prediction.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite OmniQuant reconstruction loss before training.")
            initial_total += float(loss)
            initial_count += 1
    initial_loss = initial_total / max(1, initial_count)

    if optimizer is None:
        return initial_loss, initial_loss

    checked_let_gradient = not let_parameters
    train_block.train()
    for _epoch in range(config.epochs):
        epoch_loss = 0.0
        for fp_batch, quant_batch in zip(fp_inputs, quant_inputs):
            fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
            fp_args = _move_tree_to_device(fp_args, device)
            fp_kwargs = _move_tree_to_device(fp_kwargs, device)
            quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
            quant_args = _move_tree_to_device(quant_args, device)
            quant_kwargs = _move_tree_to_device(quant_kwargs, device)
            with torch.no_grad():
                target = _first_tensor(teacher_block(*fp_args, **fp_kwargs)).detach()
            prediction = _first_tensor(train_block(*quant_args, **quant_kwargs))
            loss = F.mse_loss(prediction.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite OmniQuant reconstruction loss before optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable_parameters = lwc_parameters + let_parameters
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
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=config.max_grad_norm)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in trainable_parameters):
                raise FloatingPointError("Non-finite OmniQuant parameter encountered after optimizer step.")
            epoch_loss += float(loss.detach())
        final_loss = epoch_loss / max(1, len(fp_inputs))
    train_block.eval()

    # Evaluate the actual post-update parameters.  The running epoch mean is
    # measured across changing parameters and can hide a bad final update.
    post_total = 0.0
    post_count = 0
    with torch.no_grad():
        for fp_batch, quant_batch in zip(fp_inputs, quant_inputs):
            fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
            fp_args = _move_tree_to_device(fp_args, device)
            fp_kwargs = _move_tree_to_device(fp_kwargs, device)
            quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
            quant_args = _move_tree_to_device(quant_args, device)
            quant_kwargs = _move_tree_to_device(quant_kwargs, device)
            target = _first_tensor(teacher_block(*fp_args, **fp_kwargs)).detach()
            prediction = _first_tensor(train_block(*quant_args, **quant_kwargs))
            loss = F.mse_loss(prediction.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite OmniQuant reconstruction loss after training.")
            post_total += float(loss)
            post_count += 1
    final_loss = post_total / max(1, post_count)

    return initial_loss, final_loss


def _get_linear(module: nn.Module, name: str) -> nn.Linear:
    linear = module.get_submodule(name)
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected {name} to remain nn.Linear during finalization.")
    return linear


def _scale(train_block: _TrainableOmniBlock, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _bounded_let_scale(train_block.let_parameters[name], train_block.config).to(device, dtype)


def _finalize_block(train_block: _TrainableOmniBlock) -> tuple[nn.Module, int]:
    """Fold LET into a fresh FP block, then install frozen learned-QDQ linears."""
    source = copy.deepcopy(train_block.block)
    config = train_block.config
    device = _module_device(source) or torch.device("cpu")
    names = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    trained_linears = {name: train_block.block.get_submodule(name) for name in names}
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
    return source, replaced


def _fold_norm_input(module: nn.Module, norm_name: str, linear_names: Sequence[str], scale: torch.Tensor) -> None:
    norm = module.get_submodule(norm_name)
    weight = getattr(norm, "weight", None)
    if not torch.is_tensor(weight) or weight.numel() != scale.numel():
        raise ValueError(f"LET scale cannot fold into {norm_name}.")
    with torch.no_grad():
        weight.div_(scale.to(weight.device, weight.dtype).reshape_as(weight))
        bias = getattr(norm, "bias", None)
        if torch.is_tensor(bias):
            bias.div_(scale.to(bias.device, bias.dtype).reshape_as(bias))
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


def apply_omniquant_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    config: OmniQuantConfig,
    capture_layer_input_batches: Any,
    act_quant_mode: str = "per_linear",
    checkpoint_dir: str | Path | None = None,
) -> dict[int, OmniQuantSummary]:
    """Calibrate and replace selected Qwen3 blocks sequentially.

    ``capture_layer_input_batches`` is injected by the runner to avoid a
    circular import.  Captured and propagated streams are moved to CPU between
    blocks, which keeps variable-length OneRec calibration prompts from
    occupying the entire GPU.
    """
    config.validate()
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.layers
    summaries: dict[int, OmniQuantSummary] = {}
    fp_inputs: list[Batch] | None = None
    quant_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    for layer_idx in sorted(layer_indices):
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
        init_scales = collect_smoothquant_scales(teacher_block, fp_inputs) if config.use_let else {}
        train_block = _TrainableOmniBlock(copy.deepcopy(teacher_block), config=config, init_scales=init_scales)
        initial_loss, final_loss = _train_block(
            teacher_block=teacher_block,
            train_block=train_block,
            fp_inputs=fp_inputs,
            quant_inputs=quant_inputs,
            config=config,
        )
        next_fp_inputs = _advance_cpu(teacher_block, fp_inputs)
        final_block, replaced = _finalize_block(train_block)
        shared_attention_modules = 0
        shared_mlp_modules = 0
        if config.activation_quant_format != "none" and act_quant_mode == "shared_input":
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
                    "min_let_scale": config.min_let_scale,
                    "max_let_scale": config.max_let_scale,
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
            }
            torch.save(learned_state, checkpoint_root / f"layer_{layer_idx:02d}.pt")
        print(
            f"[omniquant] layer={layer_idx} replaced_linears={replaced} "
            f"loss={initial_loss:.6e}->{final_loss:.6e} "
            f"let_mode={'learned' if config.learn_let else ('fixed' if config.use_let else 'none')} "
            f"let={','.join(train_block.let_parameters.keys()) or 'off'}"
        )
    return summaries
