from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .smoothquant_core import compute_smooth_scale, smooth_linear_weight

from ..apply import SmoothScope, should_apply_smooth_transform
from ..modules import BaselineFakeQuantLinear, SmoothQuantFakeQuantLinear
from ..quant import ActQuant, fp8_weight_per_channel_forward
from .runtime_utils import _module_device, _move_tree_to_device


Batch = Any

DEFAULT_SMOOTHQUANT_ALPHA = 0.5
DEFAULT_SMOOTHQUANT_MIN_SCALE = None
DEFAULT_SMOOTHQUANT_MAX_SCALE = None
DEFAULT_SMOOTH_SCOPE: SmoothScope = "omni"
DEFAULT_SMOOTH_FOLD = True


def _batch_to_args_kwargs(batch: Batch) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[1], dict):
        args, kwargs = batch
        if isinstance(args, tuple):
            return args, dict(kwargs)
        return (args,), dict(kwargs)
    if isinstance(batch, Mapping):
        return (), dict(batch)
    if isinstance(batch, tuple):
        return batch, {}
    return (batch,), {}


def collect_smoothquant_scales(
    module: nn.Module,
    batches: Sequence[Batch],
    *,
    alpha: float = DEFAULT_SMOOTHQUANT_ALPHA,
    min_scale: float | None = DEFAULT_SMOOTHQUANT_MIN_SCALE,
    max_scale: float | None = DEFAULT_SMOOTHQUANT_MAX_SCALE,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"smoothquant alpha must be in [0, 1], got {alpha}")
    linear_modules = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and should_apply_smooth_transform(name, smooth_scope)
    }
    if isinstance(module, nn.Linear):
        linear_modules = {"": module}
    if not linear_modules:
        return {}

    act_absmax = _collect_linear_input_absmax(module, batches, linear_modules)
    scales: dict[str, torch.Tensor] = {}
    grouped: set[str] = set()
    for group in _known_smoothquant_input_group_names(linear_modules):
        members = [name for name in group if name in act_absmax]
        if len(members) != len(group):
            continue
        act_max = torch.stack([act_absmax[name].float().cpu() for name in members]).amax(dim=0)
        weight_max = torch.stack(
            [_linear_input_weight_absmax(linear_modules[name], eps=eps).cpu() for name in members]
        ).amax(dim=0)
        scale = _smoothquant_scale(
            act_max,
            weight_max,
            alpha=alpha,
            min_scale=min_scale,
            max_scale=max_scale,
            eps=eps,
        )
        for name in members:
            scales[name] = scale.clone()
            grouped.add(name)

    for name, linear in linear_modules.items():
        if name in grouped or name not in act_absmax:
            continue
        scales[name] = _smoothquant_scale(
            act_absmax[name].float().cpu(),
            _linear_input_weight_absmax(linear, eps=eps).cpu(),
            alpha=alpha,
            min_scale=min_scale,
            max_scale=max_scale,
            eps=eps,
        )
    return scales


def _collect_linear_input_absmax(
    module: nn.Module,
    batches: Sequence[Batch],
    linear_modules: Mapping[str, nn.Linear],
) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str, linear: nn.Linear):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if args:
                x = args[0]
            else:
                x = kwargs.get("input", kwargs.get("hidden_states"))
            if not torch.is_tensor(x):
                return
            if x.shape[-1] != linear.in_features:
                raise ValueError(
                    f"Expected input last dim {linear.in_features} for {name!r}, got {tuple(x.shape)}"
                )
            reduce_dims = tuple(range(x.ndim - 1))
            current = x.detach().float().abs()
            current = current.amax(dim=reduce_dims) if reduce_dims else current
            previous = stats.get(name)
            stats[name] = current.cpu() if previous is None else torch.maximum(previous, current.cpu())

        return hook

    for name, linear in linear_modules.items():
        handles.append(linear.register_forward_pre_hook(make_hook(name, linear), with_kwargs=True))

    was_training = module.training
    target_device = _module_device(module)
    module.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, target_device)
                kwargs = _move_tree_to_device(kwargs, target_device)
                module(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return stats


def _linear_input_weight_absmax(linear: nn.Linear, *, eps: float = 1e-12) -> torch.Tensor:
    return linear.weight.detach().float().abs().amax(dim=0).clamp_min(eps)


def _smoothquant_scale(
    act_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    *,
    alpha: float,
    min_scale: float | None,
    max_scale: float | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    scale = compute_smooth_scale(act_absmax, weight_absmax, alpha=alpha, eps=eps)
    if min_scale is not None:
        scale = scale.clamp_min(float(min_scale))
    if max_scale is not None:
        scale = scale.clamp_max(float(max_scale))
    return scale


def _known_smoothquant_input_group_names(modules: Mapping[str, nn.Module]) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for name in sorted(modules):
        if name.endswith(".q_proj"):
            prefix = name[: -len(".q_proj")]
            group = (f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj")
            if all(member in modules for member in group):
                groups.append(group)
        elif name.endswith(".gate_proj"):
            prefix = name[: -len(".gate_proj")]
            group = (f"{prefix}.gate_proj", f"{prefix}.up_proj")
            if all(member in modules for member in group):
                groups.append(group)
    return groups


def smoothquant_quantized_module_from_scales(
    module: nn.Module,
    scales: Mapping[str, torch.Tensor],
    *,
    act_quant: ActQuant,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    folded_names: set[str] | None = None,
) -> tuple[nn.Module, int]:
    if isinstance(module, nn.Linear):
        scale = scales.get("")
        if scale is None:
            raise ValueError("Missing SmoothQuant scale for root Linear module.")
        return _smoothquant_fake_quant_linear(module, scale, act_quant=act_quant), 1
    return module, _replace_children_smoothquant(
        module,
        scales=scales,
        prefix="",
        act_quant=act_quant,
        smooth_scope=smooth_scope,
        folded_names=folded_names or set(),
    )


def _replace_children_smoothquant(
    module: nn.Module,
    *,
    scales: Mapping[str, torch.Tensor],
    prefix: str,
    act_quant: ActQuant,
    smooth_scope: SmoothScope,
    folded_names: set[str],
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, (BaselineFakeQuantLinear, SmoothQuantFakeQuantLinear)):
            continue
        if isinstance(child, nn.Linear):
            if should_apply_smooth_transform(full_name, smooth_scope):
                scale = scales.get(full_name)
                if scale is None:
                    raise KeyError(f"Missing SmoothQuant scale for Linear module: {full_name}")
                replacement = _smoothquant_fake_quant_linear(
                    child,
                    scale,
                    act_quant=act_quant,
                    fold_activation=full_name in folded_names,
                )
            else:
                replacement = BaselineFakeQuantLinear(child, act_quant=act_quant)
            setattr(module, child_name, replacement)
            replaced += 1
            continue
        replaced += _replace_children_smoothquant(
            child,
            scales=scales,
            prefix=full_name,
            act_quant=act_quant,
            smooth_scope=smooth_scope,
            folded_names=folded_names,
        )
    return replaced


def _smoothquant_fake_quant_linear(
    linear: nn.Linear,
    scale: torch.Tensor,
    *,
    act_quant: ActQuant,
    fold_activation: bool = False,
) -> SmoothQuantFakeQuantLinear:
    scale = scale.detach().float().reshape(-1).to(device=linear.weight.device)
    if scale.numel() != linear.in_features:
        raise ValueError(f"Expected SmoothQuant scale shape ({linear.in_features},), got {tuple(scale.shape)}")
    with torch.no_grad():
        scaled_weight = linear.weight.detach() if fold_activation else smooth_linear_weight(linear.weight.detach(), scale)
        weight_qdq = fp8_weight_per_channel_forward(scaled_weight)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        input_scale = None if fold_activation else scale.detach().cpu()
    return SmoothQuantFakeQuantLinear(
        weight_qdq=weight_qdq,
        bias=bias,
        act_quant=act_quant,
        input_scale=input_scale,
    )


def fold_smoothquant_scales_inplace(
    module: nn.Module,
    scales: Mapping[str, torch.Tensor],
    *,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
) -> set[str]:
    """Fold explicit SmoothQuant activation scaling into adjacent FP modules where exact."""
    folded: set[str] = set()
    if smooth_scope != "omni":
        return folded
    folded.update(
        _fold_norm_to_linear_input_group(
            module,
            norm_name="input_layernorm",
            linear_names=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            scales=scales,
        )
    )
    folded.update(
        _fold_norm_to_linear_input_group(
            module,
            norm_name="post_attention_layernorm",
            linear_names=("mlp.gate_proj", "mlp.up_proj"),
            scales=scales,
        )
    )
    if _fold_linear_output_to_linear_input(
        module,
        source_name="self_attn.v_proj",
        target_name="self_attn.o_proj",
        scales=scales,
    ):
        folded.add("self_attn.o_proj")
    if _fold_linear_output_to_linear_input(
        module,
        source_name="mlp.up_proj",
        target_name="mlp.down_proj",
        scales=scales,
    ):
        folded.add("mlp.down_proj")
    return folded


def _fold_norm_to_linear_input_group(
    module: nn.Module,
    *,
    norm_name: str,
    linear_names: Sequence[str],
    scales: Mapping[str, torch.Tensor],
) -> set[str]:
    norm = _maybe_get_submodule(module, norm_name)
    if norm is None or not hasattr(norm, "weight"):
        return set()
    scale = _shared_scale_from_mapping(scales, linear_names)
    if scale is None:
        return set()
    linears = [_maybe_get_submodule(module, name) for name in linear_names]
    if not all(isinstance(linear, nn.Linear) for linear in linears):
        return set()
    if not _scale_matches_norm_and_linear_inputs(scale, norm, linears):
        return set()

    with torch.no_grad():
        _divide_weight_or_bias(norm, "weight", scale)
        _divide_weight_or_bias(norm, "bias", scale)
        for linear in linears:
            assert isinstance(linear, nn.Linear)
            linear.weight.mul_(scale.to(device=linear.weight.device, dtype=linear.weight.dtype).view(1, -1))
    return set(linear_names)


def _fold_linear_output_to_linear_input(
    module: nn.Module,
    *,
    source_name: str,
    target_name: str,
    scales: Mapping[str, torch.Tensor],
) -> bool:
    source = _maybe_get_submodule(module, source_name)
    target = _maybe_get_submodule(module, target_name)
    scale = scales.get(target_name)
    if not isinstance(source, nn.Linear) or not isinstance(target, nn.Linear) or scale is None:
        return False
    scale = scale.detach().float().reshape(-1)
    if source.out_features != target.in_features or scale.numel() != target.in_features:
        return False

    with torch.no_grad():
        source.weight.div_(scale.to(device=source.weight.device, dtype=source.weight.dtype).view(-1, 1))
        if source.bias is not None:
            source.bias.div_(scale.to(device=source.bias.device, dtype=source.bias.dtype))
        target.weight.mul_(scale.to(device=target.weight.device, dtype=target.weight.dtype).view(1, -1))
    return True


def _shared_scale_from_mapping(
    scales: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> torch.Tensor | None:
    tensors = [scales.get(name) for name in names]
    if any(tensor is None for tensor in tensors):
        return None
    scale = tensors[0].detach().float().reshape(-1)
    for tensor in tensors[1:]:
        other = tensor.detach().float().reshape(-1)
        if scale.shape != other.shape or not torch.allclose(scale, other, rtol=1e-4, atol=1e-6):
            return None
    return scale


def _scale_matches_norm_and_linear_inputs(
    scale: torch.Tensor,
    norm: nn.Module,
    linears: Sequence[nn.Module | None],
) -> bool:
    weight = getattr(norm, "weight", None)
    if not torch.is_tensor(weight) or scale.numel() != weight.numel():
        return False
    for linear in linears:
        in_features = getattr(linear, "in_features", None)
        if in_features != scale.numel():
            return False
    return True


def _divide_weight_or_bias(module: nn.Module, attr_name: str, scale: torch.Tensor) -> None:
    value = getattr(module, attr_name, None)
    if not torch.is_tensor(value):
        return
    value.div_(scale.to(device=value.device, dtype=value.dtype).reshape_as(value))


def _maybe_get_submodule(module: nn.Module, name: str) -> nn.Module | None:
    try:
        return module.get_submodule(name)
    except AttributeError:
        return None
