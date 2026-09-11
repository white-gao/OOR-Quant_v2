from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import re
from typing import Any, Iterable, Iterator, Literal

import torch
import torch.nn as nn

from .modules import BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear
from .quant import (
    ActQuant,
    ActQuantMode,
    QuantFormat,
    WeightQuantScheme,
    activation_per_token_qdq_by_format,
    normalize_weight_group_size,
    resolve_weight_quant_scheme,
    validate_quant_format,
)


SmoothScope = Literal["all", "omni"]
OMNI_SMOOTH_LINEAR_LEAF_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


@dataclass(frozen=True)
class BaselineQuantSummary:
    replaced_linears: int
    skipped_linears: int
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def apply_baseline_w8a8(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    act_quant_mode: ActQuantMode = "per_linear",
    weight_group_size: int | None = None,
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> BaselineQuantSummary:
    """Backward-compatible alias for FP8 E4M3 weight/activation fake QDQ."""
    activation_quant_format: QuantFormat = "fp8_e4m3fn" if act_quant == "per_token" else "none"
    return apply_baseline_qdq(
        model,
        weight_quant_format="fp8_e4m3fn",
        weight_group_size=weight_group_size,
        activation_quant_format=activation_quant_format,
        act_quant_mode=act_quant_mode,
        skip_module_names=skip_module_names,
        target_regex=target_regex,
        skip_regex=skip_regex,
    )


def apply_baseline_qdq(
    model: nn.Module,
    *,
    weight_quant_format: QuantFormat = "int8",
    weight_quant_scheme: WeightQuantScheme | None = None,
    weight_group_size: int | None = None,
    activation_quant_format: QuantFormat = "int8",
    act_quant_mode: ActQuantMode = "per_linear",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> BaselineQuantSummary:
    """Replace selected Linear modules with composable fake-QDQ wrappers.

    ``weight_quant_format`` and ``activation_quant_format`` are independently
    selected from ``none``, ``fp8_e4m3fn``, ``fp4_e2m1``, ``int8``, ``int6``,
    and ``int4``.
    Quantized values are dequantized before ``F.linear``; this path is for
    controlled quality evaluation, not low-bit kernel benchmarking.
    """
    weight_format = validate_quant_format(weight_quant_format)
    weight_scheme = resolve_weight_quant_scheme(weight_format, weight_quant_scheme)
    normalized_weight_group_size = normalize_weight_group_size(weight_group_size)
    activation_format = validate_quant_format(activation_quant_format)
    act_quant: ActQuant = "none" if activation_format == "none" else "per_token"
    _validate_act_quant_mode(act_quant=act_quant, act_quant_mode=act_quant_mode)
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    replaced, skipped = _replace_children_baseline(
        model,
        prefix="",
        act_quant=act_quant,
        weight_quant_format=weight_format,
        weight_quant_scheme=weight_scheme,
        weight_group_size=normalized_weight_group_size,
        activation_quant_format=activation_format,
        skip_names=skip_names,
        target_pattern=target_pattern,
        skip_pattern=skip_pattern,
    )
    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(model)
    return BaselineQuantSummary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def iter_baseline_w8a8_modules(model: nn.Module) -> Iterator[BaselineFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, BaselineFakeQuantLinear):
            yield module


def iter_baseline_qdq_modules(model: nn.Module) -> Iterator[BaselineFakeQuantLinear]:
    """Yield every baseline fake-QDQ Linear wrapper, regardless of format."""
    yield from iter_baseline_w8a8_modules(model)


def iter_smoothquant_w8a8_modules(model: nn.Module) -> Iterator[SmoothQuantFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, SmoothQuantFakeQuantLinear):
            yield module


def iter_gptq_w8a8_modules(model: nn.Module) -> Iterator[GPTQFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, GPTQFakeQuantLinear):
            yield module


def install_shared_input_activation_quantization(model: nn.Module) -> tuple[int, int]:
    """Patch Qwen-style attention/MLP modules to reuse activation QDQ at shared inputs."""
    attention_modules = 0
    mlp_modules = 0
    for module in model.modules():
        if _is_qwen3_attention_like(module):
            module.forward = MethodType(_shared_qwen3_attention_forward, module)
            attention_modules += 1
        elif _is_qwen3_mlp_like(module):
            module.forward = MethodType(_shared_qwen3_mlp_forward, module)
            mlp_modules += 1
    return attention_modules, mlp_modules


def _validate_act_quant_mode(*, act_quant: ActQuant, act_quant_mode: ActQuantMode) -> None:
    if act_quant_mode not in ("per_linear", "shared_input"):
        raise ValueError(f"Unsupported act_quant_mode: {act_quant_mode}")
    if act_quant == "none" and act_quant_mode != "per_linear":
        raise ValueError("act_quant_mode is only meaningful when activation quantization is enabled.")


def _validate_smooth_scope(scope: str) -> None:
    if scope not in ("all", "omni"):
        raise ValueError(f"Unsupported smooth scope: {scope!r}")


def should_apply_smooth_transform(full_name: str, scope: SmoothScope) -> bool:
    """Return whether SmoothQuant should use an equivalent-transform scale."""
    _validate_smooth_scope(scope)
    if scope == "all":
        return True
    if full_name == "":
        return True
    return full_name.rsplit(".", 1)[-1] in OMNI_SMOOTH_LINEAR_LEAF_NAMES


def _replace_children_baseline(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    weight_quant_format: QuantFormat,
    weight_quant_scheme: WeightQuantScheme,
    weight_group_size: int | None,
    activation_quant_format: QuantFormat,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> tuple[int, int]:
    replaced = 0
    skipped = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, (BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear)):
            continue

        if isinstance(child, nn.Linear):
            if _should_skip(
                child_name=child_name,
                full_name=full_name,
                skip_names=skip_names,
                target_pattern=target_pattern,
                skip_pattern=skip_pattern,
            ):
                skipped += 1
                continue
            setattr(
                module,
                child_name,
                BaselineFakeQuantLinear(
                    child,
                    act_quant=act_quant,
                    weight_quant_format=weight_quant_format,
                    weight_quant_scheme=weight_quant_scheme,
                    weight_group_size=weight_group_size,
                    activation_quant_format=activation_quant_format,
                ),
            )
            replaced += 1
            continue

        child_replaced, child_skipped = _replace_children_baseline(
            child,
            prefix=full_name,
            act_quant=act_quant,
            weight_quant_format=weight_quant_format,
            weight_quant_scheme=weight_quant_scheme,
            weight_group_size=weight_group_size,
            activation_quant_format=activation_quant_format,
            skip_names=skip_names,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
        )
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped


def _should_skip(
    *,
    child_name: str,
    full_name: str,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> bool:
    if child_name in skip_names or full_name in skip_names:
        return True
    if skip_pattern is not None and skip_pattern.search(full_name) is not None:
        return True
    if target_pattern is not None and target_pattern.search(full_name) is None:
        return True
    return False


def _is_qwen3_attention_like(module: nn.Module) -> bool:
    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "head_dim",
        "config",
        "attention_dropout",
        "layer_idx",
        "sliding_window",
        "scaling",
    )
    return all(hasattr(module, name) for name in required)


def _is_qwen3_mlp_like(module: nn.Module) -> bool:
    required = ("gate_proj", "up_proj", "down_proj", "act_fn")
    return all(hasattr(module, name) for name in required)


def _uses_quant_linear(module: Any) -> bool:
    return isinstance(
        module,
        (BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear),
    ) or bool(getattr(module, "is_flatquant_linear", False))


def _shared_prepare_input(modules: tuple[Any, ...], x: torch.Tensor) -> torch.Tensor | None:
    quant_modules = [module for module in modules if _uses_quant_linear(module)]
    if not quant_modules:
        return None

    first = quant_modules[0]
    if isinstance(first, SmoothQuantFakeQuantLinear):
        x = first.smooth_activation(x)

    if not any(getattr(module, "act_quant", "none") == "per_token" for module in quant_modules):
        return x

    activation_quant_format = validate_quant_format(
        getattr(first, "activation_quant_format", "fp8_e4m3fn")
    )
    eps = float(getattr(first, "eps", 1e-12))
    fp8_qmax = float(getattr(first, "qmax", 448.0))
    quantize_activation = getattr(first, "quantize_activation", None)
    if callable(quantize_activation):
        return quantize_activation(x)
    return activation_per_token_qdq_by_format(
        x,
        quant_format=activation_quant_format,
        eps=eps,
        fp8_qmax=fp8_qmax,
    )


def _shared_linear_forward(module: Any, raw_x: torch.Tensor, prepared_x: torch.Tensor | None) -> torch.Tensor:
    if prepared_x is not None and _uses_quant_linear(module):
        return module.forward_prepared(prepared_x)
    return module(raw_x)


def _shared_qwen3_attention_forward(
    self: nn.Module,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv_hidden_states = _shared_prepare_input((self.q_proj, self.k_proj, self.v_proj), hidden_states)

    query_states = self.q_norm(
        _shared_linear_forward(self.q_proj, hidden_states, qkv_hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = self.k_norm(
        _shared_linear_forward(self.k_proj, hidden_states, qkv_hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = _shared_linear_forward(self.v_proj, hidden_states, qkv_hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    o_input = _shared_prepare_input((self.o_proj,), attn_output)
    attn_output = _shared_linear_forward(self.o_proj, attn_output, o_input)
    return attn_output, attn_weights


def _shared_qwen3_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_up_x = _shared_prepare_input((self.gate_proj, self.up_proj), x)
    gate = _shared_linear_forward(self.gate_proj, x, gate_up_x)
    up = _shared_linear_forward(self.up_proj, x, gate_up_x)
    down_input = self.act_fn(gate) * up
    prepared_down_input = _shared_prepare_input((self.down_proj,), down_input)
    return _shared_linear_forward(self.down_proj, down_input, prepared_down_input)
