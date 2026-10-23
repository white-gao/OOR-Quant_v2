"""Isolated GPU primitives for the efficiency experiment; no training changes.

The RecQuant operator probe uses learned parameters, but is NOT a full-model
RecQuant deployment. Its full-stage time is estimated by the collector.
"""
from __future__ import annotations

import copy
import math
import random
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

from flat_quant.flatquant.runtime import (
    FlatQuantCoreConfig, _TrainableFlatQuantBlock, _flat_wrappers,
    _load_flatquant_checkpoint, _validate_checkpoint_config,
    _restore_flatquant_train_state, _materialize_transform_bank,
    _official_transform_activation, _activation_site_for_role,
)
from real_quant.naive_w8a8.modules import RealFP8Linear, FP8_MAX


GROUPS = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "down": ("mlp.down_proj",),
}
ROLES = {"qkv": "q", "o": "o", "gate_up": "gate", "down": "down"}


def load_config(path: Path, layer: int):
    state = _load_flatquant_checkpoint(path, layer_idx=layer)
    names = {field.name for field in fields(FlatQuantCoreConfig)}
    unknown = set(state["config"]) - names
    if unknown:
        raise ValueError(f"Unknown checkpoint config fields: {sorted(unknown)}")
    config = FlatQuantCoreConfig(**state["config"])
    config.validate()
    if (config.weight_quant_format, config.activation_quant_format,
            config.weight_quant_scheme, config.weight_group_size) != (
            "fp8_e4m3fn", "fp8_e4m3fn", "symmetric", 0):
        raise ValueError("This collector requires per-channel symmetric FP8 W8A8.")
    if config.transform_kind != "kronecker" or not config.use_lac:
        raise ValueError("Expected the final matrix-transform + LAC RecQuant topology.")
    if state.get("act_quant_mode") != "shared_input":
        raise ValueError("Expected shared-input activation quantization.")
    _validate_checkpoint_config(state, config=config, checkpoint_path=path)
    return state, config


@torch.no_grad()
def export_weight(wrapper):
    """Export learned codes/scales BEFORE BF16 QDQ rounding; never RTN-requantize."""
    weight = wrapper.transformed_weight().float()
    upper = weight.amax(-1, keepdim=True)
    lower = weight.amin(-1, keepdim=True)
    if wrapper.config.use_lwc:
        upper = upper * wrapper.upper_clip_logits.sigmoid()
        lower = lower * wrapper.lower_clip_logits.sigmoid()
    clipped = torch.minimum(torch.maximum(weight, lower), upper)
    scale = clipped.abs().amax(-1, keepdim=True).clamp_min(wrapper.config.eps) / FP8_MAX
    codes = (clipped / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # This checks export, not numerical equivalence of native vs fake GEMM.
    reconstructed = (codes.float() * scale).to(wrapper.execution_dtype)
    expected = wrapper.finalize_qdq_weight(weight).to(wrapper.execution_dtype)
    if not torch.equal(reconstructed, expected):
        raise ValueError("Native weight export does not reproduce checkpoint QDQ weights.")
    bias = wrapper.transformed_bias()
    return RealFP8Linear(
        weight_fp8_t=codes.t(), weight_scale=scale.t().contiguous(),
        bias=None if bias is None else bias.to(wrapper.execution_dtype),
        in_features=weight.shape[1], out_features=weight.shape[0],
        output_dtype=wrapper.execution_dtype, eps=wrapper.config.eps,
    )


class ProjectionGroup(nn.Module):
    def __init__(self, linears, *, bank=None, clip=None, role=None, head_dim=128):
        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.bank = bank
        self.clip = clip
        self.role = role
        self.head_dim = head_dim

    def transform(self, x):
        if self.bank is None:
            return x
        return _official_transform_activation(
            x, role=self.role, transform_bank=self.bank, head_dim=self.head_dim,
        )

    def prepare(self, x):
        x = self.transform(x)
        if self.clip is not None:
            x = self.clip.clipped(x)
        return self.linears[0].prepare_input(x)

    def forward(self, x):
        prepared = self.prepare(x)  # Exactly once for QKV or gate/up.
        return tuple(module.forward_prepared(prepared) for module in self.linears)


def build_probes(block, checkpoint: Path, layer: int):
    state, config = load_config(checkpoint, layer)
    train = _TrainableFlatQuantBlock(copy.deepcopy(block), config=config,
                                     act_quant_mode="shared_input")
    _restore_flatquant_train_state(train, state, checkpoint_path=checkpoint)
    bank = _materialize_transform_bank(train)
    wrappers = _flat_wrappers(train.block)
    groups = {}
    for name, paths in GROUPS.items():
        rtn = ProjectionGroup([RealFP8Linear.from_linear(block.get_submodule(path)) for path in paths])
        clip = copy.deepcopy(train.activation_quantizers[_activation_site_for_role(ROLES[name])])
        rec = ProjectionGroup([export_weight(wrappers[path]) for path in paths],
                              bank=bank, clip=clip, role=ROLES[name], head_dim=train.head_dim)
        rtn.requires_grad_(False).eval()
        rec.requires_grad_(False).eval()
        groups[name] = (rtn, rec)
    return groups


def generation_kwargs(tokenizer, beam: int):
    # Fixed-workload efficiency test: no early EOS; not a quality evaluation.
    return dict(num_beams=beam, num_return_sequences=beam, do_sample=False,
                use_cache=True, min_new_tokens=3, max_new_tokens=3,
                eos_token_id=None, forced_eos_token_id=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0)


def forward_input(args, kwargs):
    value = kwargs.get("input_ids")
    if value is None:
        value = kwargs.get("inputs_embeds")
    if value is None and args:
        value = args[0]
    if value is None:
        raise ValueError("Cannot identify root model input shape.")
    return tuple(value.shape)


@torch.inference_mode()
def timed_generate(model, inputs, kwargs, *, split=True):
    """Wall-clock first-iteration vs remaining iterations, including beam work.

    The first decode pre-hook is the boundary: it includes all first-step
    candidate selection/cache bookkeeping in prefill. The decode interval ends
    after generate finalization. A separate unsplit run measures synchronization
    perturbation. No tokenizer or input transfer is inside these timers.
    """
    calls, boundary = [], []

    def before(_module, args, kw):
        calls.append(forward_input(args, kw))
        if len(calls) == 2 and split:
            torch.cuda.synchronize()
            boundary.append(time.perf_counter())

    handle = model.register_forward_pre_hook(before, with_kwargs=True)
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
    finally:
        handle.remove()
    if len(calls) != 3 or any(shape[1] != 1 for shape in calls[1:]):
        raise ValueError(f"Expected one prefill and two single-token decode calls: {calls}")
    if output.shape[-1] != inputs["input_ids"].shape[-1] + 3:
        raise ValueError("Generation did not execute the fixed three-token workload.")
    result = {"total_ms": (end - start) * 1000, "forward_shapes": [list(shape) for shape in calls]}
    if split:
        result.update(prefill_ms=(boundary[0] - start) * 1000,
                      decode_ms=(end - boundary[0]) * 1000)
    return result


@torch.inference_mode()
def trace_shapes(model, inputs, kwargs):
    """Read-only BF16 hooks capture actual beam-expanded group call counts."""
    counter, calls, handles = Counter(), [], []

    def root_hook(_module, args, kw):
        calls.append(forward_input(args, kw))

    handles.append(model.register_forward_pre_hook(root_hook, with_kwargs=True))
    for layer in model.model.layers:
        for group, paths in GROUPS.items():
            def group_hook(_module, args, group=group):
                stage = "prefill" if len(calls) == 1 else "decode"
                counter[(stage, group, tuple(args[0].shape))] += 1
            handles.append(layer.get_submodule(paths[0]).register_forward_pre_hook(group_hook))
    try:
        model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if len(calls) != 3:
        raise ValueError(f"Unexpected number of forward calls: {calls}")
    for stage, steps in (("prefill", 1), ("decode", 2)):
        for group in GROUPS:
            count = sum(n for (s, g, _shape), n in counter.items() if (s, g) == (stage, group))
            if count != len(model.model.layers) * steps:
                raise ValueError(f"Unexpected {stage}/{group} calls: {count}")
    return [{"stage": s, "group": g, "shape": list(shape), "calls": n}
            for (s, g, shape), n in sorted(counter.items())]


@torch.inference_mode()
def benchmark_group(rtn, rec, shape, *, warmup, repeats, seed):
    device = rtn.linears[0].weight_fp8_t.device
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tuple(shape), device=device, dtype=torch.bfloat16, generator=rng)
    for _ in range(warmup):
        rtn(x)
        rec(x)
    torch.cuda.synchronize()
    methods = {"RTN": rtn, "RecQuant": rec}
    paired = {name: [] for name in methods}
    order_rng = random.Random(seed)
    # CUDA events avoid a per-kernel host synchronization charge. Time the
    # COMPLETE shared-input projection group, not standalone GEMM speed ratios.
    for _ in range(repeats):
        order = list(methods)
        order_rng.shuffle(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = methods[name](x)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"Invalid operator time: {elapsed}")
            paired[name].append(elapsed)
            del output
    # Untimed numerical sanity, not full model quality or fake/native parity.
    if not all(torch.isfinite(y).all().item() for y in rec(x)):
        raise ValueError("Non-finite RecQuant operator output.")
    return paired
