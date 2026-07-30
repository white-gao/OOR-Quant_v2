from __future__ import annotations

import copy
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from fake_quant.gptq import collect_gptq_hessians
from fake_quant.support.runtime_utils import _detach_tree, _module_device, _move_tree_to_device
from fake_quant.support.smoothquant_runtime import Batch, _batch_to_args_kwargs

from .apply import NaiveW8A8Summary, apply_naive_w8a8, install_shared_input_activation_quantization
from .modules import ActivationQuantMode, RealFP8Linear, activation_qdq_like_runtime


WEIGHT_QUANT_MODES = (
    "minmax",
    "gptq",
    "gptaq",
)

SID_ITEM_RE = re.compile(
    r"<\|sid_begin\|>"
    r"(?P<sid><s_a_[^>]+><s_b_[^>]+><s_c_[^>]+>)"
    r"<\|sid_end\|>"
)


def default_calib_split(
    data_dir: str | Path,
    fallback_split: str,
    *,
    task_name: str,
    resolve_path,
) -> str:
    calib_file = resolve_path(data_dir) / task_name / f"{task_name}_calib.parquet"
    return "calib" if calib_file.exists() else fallback_split


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def extract_sid_teacher_forcing_targets(ground_truth: str, *, max_items: int = 1) -> list[str]:
    if max_items <= 0:
        return []
    targets: list[str] = []
    for match in SID_ITEM_RE.finditer(ground_truth or ""):
        targets.append(match.group("sid"))
        if len(targets) >= max_items:
            break
    return targets


def build_model_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        batches.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
        )
    return batches


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("Could not find transformer layers at model.model.layers or model.layers.")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Expected layers to be nn.ModuleList, got {type(layers)!r}.")
    return layers


def parse_layer_indices(spec: str, *, num_layers: int) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("last:"):
        count = int(spec.split(":", 1)[1])
        if count <= 0 or count > num_layers:
            raise ValueError(f"Invalid last layer count: {count}")
        return list(range(num_layers - count, num_layers))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    deduped = sorted(set(indices))
    if not deduped:
        raise ValueError("No layer indices selected.")
    for idx in deduped:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers.")
    return deduped


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Could not find a tensor in output type {type(value)!r}.")


def capture_layer_input_batches(
    *,
    model: nn.Module,
    layer: nn.Module,
    model_batches: Iterable[Mapping[str, Any]],
) -> list[Batch]:
    captured: list[Batch] = []

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((_detach_tree(args), _detach_tree(kwargs)))

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in model_batches:
                try:
                    model(**batch, use_cache=False)
                except TypeError:
                    model(**batch)
    finally:
        handle.remove()
        model.train(was_training)
    return captured


def advance_layer_input_batches(
    *,
    layer: nn.Module,
    batches: Sequence[Batch],
) -> list[Batch]:
    advanced: list[Batch] = []
    was_training = layer.training
    target_device = _module_device(layer)
    layer.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, target_device)
                kwargs = _move_tree_to_device(kwargs, target_device)
                output = layer(*args, **kwargs)
                hidden = _first_tensor(output).detach()
                next_args: tuple[Any, ...] = (hidden, *args[1:]) if args else ()
                next_kwargs = dict(kwargs)
                if not next_args:
                    if "hidden_states" in next_kwargs:
                        next_kwargs["hidden_states"] = hidden
                    else:
                        next_args = (hidden,)
                advanced.append((_detach_tree(next_args), _detach_tree(next_kwargs)))
    finally:
        layer.train(was_training)
    return advanced


@dataclass(frozen=True)
class GPTAQLinearStats:
    hessian_q: torch.Tensor
    dxx_t: torch.Tensor


def collect_gptaq_linear_stats(
    fp_module: nn.Module,
    fp_batches: Sequence[Batch],
    q_batches: Sequence[Batch],
    *,
    q_module: nn.Module | None = None,
    target_names: Iterable[str] | None = None,
    activation_quant_mode: ActivationQuantMode = "dynamic",
    static_activation_scales: Mapping[str, torch.Tensor | float] | None = None,
    decode_a16_when_single_token: bool = False,
    activation_aware: bool = True,
    eps: float = 1e-12,
) -> dict[str, GPTAQLinearStats]:
    """Collect asymmetric GPTAQ statistics under the propagated quantized path.

    For every Linear, ``X_fp`` is captured from the BF16 teacher and ``X_hat``
    is the propagated quantized-path input. When ``activation_aware`` is true,
    ``X_hat`` additionally passes through the same FP8-QDQ, decode-A16, and
    tail branches used by the current Linear at inference. The returned
    statistics are ``X_hat.T @ X_hat`` and
    ``(X_fp - X_hat).T @ X_hat``, with optional token weighting.
    """
    if len(fp_batches) != len(q_batches):
        raise ValueError(f"fp_batches length {len(fp_batches)} does not match q_batches length {len(q_batches)}")
    q_module = fp_module if q_module is None else q_module
    fp_linears = _named_linear_modules(fp_module)
    q_linears = _named_linear_modules(q_module)
    if target_names is not None:
        target_set = set(target_names)
        fp_linears = {name: linear for name, linear in fp_linears.items() if name in target_set}
        q_linears = {name: linear for name, linear in q_linears.items() if name in target_set}
    names = [name for name in fp_linears if name in q_linears]
    if not names:
        return {}

    stats_h: dict[str, torch.Tensor] = {}
    stats_d: dict[str, torch.Tensor] = {}
    counts: dict[str, float] = {name: 0.0 for name in names}

    fp_captures: dict[str, torch.Tensor] = {}
    q_captures: dict[str, torch.Tensor] = {}

    def make_hook(store: dict[str, torch.Tensor], name: str, linear: nn.Linear):
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
            store[name] = x.detach()

        return hook

    same_module = q_module is fp_module
    if same_module:
        active_capture = {"store": fp_captures}

        def make_shared_hook(name: str, linear: nn.Linear):
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
                active_capture["store"][name] = x.detach()

            return hook

        fp_handles = [
            fp_linears[name].register_forward_pre_hook(make_shared_hook(name, fp_linears[name]), with_kwargs=True)
            for name in names
        ]
        q_handles = []
    else:
        active_capture = None
        fp_handles = [
            fp_linears[name].register_forward_pre_hook(make_hook(fp_captures, name, fp_linears[name]), with_kwargs=True)
            for name in names
        ]
        q_handles = [
            q_linears[name].register_forward_pre_hook(make_hook(q_captures, name, q_linears[name]), with_kwargs=True)
            for name in names
        ]

    fp_was_training = fp_module.training
    q_was_training = q_module.training
    fp_device = _module_device(fp_module)
    q_device = _module_device(q_module)
    fp_module.eval()
    q_module.eval()
    try:
        with torch.no_grad():
            for batch_idx, (fp_batch, q_batch) in enumerate(zip(fp_batches, q_batches)):
                fp_captures.clear()
                q_captures.clear()

                fp_args, fp_kwargs = _batch_to_args_kwargs(fp_batch)
                fp_args = _move_tree_to_device(fp_args, fp_device)
                fp_kwargs = _move_tree_to_device(fp_kwargs, fp_device)
                if active_capture is not None:
                    active_capture["store"] = fp_captures
                fp_module(*fp_args, **fp_kwargs)

                q_args, q_kwargs = _batch_to_args_kwargs(q_batch)
                q_args = _move_tree_to_device(q_args, q_device)
                q_kwargs = _move_tree_to_device(q_kwargs, q_device)
                if active_capture is not None:
                    active_capture["store"] = q_captures
                q_module(*q_args, **q_kwargs)

                for name in names:
                    if name not in fp_captures or name not in q_captures:
                        continue
                    x_fp_raw = fp_captures[name]
                    x_path_raw = q_captures[name]
                    if x_fp_raw.shape != x_path_raw.shape:
                        raise ValueError(
                            f"FP and quantized inputs for {name!r} have different shapes: "
                            f"{tuple(x_fp_raw.shape)} vs {tuple(x_path_raw.shape)}"
                        )
                    x_fp = x_fp_raw.float().reshape(-1, fp_linears[name].in_features)
                    if activation_aware:
                        x_hat = activation_qdq_like_runtime(
                            x_path_raw,
                            activation_quant_mode=activation_quant_mode,
                            static_scale=(
                                static_activation_scales.get(name)
                                if static_activation_scales is not None
                                else None
                            ),
                            decode_a16_when_single_token=decode_a16_when_single_token,
                            eps=eps,
                        ).reshape(-1, q_linears[name].in_features)
                    else:
                        x_hat = x_path_raw.float().reshape(-1, q_linears[name].in_features)
                    if x_fp.shape != x_hat.shape:
                        raise ValueError(
                            f"FP and runtime-QDQ inputs for {name!r} have different shapes: "
                            f"{tuple(x_fp.shape)} vs {tuple(x_hat.shape)}"
                        )
                    if x_hat.numel() == 0:
                        continue
                    current_h = x_hat.t().matmul(x_hat)
                    current_d = (x_fp - x_hat).t().matmul(x_hat)
                    denominator = float(x_hat.shape[0])
                    stats_h[name] = current_h if name not in stats_h else stats_h[name].add_(current_h)
                    stats_d[name] = current_d if name not in stats_d else stats_d[name].add_(current_d)
                    counts[name] += denominator
    finally:
        for handle in fp_handles + q_handles:
            handle.remove()
        fp_module.train(fp_was_training)
        q_module.train(q_was_training)

    result: dict[str, GPTAQLinearStats] = {}
    for name in names:
        count = counts.get(name, 0.0)
        in_features = fp_linears[name].in_features
        if count <= 0.0 or name not in stats_h:
            hessian_q = torch.eye(in_features, dtype=torch.float32) * eps
            dxx_t = torch.zeros(in_features, in_features, dtype=torch.float32)
        else:
            hessian_q = stats_h[name] / float(count)
            hessian_q = 0.5 * (hessian_q + hessian_q.t())
            dxx_t = stats_d[name] / float(count)
        result[name] = GPTAQLinearStats(hessian_q=hessian_q.detach().cpu(), dxx_t=dxx_t.detach().cpu())
    return result


def _named_linear_modules(module: nn.Module) -> dict[str, nn.Linear]:
    if isinstance(module, nn.Linear):
        return {"": module}
    return {name: child for name, child in module.named_modules() if isinstance(child, nn.Linear)}


def apply_gptq_real_w8a8_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    output_dtype: torch.dtype,
    target_regex: str | None,
    skip_regex: str | None,
    use_fast_accum: bool,
    activation_quant_mode: ActivationQuantMode,
    decode_a16_when_single_token: bool,
    damp_percent: float,
    block_size: int,
) -> NaiveW8A8Summary:
    layers = get_transformer_layers(model)
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    replaced = 0
    skipped = 0
    shared_attention = 0
    shared_mlp = 0

    for layer_idx in selected_layer_indices:
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None:
                raise RuntimeError("Internal error: stream_layer_idx is not initialized.")
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=fp_inputs,
                )
                stream_layer_idx += 1

        layer = layers[layer_idx]
        hessians = collect_gptq_hessians(layer, fp_inputs)
        next_fp_inputs = advance_layer_input_batches(layer=layer, batches=fp_inputs)
        summary = apply_naive_w8a8(
            layer,
            skip_module_names=(),
            target_regex=target_regex,
            skip_regex=skip_regex,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
            gptq_hessians=hessians,
            gptq_damp_percent=damp_percent,
            gptq_block_size=block_size,
        )
        replaced += summary.replaced_linears
        skipped += summary.skipped_linears
        shared_attention += summary.shared_attention_modules
        shared_mlp += summary.shared_mlp_modules
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(
            f"[hf_naive_w8a8] gptq layer={layer_idx} replaced_linears={summary.replaced_linears}, "
            f"damp_percent={damp_percent}, block_size={block_size}"
        )

    return NaiveW8A8Summary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention,
        shared_mlp_modules=shared_mlp,
    )

def apply_gptaq_real_w8a8_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    output_dtype: torch.dtype,
    target_regex: str | None,
    skip_regex: str | None,
    use_fast_accum: bool,
    activation_quant_mode: ActivationQuantMode,
    decode_a16_when_single_token: bool,
    damp_percent: float,
    block_size: int,
    alpha: float = 1.0,
    activation_aware: bool = True,
) -> NaiveW8A8Summary:
    layers = get_transformer_layers(model)
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    q_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    replaced = 0
    skipped = 0

    for layer_idx in selected_layer_indices:
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            q_inputs = fp_inputs
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None or q_inputs is None:
                raise RuntimeError("Internal error: GPTAQ calibration streams are not initialized.")
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=fp_inputs,
                )
                q_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=q_inputs,
                )
                stream_layer_idx += 1

        if q_inputs is None:
            raise RuntimeError("Internal error: q_inputs is not initialized.")
        layer = layers[layer_idx]
        teacher_layer = copy.deepcopy(layer)
        teacher_layer.to(_module_device(layer))
        teacher_layer.eval()
        next_fp_inputs = advance_layer_input_batches(layer=teacher_layer, batches=fp_inputs)
        layer_replaced, layer_skipped = _quantize_layer_gptaq_sequential(
            layer=layer,
            teacher_layer=teacher_layer,
            fp_inputs=fp_inputs,
            q_inputs=q_inputs,
            target_regex=target_regex,
            skip_regex=skip_regex,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
            damp_percent=damp_percent,
            block_size=block_size,
            alpha=alpha,
            activation_aware=activation_aware,
        )
        next_q_inputs = advance_layer_input_batches(layer=layer, batches=q_inputs)

        replaced += layer_replaced
        skipped += layer_skipped
        fp_inputs = next_fp_inputs
        q_inputs = next_q_inputs
        stream_layer_idx = layer_idx + 1
        del teacher_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[hf_naive_w8a8] gptaq layer={layer_idx} replaced_linears={layer_replaced}, "
            f"damp_percent={damp_percent}, block_size={block_size}, alpha={alpha}, "
            f"activation_target={'runtime_fp8_qdq' if activation_aware else 'propagated_bf16_path'}"
        )

    shared_attention, shared_mlp = install_shared_input_activation_quantization(model)
    return NaiveW8A8Summary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention,
        shared_mlp_modules=shared_mlp,
    )


def _quantize_layer_gptaq_sequential(
    *,
    layer: nn.Module,
    teacher_layer: nn.Module,
    fp_inputs: Sequence[Batch],
    q_inputs: Sequence[Batch],
    target_regex: str | None,
    skip_regex: str | None,
    output_dtype: torch.dtype,
    use_fast_accum: bool,
    activation_quant_mode: ActivationQuantMode,
    decode_a16_when_single_token: bool,
    damp_percent: float,
    block_size: int,
    alpha: float,
    activation_aware: bool,
) -> tuple[int, int]:
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    initial_linears = _named_linear_modules(layer)
    groups = _gptaq_linear_groups(tuple(initial_linears))
    replaced = 0
    skipped = 0
    seen: set[str] = set()

    for group in groups:
        selected: list[str] = []
        for name in group:
            if name in seen:
                continue
            seen.add(name)
            current = _module_by_name(layer, name)
            if not isinstance(current, nn.Linear):
                continue
            if _should_skip_linear_name(name, target_pattern=target_pattern, skip_pattern=skip_pattern):
                skipped += 1
                continue
            selected.append(name)
        if not selected:
            continue

        stats = collect_gptaq_linear_stats(
            teacher_layer,
            fp_inputs,
            q_inputs,
            q_module=layer,
            target_names=selected,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
            activation_aware=activation_aware,
        )
        for name in selected:
            current = _module_by_name(layer, name)
            if not isinstance(current, nn.Linear):
                continue
            stat = stats.get(name)
            if stat is None:
                raise KeyError(f"Missing GPTAQ stats for Linear module: {name}")
            quant_child = RealFP8Linear.from_gptaq_linear(
                current,
                stat.hessian_q,
                stat.dxx_t,
                alpha=alpha,
                output_dtype=output_dtype,
                use_fast_accum=use_fast_accum,
                activation_quant_mode=activation_quant_mode,
                decode_a16_when_single_token=decode_a16_when_single_token,
                damp_percent=damp_percent,
                block_size=block_size,
            )
            _set_module_by_name(layer, name, quant_child)
            replaced += 1
    return replaced, skipped


def _gptaq_linear_groups(linear_names: Sequence[str]) -> list[tuple[str, ...]]:
    available = set(linear_names)
    preferred = (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("self_attn.o_proj",),
        ("mlp.gate_proj", "mlp.up_proj"),
        ("mlp.down_proj",),
    )
    groups: list[tuple[str, ...]] = []
    used: set[str] = set()
    for group in preferred:
        present = tuple(name for name in group if name in available)
        if present:
            groups.append(present)
            used.update(present)
    for name in linear_names:
        if name not in used:
            groups.append((name,))
            used.add(name)
    return groups


def _should_skip_linear_name(
    name: str,
    *,
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> bool:
    child_name = name.rsplit(".", 1)[-1]
    if child_name == "lm_head" or name == "lm_head":
        return True
    if skip_pattern is not None and skip_pattern.search(name) is not None:
        return True
    if target_pattern is not None and target_pattern.search(name) is None:
        return True
    return False


def _module_by_name(module: nn.Module, name: str) -> nn.Module:
    if name == "":
        return module
    current: nn.Module = module
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _set_module_by_name(module: nn.Module, name: str, child: nn.Module) -> None:
    if name == "":
        raise ValueError("Cannot replace the root module in GPTAQ layer quantization.")
    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = module if parent_name == "" else _module_by_name(module, parent_name)
    setattr(parent, child_name, child)
