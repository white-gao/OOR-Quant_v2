from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .modules import BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear
from .quant import ActQuant, FP8_MAX, fp8_e4m3_qdq_forward
from .support.runtime_utils import _module_device, _move_tree_to_device
from .support.smoothquant_runtime import Batch, _batch_to_args_kwargs


DEFAULT_GPTQ_DAMP_PERCENT = 0.01
DEFAULT_GPTQ_BLOCK_SIZE = 128


def collect_gptq_hessians(
    module: nn.Module,
    batches: Sequence[Batch],
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Collect per-Linear GPTQ Hessians from calibration activations.

    Each Hessian is the calibration average of X^T X where X is the flattened
    input to that Linear. It is returned on CPU so callers can quantize one
    module at a time without keeping all GPU Hessians alive.
    """
    linear_modules = _named_linear_modules(module)
    if not linear_modules:
        return {}

    stats: dict[str, torch.Tensor] = {}
    counts: dict[str, float] = {name: 0.0 for name in linear_modules}
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
            x2d = x.detach().float().reshape(-1, linear.in_features)
            if x2d.numel() == 0:
                return

            current = x2d.t().matmul(x2d)
            denominator = float(x2d.shape[0])

            previous = stats.get(name)
            stats[name] = current if previous is None else previous.add_(current)
            counts[name] += denominator

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

    hessians: dict[str, torch.Tensor] = {}
    for name, linear in linear_modules.items():
        count = counts.get(name, 0.0)
        if count <= 0.0 or name not in stats:
            hessians[name] = torch.eye(linear.in_features, dtype=torch.float32) * eps
            continue
        hessian = stats[name] / float(count)
        hessian = 0.5 * (hessian + hessian.t())
        hessians[name] = hessian.detach().cpu()
    return hessians


def gptq_quantized_module_from_hessians(
    module: nn.Module,
    hessians: Mapping[str, torch.Tensor],
    *,
    act_quant: ActQuant,
    damp_percent: float = DEFAULT_GPTQ_DAMP_PERCENT,
    block_size: int = DEFAULT_GPTQ_BLOCK_SIZE,
) -> tuple[nn.Module, int]:
    """Replace Linear modules with GPTQ-calibrated FP8 fake-quant wrappers."""
    if isinstance(module, nn.Linear):
        hessian = hessians.get("")
        if hessian is None:
            raise KeyError("Missing GPTQ Hessian for root Linear module.")
        return _gptq_fake_quant_linear(
            module,
            hessian,
            act_quant=act_quant,
            damp_percent=damp_percent,
            block_size=block_size,
        ), 1
    return module, _replace_children_gptq(
        module,
        hessians=hessians,
        prefix="",
        act_quant=act_quant,
        damp_percent=damp_percent,
        block_size=block_size,
    )


def gptq_fp8_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    damp_percent: float = DEFAULT_GPTQ_DAMP_PERCENT,
    block_size: int = DEFAULT_GPTQ_BLOCK_SIZE,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Quantize a Linear weight matrix with GPTQ error compensation and FP8 levels."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    rows, columns = weight.shape
    if hessian.shape != (columns, columns):
        raise ValueError(
            f"Expected Hessian shape ({columns}, {columns}) for weight {tuple(weight.shape)}, "
            f"got {tuple(hessian.shape)}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if damp_percent < 0:
        raise ValueError(f"damp_percent must be non-negative, got {damp_percent}")

    del rows
    orig_dtype = weight.dtype
    device = weight.device
    W = weight.detach().float().clone()
    H = hessian.detach().float().to(device=device).clone()
    H = 0.5 * (H + H.t())

    diag_idx = torch.arange(columns, device=device)
    diag = torch.diag(H)
    dead = diag <= eps
    if torch.any(dead):
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    live_diag = torch.diag(H)[~dead]
    damp_base = live_diag.mean() if live_diag.numel() else torch.tensor(1.0, device=device)
    H[diag_idx, diag_idx] += float(damp_percent) * damp_base.clamp_min(eps)

    Hinv = _stable_cholesky_inverse_factor(H, eps=eps)
    scale = _fp8_row_scale(weight.detach().float(), qmax=qmax, eps=eps).to(device=device)

    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        width = end - start
        W_block = W[:, start:end].clone()
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        Hinv_block = Hinv[start:end, start:end]

        for idx in range(width):
            w = W_block[:, idx]
            d = Hinv_block[idx, idx].clamp_min(eps)
            q = fp8_e4m3_qdq_forward(w, scale, qmax=qmax, eps=eps).float()
            Q_block[:, idx] = q
            err = (w - q) / d
            if idx + 1 < width:
                W_block[:, idx + 1 :] -= err.unsqueeze(1).matmul(Hinv_block[idx, idx + 1 :].unsqueeze(0))
            Err_block[:, idx] = err

        W[:, start:end] = Q_block
        if end < columns:
            W[:, end:] -= Err_block.matmul(Hinv[start:end, end:])

    return W.to(orig_dtype)

def gptaq_fp8_quantize_weight(
    weight: torch.Tensor,
    hessian_q: torch.Tensor,
    dxx_t: torch.Tensor,
    *,
    alpha: float = 1.0,
    damp_percent: float = DEFAULT_GPTQ_DAMP_PERCENT,
    block_size: int = DEFAULT_GPTQ_BLOCK_SIZE,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Quantize a Linear weight matrix with GPTAQ asymmetric correction.

    ``hessian_q`` is built from quantized-path Linear inputs. ``dxx_t`` is
    ``(X_fp - X_q)^T X_q`` under the same calibration weighting. When
    ``dxx_t`` is zero, this routine reduces to ``gptq_fp8_quantize_weight``.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    rows, columns = weight.shape
    if hessian_q.shape != (columns, columns):
        raise ValueError(
            f"Expected GPTAQ Hessian shape ({columns}, {columns}) for weight {tuple(weight.shape)}, "
            f"got {tuple(hessian_q.shape)}"
        )
    if dxx_t.shape != (columns, columns):
        raise ValueError(
            f"Expected GPTAQ dxx_t shape ({columns}, {columns}) for weight {tuple(weight.shape)}, "
            f"got {tuple(dxx_t.shape)}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if damp_percent < 0:
        raise ValueError(f"damp_percent must be non-negative, got {damp_percent}")

    del rows
    orig_dtype = weight.dtype
    device = weight.device
    W = weight.detach().float().clone()
    H = hessian_q.detach().float().to(device=device).clone()
    H = 0.5 * (H + H.t())
    D = dxx_t.detach().float().to(device=device).clone()

    diag_idx = torch.arange(columns, device=device)
    diag = torch.diag(H)
    dead = diag <= eps
    if torch.any(dead):
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        D[:, dead] = 0.0

    live_diag = torch.diag(H)[~dead]
    damp_base = live_diag.mean() if live_diag.numel() else torch.tensor(1.0, device=device)
    H[diag_idx, diag_idx] += float(damp_percent) * damp_base.clamp_min(eps)

    Hinv = _stable_cholesky_inverse_factor(H, eps=eps)
    if float(alpha) == 0.0 or torch.count_nonzero(D).item() == 0:
        P = torch.zeros_like(Hinv)
    else:
        P = float(alpha) * torch.triu(D.matmul(Hinv.t()), diagonal=1).matmul(Hinv)
    scale = _fp8_row_scale(weight.detach().float(), qmax=qmax, eps=eps).to(device=device)

    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        width = end - start
        W_block = W[:, start:end].clone()
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        Hinv_block = Hinv[start:end, start:end]
        P_block = P[start:end, start:end]

        for idx in range(width):
            w = W_block[:, idx]
            d = Hinv_block[idx, idx].clamp_min(eps)
            q = fp8_e4m3_qdq_forward(w, scale, qmax=qmax, eps=eps).float()
            Q_block[:, idx] = q
            err = (w - q) / d
            if idx + 1 < width:
                W_block[:, idx + 1 :] -= (
                    err.unsqueeze(1).matmul(Hinv_block[idx, idx + 1 :].unsqueeze(0))
                    - w.unsqueeze(1).matmul(P_block[idx, idx + 1 :].unsqueeze(0))
                )
            Err_block[:, idx] = err

        W[:, start:end] = Q_block
        if end < columns:
            W[:, end:] -= Err_block.matmul(Hinv[start:end, end:]) - Q_block.matmul(P[start:end, end:])

    return W.to(orig_dtype)


def _stable_cholesky_inverse_factor(H: torch.Tensor, *, eps: float) -> torch.Tensor:
    diag_idx = torch.arange(H.shape[0], device=H.device)
    diag_scale = torch.diag(H).detach().abs().mean().clamp_min(float(eps))
    jitter = max(float(eps), float(diag_scale.item()) * 1e-6, 1e-6)
    last_error: RuntimeError | None = None
    for _ in range(12):
        try:
            chol = torch.linalg.cholesky(H)
            inv = torch.cholesky_inverse(chol)
            inv = 0.5 * (inv + inv.t())
            return torch.linalg.cholesky(inv, upper=True)
        except RuntimeError as exc:
            last_error = exc
            H[diag_idx, diag_idx] += jitter
            jitter *= 10.0
    raise RuntimeError("Failed to build GPTQ Hessian inverse factor after adaptive jitter.") from last_error


def _fp8_row_scale(
    weight: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    absmax = weight.detach().float().abs().amax(dim=1)
    return torch.clamp(absmax / qmax, min=eps)


def _gptq_fake_quant_linear(
    linear: nn.Linear,
    hessian: torch.Tensor,
    *,
    act_quant: ActQuant,
    damp_percent: float,
    block_size: int,
) -> GPTQFakeQuantLinear:
    with torch.no_grad():
        weight_qdq = gptq_fp8_quantize_weight(
            linear.weight.detach(),
            hessian,
            damp_percent=damp_percent,
            block_size=block_size,
        )
        bias = None if linear.bias is None else linear.bias.detach().clone()
    return GPTQFakeQuantLinear(
        weight_qdq=weight_qdq,
        bias=bias,
        act_quant=act_quant,
    )


def _replace_children_gptq(
    module: nn.Module,
    *,
    hessians: Mapping[str, torch.Tensor],
    prefix: str,
    act_quant: ActQuant,
    damp_percent: float,
    block_size: int,
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, (BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear)):
            continue
        if isinstance(child, nn.Linear):
            hessian = hessians.get(full_name)
            if hessian is None:
                raise KeyError(f"Missing GPTQ Hessian for Linear module: {full_name}")
            setattr(
                module,
                child_name,
                _gptq_fake_quant_linear(
                    child,
                    hessian,
                    act_quant=act_quant,
                    damp_percent=damp_percent,
                    block_size=block_size,
                ),
            )
            replaced += 1
            continue
        replaced += _replace_children_gptq(
            child,
            hessians=hessians,
            prefix=full_name,
            act_quant=act_quant,
            damp_percent=damp_percent,
            block_size=block_size,
        )
    return replaced


def _named_linear_modules(module: nn.Module) -> dict[str, nn.Linear]:
    if isinstance(module, nn.Linear):
        return {"": module}
    return {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    }
