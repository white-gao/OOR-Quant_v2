from __future__ import annotations

import torch


def compute_smooth_scale(
    activation_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute SmoothQuant per-input-channel scales."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if activation_absmax.shape != weight_absmax.shape:
        raise ValueError(
            "activation_absmax and weight_absmax must have the same shape, "
            f"got {tuple(activation_absmax.shape)} and {tuple(weight_absmax.shape)}"
        )

    x = torch.clamp(activation_absmax.detach().float(), min=eps)
    w = torch.clamp(weight_absmax.detach().float(), min=eps)
    return torch.pow(x, alpha) / torch.pow(w, 1.0 - alpha)


def smooth_linear_weight(weight: torch.Tensor, smooth_scale: torch.Tensor) -> torch.Tensor:
    """Fold SmoothQuant input scales into Linear weight columns."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    if smooth_scale.ndim != 1 or smooth_scale.numel() != weight.shape[1]:
        raise ValueError(
            f"Expected scale shape ({weight.shape[1]},), got {tuple(smooth_scale.shape)}"
        )
    return weight * smooth_scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)
