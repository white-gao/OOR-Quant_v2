from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def _detach_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_detach_tree(item) for item in obj)
    if isinstance(obj, list):
        return [_detach_tree(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _detach_tree(value) for key, value in obj.items()}
    return obj


def _module_device(module: nn.Module) -> torch.device | None:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _move_tree_to_device(obj: Any, device: torch.device | None) -> Any:
    if device is None:
        return obj
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_move_tree_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_tree_to_device(item, device) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _move_tree_to_device(value, device) for key, value in obj.items()}
    return obj
