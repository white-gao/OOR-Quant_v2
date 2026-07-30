"""
Item Understand Task Module
"""

from .config import ITEM_UNDERSTAND_CONFIG
__all__ = [
    "ITEM_UNDERSTAND_CONFIG",
    "ItemUnderstandEvaluator",
    "utils",
]


def __getattr__(name: str):
    if name == "ItemUnderstandEvaluator":
        from .evaluator import ItemUnderstandEvaluator

        return ItemUnderstandEvaluator
    if name == "utils":
        from importlib import import_module

        return import_module(f"{__name__}.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
