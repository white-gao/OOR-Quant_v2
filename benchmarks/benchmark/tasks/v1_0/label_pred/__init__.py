"""
Label Prediction Task Module

Classification task for predicting user engagement with video content.
Uses logprobs-based classification with AUC and wuAUC metrics.
"""

from .config import LABEL_PRED_CONFIG
__all__ = [
    "LABEL_PRED_CONFIG",
    "LabelPredEvaluator",
    "utils",
]


def __getattr__(name: str):
    if name == "LabelPredEvaluator":
        from .evaluator import LabelPredEvaluator

        return LabelPredEvaluator
    if name == "utils":
        from importlib import import_module

        return import_module(f"{__name__}.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
