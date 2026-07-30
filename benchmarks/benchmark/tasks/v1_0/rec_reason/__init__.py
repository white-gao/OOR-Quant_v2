"""
Recommendation Reason Task Module
"""

from .config import REC_REASON_CONFIG
__all__ = [
    "REC_REASON_CONFIG",
    "RecoReasonEvaluator",
    "utils",
]


def __getattr__(name: str):
    if name == "RecoReasonEvaluator":
        from .evaluator import RecoReasonEvaluator

        return RecoReasonEvaluator
    if name == "utils":
        from importlib import import_module

        return import_module(f"{__name__}.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
