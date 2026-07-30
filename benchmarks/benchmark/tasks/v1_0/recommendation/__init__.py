"""
Recommendation Task Module

Universal module for all recommendation tasks including:
- label_cond: Predict next video given specified consumption behavior
- video: Next video prediction
- product: Predict next clicked product
- ad: Predict next clicked advertisement
"""

from .config import (
    LABEL_COND_CONFIG,
    VIDEO_CONFIG,
    PRODUCT_CONFIG,
    AD_CONFIG,
    INTERACTIVE_CONFIG,
    RECOMMENDATION_PROMPT_CONFIG,
    RECOMMENDATION_TASK_CONFIGS,
    RECOMMENDATION_GENERATION_CONFIG,
    RECOMMENDATION_EVALUATION_CONFIG,
)
__all__ = [
    # Configs
    "LABEL_COND_CONFIG",
    "VIDEO_CONFIG",
    "PRODUCT_CONFIG",
    "AD_CONFIG",
    "INTERACTIVE_CONFIG",
    "RECOMMENDATION_PROMPT_CONFIG",
    "RECOMMENDATION_TASK_CONFIGS",
    "RECOMMENDATION_GENERATION_CONFIG",
    "RECOMMENDATION_EVALUATION_CONFIG",
    # Classes
    "RecommendationEvaluator",
    # Utils module
    "utils",
]


def __getattr__(name: str):
    if name == "RecommendationEvaluator":
        from .evaluator import RecommendationEvaluator

        return RecommendationEvaluator
    if name == "utils":
        from importlib import import_module

        return import_module(f"{__name__}.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
