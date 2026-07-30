"""
v1.0 Version Task Definitions
"""

__all__ = ["TaskTable"]


def __getattr__(name: str):
    if name == "TaskTable":
        from .registry import TaskTable

        return TaskTable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
