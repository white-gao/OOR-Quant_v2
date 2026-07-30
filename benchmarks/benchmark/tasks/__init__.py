"""
Tasks definition for Benchmark
"""

__all__ = [
    "BenchmarkTable",
    "check_benchmark_version",
    "check_task_types",
    "check_splits",
    "LATEST_BENCHMARK_VERSION",
]


def __getattr__(name: str):
    if name in __all__:
        from . import tasks as task_definitions

        return getattr(task_definitions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
