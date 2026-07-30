__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "Generator",
    "GenerationRunner",
]


def __getattr__(name: str):
    """Keep legacy exports available without importing the runner on package load."""
    if name == "Benchmark":
        from benchmark.benchmark import Benchmark

        return Benchmark
    if name == "Generator":
        from benchmark.base_generator import Generator

        return Generator
    if name == "GenerationRunner":
        from benchmark.generation_runner import GenerationRunner

        return GenerationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
