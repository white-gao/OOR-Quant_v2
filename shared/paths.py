"""Canonical, overridable locations for non-source project artifacts.

Keeping models, datasets, and experiment outputs under one ignored root makes
the source tree portable and prevents real-quant and fake-quant results from
being mixed accidentally.  Set ``OOR_QUANT_ARTIFACTS`` to place these files on
another filesystem; by default they live in ``<repo>/artifacts``.
"""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the repository root without depending on the working directory."""

    return _REPO_ROOT


def artifacts_root() -> Path:
    """Return the root for ignored, potentially large project artifacts."""

    configured = os.environ.get("OOR_QUANT_ARTIFACTS")
    return Path(configured).expanduser() if configured else repo_root() / "artifacts"


def artifacts_path(*parts: str) -> Path:
    """Build a path below :func:`artifacts_root`."""

    return artifacts_root().joinpath(*parts)


def real_results_root() -> Path:
    """Root for results produced by the actual FP8/W8A8 runtime."""

    return artifacts_path("results", "real_quant")


def fake_results_root() -> Path:
    """Root for fake-QDQ and learnable-quant research outputs."""

    return artifacts_path("results", "fake_quant")


def benchmark_results_root() -> Path:
    """Root for imported/legacy OpenOneRec benchmark artifacts."""

    return artifacts_path("results", "benchmark_legacy")


def model_root() -> Path:
    """Root reserved for local model checkpoints."""

    return artifacts_path("models")


def data_root() -> Path:
    """Root reserved for local datasets."""

    return artifacts_path("data")
