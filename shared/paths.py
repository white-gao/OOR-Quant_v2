"""Canonical, overridable locations for models, data, and experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SERVER_STORAGE_ROOT = Path("/root/dataDisk/guowei")


def repo_root() -> Path:
    """Return the repository root without depending on the working directory."""

    return _REPO_ROOT


def artifacts_root() -> Path:
    """Return the root for generated, Git-ignored experiment artifacts."""

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
    """Return the root containing reusable model checkpoints."""

    configured = os.environ.get("OOR_QUANT_MODEL_ROOT")
    return Path(configured).expanduser() if configured else _SERVER_STORAGE_ROOT / "models"


def data_root() -> Path:
    """Return the root containing reusable datasets."""

    configured = os.environ.get("OOR_QUANT_DATA_ROOT")
    return Path(configured).expanduser() if configured else _SERVER_STORAGE_ROOT / "data"


def benchmark_data_root() -> Path:
    """Return the OpenOneRec benchmark-data directory."""

    configured = os.environ.get("OOR_QUANT_BENCHMARK_DATA")
    if configured:
        return Path(configured).expanduser()
    return data_root() / "onerec_data" / "benchmark_data"
