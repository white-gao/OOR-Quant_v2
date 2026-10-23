#!/usr/bin/env bash
set -euo pipefail

# Evaluation only: one AD-calibrated RecQuant model -> Product, then Video.
# Reuse the AD result and all three in-domain RecQuant reference evaluations.
# Default GPUS=6,7; override GPUS to select different cards.
# DRY_RUN=1 prints commands without GPU work or file writes.
# EVAL_OVERWRITE=1 reruns interrupted evaluations, never retrains the model.
# --check-only validates inputs; --collect-only refreshes CSV/JSON summaries.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GPUS="${GPUS:-6,7}"
exec "${PYTHON_BIN}" -u -m flat_quant.cross_domain_eval "$@"
