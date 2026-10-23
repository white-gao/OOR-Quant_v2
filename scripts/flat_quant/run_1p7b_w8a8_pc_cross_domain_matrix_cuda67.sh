#!/usr/bin/env bash
set -euo pipefail

# Four evaluation-only runs, serial: Product -> AD/Video, Video -> AD/Product.
# Reuse three diagonal results and both completed AD transfers without changes.
# Default GPUS=4,5; each evaluation uses one independent shard per GPU.
# DRY_RUN=1 prints commands without GPU work or file writes.
# Completed evaluations are validated and skipped on a normal rerun.
# EVAL_OVERWRITE=1 reruns selected new transfers only; no model is retrained.
# PAIRS="video:product" optionally selects a subset of the four new directions.
# RUN_ROOT overrides the new matrix output; AD_RESULTS_ROOT locates old AD runs.
# --check-only validates inputs; --collect-only refreshes the nine-cell summary.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GPUS="${GPUS:-4,5}"
exec "${PYTHON_BIN}" -u -m flat_quant.cross_domain_eval --complete-matrix "$@"
