#!/usr/bin/env bash
set -euo pipefail

# Four missing loss-ablation arms, serial: W8A8 then W4A8, each without
# boundary / without distribution alignment. Reuse completed full-loss and
# original MSE evaluations. The Python driver validates their provenance.
# DRY_RUN=1 prints commands without loading a model or writing outputs.
# --collect-only refreshes the CSV/JSON without training or evaluation.
# GPUS=2,3 by default: train on the first GPU, evaluate on all listed GPUs.
# Optional subsets: PRECISIONS="w4a8" VARIANTS="wo_distribution".
# RUN_EVAL=0 trains only; rerunning with RUN_EVAL=1 reuses complete checkpoints.
# EVAL_OVERWRITE=1 reruns interrupted evaluation without forcing retraining.
# Protocol knobs are pinned to the references; use RUN_ROOT for a new output root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -u -m flat_quant.loss_ablation "$@"
