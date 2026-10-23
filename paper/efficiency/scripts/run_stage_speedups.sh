#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
GPU="${GPU:-6}"
if [[ "${GPU}" == *,* ]]; then
    echo "Select one physical GPU with GPU=..." >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OOR_QUANT_ENABLE_FUSED_FP8_PROJECTIONS=0
export TOKENIZERS_PARALLELISM=false
CHECK_ONLY=0
for arg in "$@"; do
    if [[ "${arg}" == "--check-only" || "${arg}" == "--summarize-only" ]]; then CHECK_ONLY=1; fi
done
if [[ "${CHECK_ONLY}" == 0 && "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    ACTIVE_PIDS="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -n "${ACTIVE_PIDS}" ]]; then
        echo "GPU ${GPU} has active processes. Use an idle GPU for timing; no task was launched." >&2
        echo "ALLOW_BUSY_GPU=1 bypasses this check but invalidates clean latency comparisons." >&2
        exit 2
    fi
fi
echo "[launcher] GPU=${GPU}; no training; BF16/RTN measured, RecQuant operator-estimated"
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/collect_stage_speedups.py" "$@"
