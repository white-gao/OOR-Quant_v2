#!/usr/bin/env bash
set -euo pipefail

# Search SmoothQuant alpha by local block-output reconstruction MSE.
#
# The search uses FP8-W/FP8-A, dynamic per-token activation QDQ, the
# deployment-matched execution path, and one fixed AD calibration subset.
# A no-SmoothQuant FP8 W8A8 RTN control is evaluated on the same block inputs.
# It does not run recommendation generation/evaluation.
#
# Usage:
#   GPU=0 bash scripts/fake_quant/search_smoothquant_alpha_mse_fp8w8a8.sh
#
# Common overrides:
#   GPU=5 CALIB_SAMPLE_SIZE=128 ALPHAS=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 #     bash scripts/fake_quant/search_smoothquant_alpha_mse_fp8w8a8.sh
#   ALPHAS=0.35,0.4,0.45,0.5,0.55 OVERWRITE=1 #     bash scripts/fake_quant/search_smoothquant_alpha_mse_fp8w8a8.sh
#
# Additional Python arguments may be appended to the command.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
TASK="${TASK:-ad}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
CALIB_OFFSET="${CALIB_OFFSET:-0}"
ALPHAS="${ALPHAS:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}"
LAYERS="${LAYERS:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/results/fake_quant/smoothquant_alpha_mse_fp8w8a8_${TASK}_calib${CALIB_SAMPLE_SIZE}}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "${PYTHON_BIN}" -u -m fake_quant.search_smoothquant_alpha_mse
    --task "${TASK}"
    --calib_sample_size "${CALIB_SAMPLE_SIZE}"
    --calib_offset "${CALIB_OFFSET}"
    --alphas "${ALPHAS}"
    --layers "${LAYERS}"
    --device cuda:0
    --dtype bfloat16
    --act_quant_mode shared_input
    --smooth_scope omni
    --smooth_fold
    --output_dir "${OUTPUT_DIR}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMAND+=(--overwrite)
fi
COMMAND+=("$@")

echo "[sq_alpha_mse] physical_gpu=${GPU}"
echo "[sq_alpha_mse] alphas=${ALPHAS}"
echo "[sq_alpha_mse] calibration=${TASK}:${CALIB_OFFSET}+${CALIB_SAMPLE_SIZE}"
echo "[sq_alpha_mse] output=${OUTPUT_DIR}"
printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
printf '%q ' "${COMMAND[@]}"
printf '
'

CUDA_VISIBLE_DEVICES="${GPU}" "${COMMAND[@]}"
