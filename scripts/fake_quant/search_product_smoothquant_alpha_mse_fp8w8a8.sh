#!/usr/bin/env bash
set -euo pipefail

# Product-domain SmoothQuant alpha search for deployment-matched FP8 W8A8.
#
# The underlying search evaluates a no-SmoothQuant RTN control and every alpha
# from 0.0 to 1.0 in increments of 0.1 on the same Product calibration samples.
# It compares local transformer-block output MSE on identical FP block inputs;
# recommendation generation/evaluation is not run.
#
# Usage:
#   GPU=0 bash scripts/fake_quant/search_product_smoothquant_alpha_mse_fp8w8a8.sh
#
# Optional overrides:
#   GPU=6 CALIB_SAMPLE_SIZE=128 bash scripts/fake_quant/search_product_smoothquant_alpha_mse_fp8w8a8.sh
#   OVERWRITE=1 bash scripts/fake_quant/search_product_smoothquant_alpha_mse_fp8w8a8.sh
#
# Additional arguments are forwarded to the Python search command.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU="${GPU:-0}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
CALIB_OFFSET="${CALIB_OFFSET:-0}"
ALPHAS="${ALPHAS:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/results/fake_quant/smoothquant_alpha_mse_fp8w8a8_product_calib${CALIB_SAMPLE_SIZE}}"
OVERWRITE="${OVERWRITE:-0}"

exec env \
    GPU="${GPU}" \
    TASK=product \
    CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE}" \
    CALIB_OFFSET="${CALIB_OFFSET}" \
    ALPHAS="${ALPHAS}" \
    LAYERS=all \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    OVERWRITE="${OVERWRITE}" \
    bash "${SCRIPT_DIR}/search_smoothquant_alpha_mse_fp8w8a8.sh" "$@"
