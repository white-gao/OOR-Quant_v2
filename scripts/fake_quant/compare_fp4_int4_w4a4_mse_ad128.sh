#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/fake_quant/probes/fp4w_fp8a_w4a8_mse_ad_calib${CALIB_SAMPLE_SIZE}}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m fake_quant.compare_w4a4_format_mse \
    --task ad \
    --device cuda:0 \
    --calib_sample_size "${CALIB_SAMPLE_SIZE}" \
    --settings fp4w_fp8a \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
