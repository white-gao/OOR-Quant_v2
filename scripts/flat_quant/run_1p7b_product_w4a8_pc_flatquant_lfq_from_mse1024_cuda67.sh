#!/usr/bin/env bash
set -euo pipefail

# Product generalization arm: start from the strict Product MSE-1024
# checkpoint, freeze FlatQuant matrices/diagonals, and optimize final-layer
# LWC/LAC with ABC CE + 0.3 boundary on all 1024 records.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
GPUS="${GPUS:-6,7}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PRODUCT_ROOT="${PRODUCT_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/generalization/1p7b_product_w4a8_pc_prefix128_mse512_mse1024_epoch15}"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-${PRODUCT_ROOT}/final_mse1024_frozen_transform/1.7B/product/flatquant_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRODUCT_ROOT}/final_abc_boundary_w0p3_from_mse1024}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRODUCT_ROOT}/abc_boundary_w0p3_product_full_eval}"

exec env \
    TASK=product \
    GPUS="${GPUS}" \
    SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
    BOUNDARY_WEIGHT=0.3 \
    LFQ_LOSS_WEIGHT=1.0 \
    LEARN_TRANSFORM=0 \
    LEARN_LAC=1 \
    RUN_EVAL="${RUN_EVAL}" \
    OVERWRITE="${OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${REPO_ROOT}/scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh"
