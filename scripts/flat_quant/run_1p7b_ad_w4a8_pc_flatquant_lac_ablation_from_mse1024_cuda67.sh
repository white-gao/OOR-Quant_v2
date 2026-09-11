#!/usr/bin/env bash
set -euo pipefail

# FlatQuant LAC contribution ablation. Restored MSE-trained LAC values remain
# active in the forward pass but are frozen during final-layer LFQ; only LWC is
# optimized. Compare against the learn-LAC boundary=0.3 arm from the boundary
# sweep, which shares the same MSE-1024 initialization.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
GPUS="${GPUS:-6,7}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_ad_w4a8_pc_from_mse_train1024_epoch15/mse1024_frozen_transform_control/1.7B/ad/flatquant_calibration}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_ad_w4a8_pc_lac_ablation_from_mse1024_train1024_epoch15}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/abc_boundary_w0p3_lwc_only_lac_frozen}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/abc_boundary_w0p3_lwc_only_lac_frozen_ad_full_eval}"

exec env \
    TASK=ad \
    GPUS="${GPUS}" \
    SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
    BOUNDARY_WEIGHT=0.3 \
    LFQ_LOSS_WEIGHT=1.0 \
    LEARN_TRANSFORM=0 \
    LEARN_LAC=0 \
    RUN_EVAL="${RUN_EVAL}" \
    OVERWRITE="${OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${REPO_ROOT}/scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh"
