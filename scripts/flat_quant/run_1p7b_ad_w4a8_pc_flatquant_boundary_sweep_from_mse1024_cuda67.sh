#!/usr/bin/env bash
set -euo pipefail

# Strict W4A8 FlatQuant boundary sweep. Every arm starts from the same completed
# MSE-1024 checkpoint and freezes matrix/diagonal transforms while training
# final-layer LWC/LAC on all 1024 AD calibration records.
#
# Run one arm at a time, for example:
#   BOUNDARY_WEIGHTS=0 RUN_EVAL=0 bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_flatquant_boundary_sweep_from_mse1024_cuda67.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
BOUNDARY_WEIGHTS="${BOUNDARY_WEIGHTS:-0 0.3 0.5}"
GPUS="${GPUS:-6,7}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_ad_w4a8_pc_from_mse_train1024_epoch15/mse1024_frozen_transform_control/1.7B/ad/flatquant_calibration}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_ad_w4a8_pc_boundary_sweep_from_mse1024_train1024_epoch15}"
CORE_SCRIPT="${REPO_ROOT}/scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh"

if [[ -z "${BOUNDARY_WEIGHTS//[[:space:]]/}" ]]; then
    echo "BOUNDARY_WEIGHTS must contain at least one non-negative value." >&2
    exit 2
fi

for boundary_weight in ${BOUNDARY_WEIGHTS}; do
    if ! python -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and x >= 0))' "${boundary_weight}"; then
        echo "Invalid boundary weight: ${boundary_weight}" >&2
        exit 2
    fi
    boundary_tag="${boundary_weight//./p}"
    if python -c 'import sys; sys.exit(float(sys.argv[1]) != 0)' "${boundary_weight}"; then
        arm_name="abc_only"
    else
        arm_name="abc_boundary_w${boundary_tag}"
    fi
    output_dir="${RUN_ROOT}/${arm_name}"
    eval_dir="${RUN_ROOT}/${arm_name}_ad_full_eval"
    echo
    echo "[boundary sweep] arm=${arm_name} boundary_weight=${boundary_weight}"
    env \
        TASK=ad \
        GPUS="${GPUS}" \
        SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR}" \
        OUTPUT_DIR="${output_dir}" \
        EVAL_OUTPUT_DIR="${eval_dir}" \
        BOUNDARY_WEIGHT="${boundary_weight}" \
        LFQ_LOSS_WEIGHT=1.0 \
        LEARN_TRANSFORM=0 \
        LEARN_LAC=1 \
        RUN_EVAL="${RUN_EVAL}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash "${CORE_SCRIPT}"
done

echo
echo "[done] boundary sweep root=${RUN_ROOT}"
