#!/usr/bin/env bash
set -euo pipefail

# Serial OneRec-1.7B AD-full fake-W4A8 baselines:
#   1. Symmetric RTN INT4-W + dynamic per-token symmetric INT8-A.
#   2. OmniQuant asymmetric LWC-only INT4-W/INT8-A.
#   3. OmniQuant asymmetric LWC + learned LET (SmoothQuant init)
#      INT4-W/INT8-A.
#
# OmniQuant calibration is sequential and runs on the first GPU.  Each full
# evaluation is then sharded across every GPU in GPUS.  The repository default
# calibration size is 128 samples.
#
# Usage after activating the Python environment:
#   bash scripts/fake_quant/run_1p7b_ad_full_w4a8_three_baselines_cuda4567.sh
#
# Optional overrides:
#   GPUS=0,1,2,3 OVERWRITE=1 \
#     bash scripts/fake_quant/run_1p7b_ad_full_w4a8_three_baselines_cuda4567.sh
#   DRY_RUN=1 \
#     bash scripts/fake_quant/run_1p7b_ad_full_w4a8_three_baselines_cuda4567.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-4,5,6,7}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/artifacts/results/fake_quant}"

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "GPUS must contain at least two comma-separated GPU IDs; got ${GPUS}." >&2
    exit 2
fi
CALIB_GPU="${GPU_IDS[0]}"

RTN_FULL_DIR="${RTN_FULL_DIR:-${RESULTS_ROOT}/rtn_sym_int4w_int8a_ad_full}"

LWC_CALIB_DIR="${LWC_CALIB_DIR:-${RESULTS_ROOT}/omniquant_asym_lwc_nolet_int4w_int8a_calib128_ad_calibration}"
LWC_FULL_DIR="${LWC_FULL_DIR:-${RESULTS_ROOT}/omniquant_asym_lwc_nolet_int4w_int8a_calib128_ad_full}"
LWC_CHECKPOINT_DIR="${LWC_CALIB_DIR}/1.7B/ad/omniquant_calibration"

LET_CALIB_DIR="${LET_CALIB_DIR:-${RESULTS_ROOT}/omniquant_asym_lwc_learned_let_smoothquant_int4w_int8a_calib128_ad_calibration}"
LET_FULL_DIR="${LET_FULL_DIR:-${RESULTS_ROOT}/omniquant_asym_lwc_learned_let_smoothquant_int4w_int8a_calib128_ad_full}"
LET_CHECKPOINT_DIR="${LET_CALIB_DIR}/1.7B/ad/omniquant_calibration"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    WRITE_ARGS+=(--overwrite)
fi

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_calibration() {
    local label="$1"
    local output_dir="$2"
    shift 2
    local command=("$@")

    echo "[${label}] calibration_gpu=${CALIB_GPU} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CALIB_GPU}"
        print_command "${command[@]}"
        return
    fi

    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${CALIB_GPU}" \
        "${command[@]}" 2>&1 | tee "${output_dir}/calibration.log"
}

check_omniquant_checkpoints() {
    local checkpoint_dir="$1"
    local layer_idx checkpoint_name
    for ((layer_idx = 0; layer_idx < 28; layer_idx++)); do
        printf -v checkpoint_name 'layer_%02d.pt' "${layer_idx}"
        if [[ ! -f "${checkpoint_dir}/${checkpoint_name}" ]]; then
            echo "Missing OmniQuant checkpoint: ${checkpoint_dir}/${checkpoint_name}" >&2
            return 1
        fi
    done
}

run_sharded_eval() {
    local label="$1"
    local output_dir="$2"
    shift 2

    echo "[${label}] evaluation_gpus=${GPUS} output=${output_dir}"
    GPUS="${GPUS}" \
    OUTPUT_DIR="${output_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OVERWRITE="${OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh "$@"
}

echo "Stage 1/3: symmetric RTN INT4-W/INT8-A, AD full."
run_sharded_eval \
    "RTN W4A8" \
    "${RTN_FULL_DIR}" \
    --mode baseline_qdq \
    --weight_quant_format int4 \
    --weight_quant_scheme symmetric \
    --activation_quant_format int8

echo "Stage 2/3: OmniQuant asymmetric LWC-only INT4-W/INT8-A, AD full."
run_calibration \
    "OmniQuant LWC-only" \
    "${LWC_CALIB_DIR}" \
    "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad \
    --mode omniquant \
    --weight_quant_format int4 \
    --activation_quant_format int8 \
    --omni_weight_quant_scheme asymmetric \
    --no-omni-let \
    --calibration_only \
    --output_dir "${LWC_CALIB_DIR}" \
    "${WRITE_ARGS[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
    check_omniquant_checkpoints "${LWC_CHECKPOINT_DIR}"
fi

run_sharded_eval \
    "OmniQuant LWC-only" \
    "${LWC_FULL_DIR}" \
    --mode omniquant \
    --weight_quant_format int4 \
    --activation_quant_format int8 \
    --omni_weight_quant_scheme asymmetric \
    --no-omni-let \
    --omni_load_checkpoint_dir "${LWC_CHECKPOINT_DIR}"

echo "Stage 3/3: OmniQuant asymmetric LWC + learned SQ-init LET INT4-W/INT8-A, AD full."
run_calibration \
    "OmniQuant LWC + SQ-init LET" \
    "${LET_CALIB_DIR}" \
    "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad \
    --mode omniquant \
    --weight_quant_format int4 \
    --activation_quant_format int8 \
    --omni_weight_quant_scheme asymmetric \
    --calibration_only \
    --output_dir "${LET_CALIB_DIR}" \
    "${WRITE_ARGS[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
    check_omniquant_checkpoints "${LET_CHECKPOINT_DIR}"
fi

run_sharded_eval \
    "OmniQuant LWC + SQ-init LET" \
    "${LET_FULL_DIR}" \
    --mode omniquant \
    --weight_quant_format int4 \
    --activation_quant_format int8 \
    --omni_weight_quant_scheme asymmetric \
    --omni_load_checkpoint_dir "${LET_CHECKPOINT_DIR}"

echo "Completed all three serial fake-W4A8 AD-full baselines."
echo "RTN:              ${RTN_FULL_DIR}/eval_results.json"
echo "LWC-only:         ${LWC_FULL_DIR}/eval_results.json"
echo "LWC + SQ-init LET: ${LET_FULL_DIR}/eval_results.json"
