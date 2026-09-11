#!/usr/bin/env bash
set -euo pipefail

# Run six full 1.7B FP8-W/FP8-A recommendation baselines serially:
#   AD:      RTN, SmoothQuant, OmniQuant LWC + learned LET (SQ init)
#   Product: RTN, SmoothQuant, OmniQuant LWC + learned LET (SQ init)
#
# Default protocol:
#   deployment-matched BF16 execution
#   FP8 E4M3 weight per-output-channel QDQ
#   FP8 E4M3 activation dynamic per-token shared-input QDQ
#   SmoothQuant alpha=0.4
#   calibration samples=128
#   OmniQuant epochs=20, LWC LR=1e-2, LET LR=5e-3
#
# Defaults above are owned by the Python runner and intentionally not repeated
# when they do not change command behavior.
#
# Usage after activating the Python environment:
#   bash scripts/fake_quant/run_1p7b_ad_product_fp8w8a8_six_baselines_cuda01.sh
#
# Optional:
#   GPUS=0,1 OVERWRITE=1 #     bash scripts/fake_quant/run_1p7b_ad_product_fp8w8a8_six_baselines_cuda01.sh
#   DRY_RUN=1 #     bash scripts/fake_quant/run_1p7b_ad_product_fp8w8a8_six_baselines_cuda01.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
GPUS="${GPUS:-0,1}"
TASKS="${TASKS:-ad product}"
METHODS="${METHODS:-rtn smoothquant omniquant}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_product_full_fp8w8a8_deployment_matched}"

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "GPUS must contain at least two comma-separated GPU IDs; got ${GPUS}." >&2
    exit 2
fi
CALIB_GPU="${GPU_IDS[0]}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    WRITE_ARGS+=(--overwrite)
fi

print_command() {
    printf '%q ' "$@"
    printf '
'
}

run_sharded_eval() {
    local label="$1"
    local task="$2"
    local output_dir="$3"
    shift 3

    echo "[${label}] task=${task} evaluation_gpus=${GPUS} output=${output_dir}"
    TASK="${task}"     GPUS="${GPUS}"     MODEL_PATH="${MODEL_PATH}"     DATA_DIR="${DATA_DIR}"     OUTPUT_DIR="${output_dir}"     PYTHON_BIN="${PYTHON_BIN}"     OVERWRITE="${OVERWRITE}"     DRY_RUN="${DRY_RUN}"         bash scripts/fake_quant/run_sharded_eval_cuda.sh "$@"
}

run_omniquant_calibration() {
    local task="$1"
    local output_dir="$2"
    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        --task "${task}"
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --output_dir "${output_dir}"
        --device cuda:0
        --mode omniquant
        --weight_quant_format fp8_e4m3fn
        --activation_quant_format fp8_e4m3fn
        --calibration_only
        "${WRITE_ARGS[@]}"
    )

    echo "[OmniQuant calibration] task=${task} physical_gpu=${CALIB_GPU} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CALIB_GPU}"
        print_command "${command[@]}"
        return
    fi

    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${CALIB_GPU}" "${command[@]}" 2>&1         | tee "${output_dir}/calibration.log"
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

has_method() {
    local method="$1"
    [[ " ${METHODS} " == *" ${method} "* ]]
}

total_stages=0
for _task in ${TASKS}; do
    for _method in rtn smoothquant omniquant; do
        if has_method "${_method}"; then
            total_stages=$((total_stages + 1))
        fi
    done
done
if (( total_stages == 0 )); then
    echo "No experiments selected: TASKS='${TASKS}' METHODS='${METHODS}'." >&2
    exit 2
fi

stage=0
for task in ${TASKS}; do
    if [[ "${task}" != "ad" && "${task}" != "product" ]]; then
        echo "Unsupported task in TASKS: ${task}" >&2
        exit 2
    fi

    rtn_dir="${RESULTS_ROOT}/rtn_fp8w8a8_${task}_full"
    sq_dir="${RESULTS_ROOT}/smoothquant_fp8w8a8_alpha0p4_${task}_calib128_full"
    omni_calib_dir="${RESULTS_ROOT}/omniquant_sym_lwc_let_sqinit_fp8w8a8_alpha0p4_${task}_calib128_ep20_calibration"
    omni_eval_dir="${RESULTS_ROOT}/omniquant_sym_lwc_let_sqinit_fp8w8a8_alpha0p4_${task}_calib128_ep20_full"
    omni_checkpoint_dir="${omni_calib_dir}/1.7B/${task}/omniquant_calibration"

    if has_method rtn; then
        stage=$((stage + 1))
        echo "Stage ${stage}/${total_stages}: ${task} RTN FP8 W8A8."
        run_sharded_eval             "RTN FP8 W8A8"             "${task}"             "${rtn_dir}"             --mode baseline_qdq             --weight_quant_format fp8_e4m3fn             --activation_quant_format fp8_e4m3fn
    fi

    if has_method smoothquant; then
        stage=$((stage + 1))
        echo "Stage ${stage}/${total_stages}: ${task} SmoothQuant FP8 W8A8."
        run_sharded_eval             "SmoothQuant FP8 W8A8"             "${task}"             "${sq_dir}"             --mode smoothquant_w8a8             --weight_quant_format fp8_e4m3fn             --activation_quant_format fp8_e4m3fn
    fi

    if has_method omniquant; then
        stage=$((stage + 1))
        echo "Stage ${stage}/${total_stages}: ${task} OmniQuant LWC + learned SQ-init LET FP8 W8A8."
        run_omniquant_calibration "${task}" "${omni_calib_dir}"
        if [[ "${DRY_RUN}" != "1" ]]; then
            check_omniquant_checkpoints "${omni_checkpoint_dir}"
        fi
        run_sharded_eval             "OmniQuant FP8 W8A8"             "${task}"             "${omni_eval_dir}"             --mode omniquant             --weight_quant_format fp8_e4m3fn             --activation_quant_format fp8_e4m3fn             --omni_load_checkpoint_dir "${omni_checkpoint_dir}"
    fi
done
echo "Completed ${stage} FP8 W8A8 baseline experiment(s)."
for task in ${TASKS}; do
    if has_method rtn; then
        echo "RTN ${task}: ${RESULTS_ROOT}/rtn_fp8w8a8_${task}_full/eval_results.json"
    fi
    if has_method smoothquant; then
        echo "SQ ${task}:  ${RESULTS_ROOT}/smoothquant_fp8w8a8_alpha0p4_${task}_calib128_full/eval_results.json"
    fi
    if has_method omniquant; then
        echo "OmniQuant ${task}: ${RESULTS_ROOT}/omniquant_sym_lwc_let_sqinit_fp8w8a8_alpha0p4_${task}_calib128_ep20_full/eval_results.json"
    fi
done
