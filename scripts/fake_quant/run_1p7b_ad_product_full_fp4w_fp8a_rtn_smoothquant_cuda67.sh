#!/usr/bin/env bash
set -euo pipefail

# Serial deployment-matched fake-QDQ baselines for OneRec-1.7B:
#   1. AD full:      RTN FP4-E2M1-W / FP8-E4M3-A.
#   2. AD full:      SmoothQuant FP4-E2M1-W / FP8-E4M3-A.
#   3. Product full: RTN FP4-E2M1-W / FP8-E4M3-A.
#   4. Product full: SmoothQuant FP4-E2M1-W / FP8-E4M3-A.
#
# Weight QDQ is symmetric per output channel (group size 0). Activation QDQ is
# dynamic per token. SmoothQuant uses alpha=0.4 and the first 128 calibration
# records. Each full evaluation is sharded over GPUs 6 and 7 by default, and
# the next method starts only after the current method has merged its metrics.
#
# Usage:
#   bash scripts/fake_quant/run_1p7b_ad_product_full_fp4w_fp8a_rtn_smoothquant_cuda67.sh
#
# Useful overrides:
#   GPUS=4,5 bash scripts/fake_quant/run_1p7b_ad_product_full_fp4w_fp8a_rtn_smoothquant_cuda67.sh
#   DRY_RUN=1 bash scripts/fake_quant/run_1p7b_ad_product_full_fp4w_fp8a_rtn_smoothquant_cuda67.sh
#   OVERWRITE=1 bash scripts/fake_quant/run_1p7b_ad_product_full_fp4w_fp8a_rtn_smoothquant_cuda67.sh

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
GPUS="${GPUS:-6,7}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-128}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.4}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_product_full_fp4w_fp8a_rtn_smoothquant_deployment_matched}"
AD_RTN_DIR="${AD_RTN_DIR:-${RESULTS_ROOT}/rtn_fp4w_fp8a_ad_full}"
AD_SQ_DIR="${AD_SQ_DIR:-${RESULTS_ROOT}/smoothquant_fp4w_fp8a_alpha0p4_ad_calib${CALIB_SAMPLE_SIZE}_full}"
PRODUCT_RTN_DIR="${PRODUCT_RTN_DIR:-${RESULTS_ROOT}/rtn_fp4w_fp8a_product_full}"
PRODUCT_SQ_DIR="${PRODUCT_SQ_DIR:-${RESULTS_ROOT}/smoothquant_fp4w_fp8a_alpha0p4_product_calib${CALIB_SAMPLE_SIZE}_full}"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}." >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1; got ${DRY_RUN}." >&2
    exit 2
fi
if [[ ! "${CALIB_SAMPLE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CALIB_SAMPLE_SIZE must be a positive integer; got ${CALIB_SAMPLE_SIZE}." >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and 0.0 <= x <= 1.0))' "${SMOOTHQUANT_ALPHA}"; then
    echo "SMOOTHQUANT_ALPHA must be finite and in [0, 1]; got ${SMOOTHQUANT_ALPHA}." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "GPUS must contain at least two comma-separated GPU IDs; got ${GPUS}." >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID in GPUS=${GPUS}: ${gpu_id}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "Duplicate GPU ID in GPUS=${GPUS}: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
done

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_full_eval() {
    local stage="$1"
    local label="$2"
    local task="$3"
    local method="$4"
    local output_dir="$5"

    if [[ -s "${output_dir}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[${stage} ${label}] complete eval_results.json found; skipping."
        return
    fi

    local method_args=()
    case "${method}" in
        rtn)
            method_args+=(--mode baseline_qdq)
            ;;
        smoothquant)
            method_args+=(
                --mode smoothquant_w8a8
                --smoothquant_alpha "${SMOOTHQUANT_ALPHA}"
                --calib_sample_size "${CALIB_SAMPLE_SIZE}"
            )
            ;;
        *)
            echo "Unknown baseline method: ${method}" >&2
            return 2
            ;;
    esac

    echo "[${stage} ${label}] task=${task} evaluation_gpus=${GPUS} output=${output_dir}"
    TASK="${task}" \
    GPUS="${GPUS}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${output_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OVERWRITE="${OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh \
        "${method_args[@]}" \
        --weight_quant_format fp4_e2m1 \
        --activation_quant_format fp8_e4m3fn \
        --weight_quant_scheme symmetric \
        --weight_group_size 0 \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}"
}

echo "[protocol] model=${MODEL_PATH} tasks=ad,product seed=42"
echo "[protocol] deployment-matched fake-QDQ FP4-E2M1-W/FP8-E4M3-A, symmetric per-output-channel weight scaling"
echo "[protocol] activation=dynamic per-token; SmoothQuant alpha=${SMOOTHQUANT_ALPHA} calib=[0,${CALIB_SAMPLE_SIZE})"
echo "[protocol] each AD/Product ${EVAL_SAMPLE_SIZE} evaluation is sharded across GPUS=${GPUS}"
echo "[protocol] four stages execute serially; results_root=${RESULTS_ROOT}"

run_full_eval "1/4" "RTN W4A8 AD-full" ad rtn "${AD_RTN_DIR}"
run_full_eval "2/4" "SmoothQuant W4A8 AD-full" ad smoothquant "${AD_SQ_DIR}"
run_full_eval "3/4" "RTN W4A8 Product-full" product rtn "${PRODUCT_RTN_DIR}"
run_full_eval "4/4" "SmoothQuant W4A8 Product-full" product smoothquant "${PRODUCT_SQ_DIR}"

echo "[done] RTN AD:             ${AD_RTN_DIR}/eval_results.json"
echo "[done] SmoothQuant AD:     ${AD_SQ_DIR}/eval_results.json"
echo "[done] RTN Product:        ${PRODUCT_RTN_DIR}/eval_results.json"
echo "[done] SmoothQuant Product:${PRODUCT_SQ_DIR}/eval_results.json"
