#!/usr/bin/env bash
set -euo pipefail

# Serial AD-full comparison using the same deployment-matched fake-QDQ path:
#   1. RTN FP4-W/FP8-A, per-output-channel weight scaling.
#   2. RTN FP4-W/FP8-A, per-output-channel/input-group weight scaling (g128).
#
# Each experiment is evaluated with one shard per GPU. The second experiment
# starts only after every shard of the first experiment has completed and its
# metrics have been merged.
#
# Usage after activating the Python environment:
#   bash scripts/fake_quant/run_1p7b_ad_full_fp4w_fp8a_rtn_pc_g128_cuda0123.sh
#
# Optional:
#   OVERWRITE=1 bash scripts/fake_quant/run_1p7b_ad_full_fp4w_fp8a_rtn_pc_g128_cuda0123.sh
#   DRY_RUN=1 bash scripts/fake_quant/run_1p7b_ad_full_fp4w_fp8a_rtn_pc_g128_cuda0123.sh

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
GPUS="${GPUS:-0,1,2,3}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_full_fp4w_fp8a_deployment_matched}"

PER_CHANNEL_DIR="${PER_CHANNEL_DIR:-${RESULTS_ROOT}/rtn_fp4w_fp8a_per_channel_ad_full}"
G128_DIR="${G128_DIR:-${RESULTS_ROOT}/rtn_fp4w_fp8a_g128_ad_full}"

run_eval() {
    local label="$1"
    local output_dir="$2"
    local group_size="$3"

    echo "[${label}] task=ad evaluation_gpus=${GPUS} output=${output_dir}"
    TASK=ad \
    GPUS="${GPUS}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${output_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OVERWRITE="${OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh \
        --mode baseline_qdq \
        --weight_quant_format fp4_e2m1 \
        --activation_quant_format fp8_e4m3fn \
        --weight_group_size "${group_size}"
}

echo "Stage 1/2: RTN FP4-W/FP8-A per-channel, AD full."
run_eval "RTN FP4-W/FP8-A per-channel" "${PER_CHANNEL_DIR}" 0

echo "Stage 2/2: RTN FP4-W/FP8-A g128, AD full."
run_eval "RTN FP4-W/FP8-A g128" "${G128_DIR}" 128

echo "Completed both serial W4A8 evaluations."
echo "per-channel: ${PER_CHANNEL_DIR}/eval_results.json"
echo "g128:       ${G128_DIR}/eval_results.json"
