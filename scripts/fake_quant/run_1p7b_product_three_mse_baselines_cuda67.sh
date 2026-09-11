#!/usr/bin/env bash
set -euo pipefail

# Serial Product-domain Qwen3-1.7B MSE baselines:
#   1. W8A8 FlatQuant;
#   2. W4A8 OmniQuant-LWC (LET off);
#   3. W4A8 FlatQuant.
# Each arm trains on the first GPU and then runs Product-full evaluation
# sharded across both GPUs.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/onerec_data/benchmark_data}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

GPUS="${GPUS:-6,7}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FLAT_EPOCHS="${FLAT_EPOCHS:-15}"
OMNI_EPOCHS="${OMNI_EPOCHS:-20}"

for pair in "OVERWRITE:${OVERWRITE}" "DRY_RUN:${DRY_RUN}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
done

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} != 2 )) || [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "This launcher requires exactly two distinct GPUs; got GPUS=${GPUS}." >&2
    exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: ${gpu_id}." >&2
        exit 2
    fi
done

W8_FLAT_ROOT="${W8_FLAT_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_product_w8a8_pc_prefix128_final512_heldout512_epoch${FLAT_EPOCHS}}"
W4_OMNI_ROOT="${W4_OMNI_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_product_w4a8_pc_omniquant_lwc_mse_prefix128_final1024_epoch${OMNI_EPOCHS}}"
W4_FLAT_ROOT="${W4_FLAT_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_product_w4a8_pc_prefix128_final512_heldout512_epoch${FLAT_EPOCHS}}"

run_arm() {
    local label="$1"
    shift
    echo
    echo "============================================================"
    echo "${label}"
    echo "============================================================"
    "$@"
}

echo "[protocol] Qwen3-1.7B Product baselines, serial order: W8 FlatQuant -> W4 OmniQuant -> W4 FlatQuant"
echo "[protocol] training_gpu=${GPU_IDS[0]} evaluation_gpus=${GPUS} eval=${EVAL_SAMPLE_SIZE}"
echo "[protocol] FlatQuant: prefix128 + final train512/heldout512, epochs=${FLAT_EPOCHS}"
echo "[protocol] OmniQuant: LWC-only, prefix128 + final1024, epochs=${OMNI_EPOCHS}"

run_arm "[1/3] Product W8A8 FlatQuant" \
    env \
        TASK=product \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        RUN_ROOT="${W8_FLAT_ROOT}" \
        PREFIX_CALIB_SAMPLES=128 \
        FINAL_CALIB_SAMPLES=1024 \
        FINAL_TRAIN_SAMPLES=512 \
        FINAL_HELDOUT_SAMPLES=512 \
        EPOCHS="${FLAT_EPOCHS}" \
        RUN_EVAL=1 \
        EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/flat_quant/run_1p7b_ad_w8a8_pc_official_staged_cuda67.sh

run_arm "[2/3] Product W4A8 OmniQuant-LWC MSE" \
    env \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        RESULTS_ROOT="${W4_OMNI_ROOT}" \
        PREFIX_CALIB_SAMPLES=128 \
        FINAL_CALIB_SAMPLES=1024 \
        EPOCHS="${OMNI_EPOCHS}" \
        EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_1p7b_product_w4a8_pc_omniquant_mse_cuda67.sh

run_arm "[3/3] Product W4A8 FlatQuant" \
    env \
        TASK=product \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        RUN_ROOT="${W4_FLAT_ROOT}" \
        PREFIX_CALIB_SAMPLES=128 \
        FINAL_CALIB_SAMPLES=1024 \
        FINAL_TRAIN_SAMPLES=512 \
        FINAL_HELDOUT_SAMPLES=512 \
        EPOCHS="${FLAT_EPOCHS}" \
        RUN_EVAL=1 \
        EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh

echo
echo "[done] W8A8 FlatQuant: ${W8_FLAT_ROOT}/product_${EVAL_SAMPLE_SIZE}_eval/eval_results.json"
echo "[done] W4A8 OmniQuant: ${W4_OMNI_ROOT}/product_${EVAL_SAMPLE_SIZE}_eval/eval_results.json"
echo "[done] W4A8 FlatQuant: ${W4_FLAT_ROOT}/product_${EVAL_SAMPLE_SIZE}_eval/eval_results.json"
