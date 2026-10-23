#!/usr/bin/env bash
set -euo pipefail

# Product W4A8 FlatQuant + recommendation-task alignment:
# initialize from the completed Product FlatQuant-MSE checkpoint, fine-tune
# the final block on all 1024 calibration records with ABC soft CE + 0.3
# top-32 boundary loss, then run sharded full evaluation on GPUs 6 and 7.

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
TRAIN_GPU="${TRAIN_GPU:-6}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-${OVERWRITE}}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_EVAL_ARTIFACTS="${CLEAN_EVAL_ARTIFACTS:-1}"

for pair in \
    "RUN_EVAL:${RUN_EVAL}" \
    "OVERWRITE:${OVERWRITE}" \
    "EVAL_OVERWRITE:${EVAL_OVERWRITE}" \
    "DRY_RUN:${DRY_RUN}" \
    "CLEAN_EVAL_ARTIFACTS:${CLEAN_EVAL_ARTIFACTS}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
done

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
BASE_ROOT="${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_product_w4a8_pc_prefix128_final512_heldout512_epoch15"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-${BASE_ROOT}/final_train512_heldout512/${MODEL_NAME}/product/flatquant_calibration}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_product_w4a8_pc_from_official_mse_train1024_epoch15}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/exp3_abc_boundary0.3_joint}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/exp3_abc_boundary0.3_joint_product_full_eval}"
CORE_SCRIPT="${REPO_ROOT}/scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh"

cleanup_eval_artifacts() {
    if [[ "${RUN_EVAL}" != "1" || "${CLEAN_EVAL_ARTIFACTS}" != "1" ]]; then
        return
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[cleanup] dry-run: would remove generated samples and shards after successful merge."
        return
    fi
    if [[ ! -s "${EVAL_OUTPUT_DIR}/eval_results.json" ]]; then
        echo "[cleanup] refusing cleanup because eval_results.json is missing." >&2
        return 4
    fi
    find "${EVAL_OUTPUT_DIR}" -type f \
        \( -name 'test_generated.json' -o -name 'test_generated.json.debug' \
           -o -name 'debug_pid.json' -o -name 'debug_sid.json' \) -delete
    if [[ -d "${EVAL_OUTPUT_DIR}.shards" ]]; then
        find "${EVAL_OUTPUT_DIR}.shards" -depth -delete
    fi
    echo "[cleanup] retained eval_results/config/logs/checkpoints; removed generated samples and shards."
}

echo "[protocol] Qwen3-1.7B Product W4A8 per-output-channel FlatQuant + ABC CE + 0.3 boundary"
echo "[protocol] source=${SOURCE_CHECKPOINT_DIR}"
echo "[protocol] final layer train=[0,1024), validation=none, epochs=15, seed=42"
echo "[protocol] jointly learn matrix/diagonal + LWC/LAC"
echo "[protocol] training_gpu=${TRAIN_GPU}; evaluation_gpus=${GPUS}"
echo "[protocol] output=${OUTPUT_DIR}"

env \
    TASK=product \
    GPUS="${GPUS}" \
    TRAIN_GPU="${TRAIN_GPU}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${DATA_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
    QUANT_LABEL=w4a8 \
    WEIGHT_QUANT_FORMAT=fp4_e2m1 \
    ACTIVATION_QUANT_FORMAT=fp8_e4m3fn \
    CALIB_SAMPLES=1024 \
    TRAIN_SAMPLES=1024 \
    EPOCHS=15 \
    BOUNDARY_WEIGHT=0.3 \
    LFQ_LOSS_WEIGHT=1.0 \
    LEARN_TRANSFORM=1 \
    LEARN_LAC=1 \
    BOUNDARY_TOPK=32 \
    BOUNDARY_NEGATIVES=32 \
    BOUNDARY_TIE_THRESHOLD=0.01 \
    BOUNDARY_GAP_SCALE=1.0 \
    RUN_EVAL="${RUN_EVAL}" \
    EVAL_SAMPLE_SIZE=full \
    OVERWRITE="${OVERWRITE}" \
    EVAL_OVERWRITE="${EVAL_OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${CORE_SCRIPT}"

cleanup_eval_artifacts

echo "[done] checkpoints=${OUTPUT_DIR}/${MODEL_NAME}/product/flatquant_calibration"
if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "[done] evaluation=${EVAL_OUTPUT_DIR}/eval_results.json"
fi
