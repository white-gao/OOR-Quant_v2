#!/usr/bin/env bash
set -euo pipefail

# Cross-domain reproduction of the best AD W8A8 FlatQuant alignment arm:
# start from the official task-specific MSE checkpoint, then fine-tune only
# the final block on all 1024 calibration records for 15 epochs with
# ABC soft CE + 0.3 top-32 boundary loss. Matrix/diagonal, LWC, and LAC
# parameters are jointly optimized.

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
RUN_TASKS="${RUN_TASKS:-product video}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-${OVERWRITE}}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"

for pair in "RUN_EVAL:${RUN_EVAL}" "OVERWRITE:${OVERWRITE}" "EVAL_OVERWRITE:${EVAL_OVERWRITE}" "DRY_RUN:${DRY_RUN}"; do
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
TRAIN_GPU="${GPU_IDS[0]}"

if [[ -z "${RUN_TASKS//[[:space:]]/}" ]]; then
    echo "RUN_TASKS must select product and/or video." >&2
    exit 2
fi
for task in ${RUN_TASKS}; do
    case "${task}" in
        product|video) ;;
        *) echo "RUN_TASKS supports only product and video; got ${task}." >&2; exit 2 ;;
    esac
done

CORE_SCRIPT="${REPO_ROOT}/scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh"

echo "[protocol] Qwen3-1.7B W8A8 FlatQuant + ABC CE + 0.3 boundary"
echo "[protocol] tasks=${RUN_TASKS}; serial; training_gpu=${TRAIN_GPU}; evaluation_gpus=${GPUS}"
echo "[protocol] source=official task-specific FlatQuant MSE prefix128/final512-heldout512"
echo "[protocol] final layer train=[0,1024), validation=none, epochs=15, seed=42"
echo "[protocol] jointly learn matrix/diagonal + LWC/LAC; best_epoch=off"

for task in ${RUN_TASKS}; do
    base_root="${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_${task}_w8a8_pc_prefix128_final512_heldout512_epoch15"
    source_checkpoint_dir="${base_root}/final_train512_heldout512/${MODEL_NAME}/${task}/flatquant_calibration"
    run_root="${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_${task}_w8a8_pc_from_official_mse_train1024_epoch15"
    output_dir="${run_root}/exp3_abc_boundary0.3_joint"
    eval_output_dir="${run_root}/exp3_abc_boundary0.3_joint_${task}_full_eval"

    echo
    echo "==================== [${task}] ===================="
    echo "[${task}] source=${source_checkpoint_dir}"
    echo "[${task}] output=${output_dir}"

    env \
        TASK="${task}" \
        GPUS="${GPUS}" \
        TRAIN_GPU="${TRAIN_GPU}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        SOURCE_CHECKPOINT_DIR="${source_checkpoint_dir}" \
        OUTPUT_DIR="${output_dir}" \
        EVAL_OUTPUT_DIR="${eval_output_dir}" \
        QUANT_LABEL=w8a8 \
        WEIGHT_QUANT_FORMAT=fp8_e4m3fn \
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
done

echo
for task in ${RUN_TASKS}; do
    result="${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_${task}_w8a8_pc_from_official_mse_train1024_epoch15/exp3_abc_boundary0.3_joint_${task}_full_eval/eval_results.json"
    echo "[done] ${task}: ${result}"
done
