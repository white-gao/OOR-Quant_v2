#!/usr/bin/env bash
set -euo pipefail

# Qwen3-1.7B task-domain OmniQuant-LWC MSE baseline (Product W4A8 by default).
#
# Train layers 0-26 on 128 calibration records, optimize layer 27 with
# hidden-state MSE on all 1024 records, then evaluate on the full task.
# Weight QDQ is per-output-channel FP4-E2M1 and activation QDQ is dynamic
# per-token FP8-E4M3FN. LET is disabled.
#
# Usage: bash scripts/fake_quant/run_1p7b_product_w4a8_pc_omniquant_mse_cuda67.sh
# Set DRY_RUN=1 to inspect commands or OVERWRITE=1 to replace prior outputs.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
GPUS="${GPUS:-6,7}"
TASK="${TASK:-product}"
WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-fp4_e2m1}"
ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT:-fp8_e4m3fn}"
QUANT_LABEL="${QUANT_LABEL:-w4a8}"
PREFIX_CALIB_SAMPLES="${PREFIX_CALIB_SAMPLES:-128}"
FINAL_CALIB_SAMPLES="${FINAL_CALIB_SAMPLES:-1024}"
NUM_LAYERS="${NUM_LAYERS:-28}"
EPOCHS="${EPOCHS:-20}"
LWC_LR="${LWC_LR:-1e-2}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_${TASK}_${QUANT_LABEL}_pc_omniquant_lwc_mse_prefix${PREFIX_CALIB_SAMPLES}_final${FINAL_CALIB_SAMPLES}_epoch${EPOCHS}}"
PREFIX_OUTPUT_DIR="${PREFIX_OUTPUT_DIR:-${RESULTS_ROOT}/prefix_calib${PREFIX_CALIB_SAMPLES}}"
PREFIX_CHECKPOINT_DIR="${PREFIX_CHECKPOINT_DIR:-${PREFIX_OUTPUT_DIR}/1.7B/${TASK}/omniquant_calibration}"
MSE_CALIB_DIR="${MSE_CALIB_DIR:-${RESULTS_ROOT}/mse_lwc_final${FINAL_CALIB_SAMPLES}_calibration}"
MSE_EVAL_DIR="${MSE_EVAL_DIR:-${RESULTS_ROOT}/${TASK}_${EVAL_SAMPLE_SIZE}_eval}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
MSE_CHECKPOINT_DIR="${MSE_CALIB_DIR}/${MODEL_NAME}/${TASK}/omniquant_calibration"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}." >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1; got ${DRY_RUN}." >&2
    exit 2
fi
case "${TASK}" in
    ad|product|video|label_pred) ;;
    *) echo "Unsupported TASK=${TASK}." >&2; exit 2 ;;
esac
if (( PREFIX_CALIB_SAMPLES <= 0 || FINAL_CALIB_SAMPLES <= 0 )); then
    echo "Calibration sample counts must be positive." >&2
    exit 2
fi
if (( PREFIX_CALIB_SAMPLES > FINAL_CALIB_SAMPLES )); then
    echo "PREFIX_CALIB_SAMPLES cannot exceed FINAL_CALIB_SAMPLES." >&2
    exit 2
fi
if (( NUM_LAYERS < 2 )); then
    echo "NUM_LAYERS must be at least 2." >&2
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
CALIB_GPU="${GPU_IDS[0]}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_range_complete() {
    local checkpoint_dir="$1"
    local first_layer="$2"
    local last_layer="$3"
    local layer_idx checkpoint_name
    for ((layer_idx = first_layer; layer_idx <= last_layer; layer_idx++)); do
        printf -v checkpoint_name 'layer_%02d.pt' "${layer_idx}"
        if [[ ! -s "${checkpoint_dir}/${checkpoint_name}" ]]; then
            return 1
        fi
    done
}

checkpoint_dir_has_files() {
    compgen -G "$1/layer_*.pt" >/dev/null
}

print_command() {
    printf '  %q' "$@"
    printf '\n'
}

if checkpoint_range_complete "${PREFIX_CHECKPOINT_DIR}" 0 "${PREFIX_LAST_LAYER}" && [[ "${OVERWRITE}" != "1" ]]; then
    echo "[1/3 prefix calibration] complete checkpoints found; skipping."
else
    if checkpoint_dir_has_files "${PREFIX_CHECKPOINT_DIR}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[1/3 prefix calibration] partial checkpoints found at ${PREFIX_CHECKPOINT_DIR}." >&2
        echo "Set OVERWRITE=1 or move the partial output aside before retrying." >&2
        exit 4
    fi

    PREFIX_COMMAND=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        --task "${TASK}"
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --device cuda:0
        --mode omniquant
        --output_dir "${PREFIX_OUTPUT_DIR}"
        --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
        --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
        --weight_quant_scheme symmetric
        --weight_group_size 0
        --omni_lwc
        --omni_symmetric_lwc_mode absmax
        --omni_let_mode none
        --omni_epochs "${EPOCHS}"
        --omni_epoch_eval_interval 0
        --omni_lwc_lr "${LWC_LR}"
        --omni_init_lwc_logit "${INIT_LWC_LOGIT}"
        --calibration_only
        --layers "0-${PREFIX_LAST_LAYER}"
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}"
        --omni_final_objective mse
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        PREFIX_COMMAND+=(--overwrite)
    fi

    echo "[1/3 prefix calibration] physical_gpu=${CALIB_GPU} output=${PREFIX_OUTPUT_DIR}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${CALIB_GPU}" "${PREFIX_COMMAND[@]}"
    else
        mkdir -p "${PREFIX_OUTPUT_DIR}"
        env CUDA_VISIBLE_DEVICES="${CALIB_GPU}" "${PREFIX_COMMAND[@]}" 2>&1 | tee "${PREFIX_OUTPUT_DIR}/train.log"
        if ! checkpoint_range_complete "${PREFIX_CHECKPOINT_DIR}" 0 "${PREFIX_LAST_LAYER}"; then
            echo "[1/3 prefix calibration] expected checkpoints 0-${PREFIX_LAST_LAYER} were not produced." >&2
            exit 5
        fi
    fi
fi

echo "[protocol] model=${MODEL_PATH} task=${TASK} seed=42"
echo "[protocol] weight=${WEIGHT_QUANT_FORMAT} activation=${ACTIVATION_QUANT_FORMAT}, symmetric OmniQuant-LWC, LET disabled"
echo "[protocol] prefix layers=0-${PREFIX_LAST_LAYER} calib=[0,${PREFIX_CALIB_SAMPLES}) source=${PREFIX_CHECKPOINT_DIR}"
echo "[protocol] final layer=${FINAL_LAYER} objective=MSE calib=[0,${FINAL_CALIB_SAMPLES}) epochs=${EPOCHS}"
echo "[protocol] calibration_gpu=${CALIB_GPU}; full evaluation_gpus=${GPUS}"
echo "[protocol] results_root=${RESULTS_ROOT}"

if checkpoint_range_complete "${MSE_CHECKPOINT_DIR}" 0 "${FINAL_LAYER}" && [[ "${OVERWRITE}" != "1" ]]; then
    echo "[2/3 final MSE calibration] complete checkpoints found; skipping."
else
    if checkpoint_dir_has_files "${MSE_CHECKPOINT_DIR}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[2/3 final MSE calibration] partial checkpoints found at ${MSE_CHECKPOINT_DIR}." >&2
        echo "Set OVERWRITE=1 or move the partial output aside before retrying." >&2
        exit 4
    fi

    TRAIN_COMMAND=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        --task "${TASK}"
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --device cuda:0
        --mode omniquant
        --output_dir "${MSE_CALIB_DIR}"
        --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
        --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
        --weight_quant_scheme symmetric
        --weight_group_size 0
        --omni_lwc
        --omni_symmetric_lwc_mode absmax
        --omni_let_mode none
        --omni_epochs "${EPOCHS}"
        --omni_epoch_eval_interval 0
        --omni_lwc_lr "${LWC_LR}"
        --omni_init_lwc_logit "${INIT_LWC_LOGIT}"
        --calibration_only
        --layers all
        --calib_sample_size "${FINAL_CALIB_SAMPLES}"
        --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}"
        --omni_final_objective mse
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        TRAIN_COMMAND+=(--overwrite)
    fi

    echo "[2/3 final MSE calibration] physical_gpu=${CALIB_GPU} output=${MSE_CALIB_DIR}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${CALIB_GPU}" "${TRAIN_COMMAND[@]}"
    else
        mkdir -p "${MSE_CALIB_DIR}"
        env CUDA_VISIBLE_DEVICES="${CALIB_GPU}" "${TRAIN_COMMAND[@]}" 2>&1 | tee "${MSE_CALIB_DIR}/train.log"
        if ! checkpoint_range_complete "${MSE_CHECKPOINT_DIR}" 0 "${FINAL_LAYER}"; then
            echo "[2/3 final MSE calibration] expected checkpoints for layers 0-${FINAL_LAYER} were not produced." >&2
            exit 5
        fi
    fi
fi

if [[ -s "${MSE_EVAL_DIR}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
    echo "[3/3 ${TASK} evaluation] complete eval_results.json found; skipping."
else
    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${MSE_CHECKPOINT_DIR}" 0 "${FINAL_LAYER}"; then
        echo "[3/3 ${TASK} evaluation] checkpoint set is incomplete: ${MSE_CHECKPOINT_DIR}" >&2
        exit 5
    fi

    echo "[3/3 ${TASK} evaluation] physical_gpus=${GPUS} output=${MSE_EVAL_DIR}"
    TASK="${TASK}" \
    GPUS="${GPUS}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${MSE_EVAL_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OVERWRITE="${OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh \
        --mode omniquant \
        --weight_quant_format "${WEIGHT_QUANT_FORMAT}" \
        --activation_quant_format "${ACTIVATION_QUANT_FORMAT}" \
        --weight_quant_scheme symmetric \
        --weight_group_size 0 \
        --omni_lwc \
        --omni_symmetric_lwc_mode absmax \
        --omni_let_mode none \
        --omni_final_objective mse \
        --omni_load_checkpoint_dir "${MSE_CHECKPOINT_DIR}" \
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}"
fi

echo "[done] MSE calibration: ${MSE_CHECKPOINT_DIR}"
echo "[done] MSE ${TASK}-${EVAL_SAMPLE_SIZE}: ${MSE_EVAL_DIR}/eval_results.json"
