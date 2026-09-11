#!/usr/bin/env bash
set -euo pipefail

# Train one FlatQuant final-layer LFQ arm from a complete MSE checkpoint, then
# optionally run a sharded full-task evaluation. This is the shared protocol
# used by boundary, LAC, and cross-task wrappers.

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

TASK="${TASK:-ad}"
QUANT_LABEL="${QUANT_LABEL:-w4a8}"
WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-fp4_e2m1}"
ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT:-fp8_e4m3fn}"
GPUS="${GPUS:-6,7}"
NUM_LAYERS="${NUM_LAYERS:-28}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1024}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1024}"
EPOCHS="${EPOCHS:-15}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.3}"
LFQ_LOSS_WEIGHT="${LFQ_LOSS_WEIGHT:-1.0}"
LEARN_TRANSFORM="${LEARN_TRANSFORM:-0}"
LEARN_LAC="${LEARN_LAC:-1}"
TRANSFORM_LR="${TRANSFORM_LR:-5e-3}"
LWC_LR="${LWC_LR:-5e-2}"
LAC_LR="${LAC_LR:-5e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DIAG_ALPHA="${DIAG_ALPHA:-0.5}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
INIT_LAC_LOGIT="${INIT_LAC_LOGIT:-4.0}"
BOUNDARY_TOPK="${BOUNDARY_TOPK:-32}"
BOUNDARY_NEGATIVES="${BOUNDARY_NEGATIVES:-32}"
BOUNDARY_TIE_THRESHOLD="${BOUNDARY_TIE_THRESHOLD:-0.01}"
BOUNDARY_GAP_SCALE="${BOUNDARY_GAP_SCALE:-1.0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-${OVERWRITE}}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_AD_SOURCE="${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_ad_${QUANT_LABEL}_pc_from_mse_train1024_epoch15/mse1024_frozen_transform_control/1.7B/ad/flatquant_calibration"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-}"
if [[ -z "${SOURCE_CHECKPOINT_DIR}" && "${TASK}" == "ad" ]]; then
    SOURCE_CHECKPOINT_DIR="${DEFAULT_AD_SOURCE}"
fi
if [[ -z "${SOURCE_CHECKPOINT_DIR}" ]]; then
    echo "SOURCE_CHECKPOINT_DIR is required for TASK=${TASK}." >&2
    exit 2
fi

boundary_tag="${BOUNDARY_WEIGHT//./p}"
DEFAULT_RUN_ROOT="${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_${TASK}_${QUANT_LABEL}_pc_from_mse1024_train${TRAIN_SAMPLES}_epoch${EPOCHS}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_RUN_ROOT}/abc_boundary_w${boundary_tag}_transform${LEARN_TRANSFORM}_lac${LEARN_LAC}}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/${MODEL_NAME}/${TASK}/flatquant_calibration}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}_${TASK}_${EVAL_SAMPLE_SIZE}_eval}"
FINAL_LAYER=$((NUM_LAYERS - 1))

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}
for pair in "LEARN_TRANSFORM:${LEARN_TRANSFORM}" "LEARN_LAC:${LEARN_LAC}" "RUN_EVAL:${RUN_EVAL}" "OVERWRITE:${OVERWRITE}" "EVAL_OVERWRITE:${EVAL_OVERWRITE}" "DRY_RUN:${DRY_RUN}"; do
    validate_boolean "${pair%%:*}" "${pair#*:}"
done
case "${TASK}" in
    ad|product|video|label_pred) ;;
    *) echo "Unsupported TASK=${TASK}." >&2; exit 2 ;;
esac
if (( NUM_LAYERS < 1 || CALIB_SAMPLES != 1024 || TRAIN_SAMPLES != 1024 )); then
    echo "This protocol requires NUM_LAYERS>=1 and CALIB_SAMPLES=TRAIN_SAMPLES=1024." >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import math,sys; values=map(float,sys.argv[1:]); sys.exit(not all(math.isfinite(x) and x >= 0 for x in values))' "${BOUNDARY_WEIGHT}" "${LFQ_LOSS_WEIGHT}"; then
    echo "BOUNDARY_WEIGHT and LFQ_LOSS_WEIGHT must be finite and non-negative." >&2
    exit 2
fi
if "${PYTHON_BIN}" -c 'import sys; sys.exit(not (float(sys.argv[1]) == 0 and float(sys.argv[2]) == 0))' "${BOUNDARY_WEIGHT}" "${LFQ_LOSS_WEIGHT}"; then
    echo "At least one LFQ objective weight must be positive." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} < 1 )); then
    echo "GPUS must contain at least one GPU ID." >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ || -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "Invalid or duplicate GPU ID in GPUS=${GPUS}." >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
done
TRAIN_GPU="${TRAIN_GPU:-${GPU_IDS[0]}}"
if [[ -z "${SEEN_GPUS[${TRAIN_GPU}]:-}" ]]; then
    echo "TRAIN_GPU=${TRAIN_GPU} is not listed in GPUS=${GPUS}." >&2
    exit 2
fi
if [[ "${RUN_EVAL}" == "1" && ${#GPU_IDS[@]} -lt 2 ]]; then
    echo "RUN_EVAL=1 requires at least two GPUs for sharded evaluation." >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

print_command() {
    printf "  %q" "$@"
    printf "\n"
}

checkpoint_complete() {
    local checkpoint_dir="$1"
    local layer_idx checkpoint_name
    for ((layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++)); do
        printf -v checkpoint_name "layer_%02d.pt" "${layer_idx}"
        if [[ ! -s "${checkpoint_dir}/${checkpoint_name}" ]]; then
            return 1
        fi
    done
}

checkpoint_dir_has_files() {
    compgen -G "$1/layer_*.pt" >/dev/null
}

if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_complete "${SOURCE_CHECKPOINT_DIR}"; then
    echo "Incomplete source MSE checkpoint: ${SOURCE_CHECKPOINT_DIR}" >&2
    exit 3
fi

TRANSFORM_FLAG="--no-flat_learn_transform"
if [[ "${LEARN_TRANSFORM}" == "1" ]]; then
    TRANSFORM_FLAG="--flat_learn_transform"
fi
LAC_FLAG="--no-flat_learn_lac"
if [[ "${LEARN_LAC}" == "1" ]]; then
    LAC_FLAG="--flat_learn_lac"
fi

COMMON_MODEL_ARGS=(
    --mode flatquant_core
    --layers all
    --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
    --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --flat_lac
    "${TRANSFORM_FLAG}"
    "${LAC_FLAG}"
    --flat_transform_kind kronecker
    --flat_transform_init random_orthogonal
    --flat_train_sample_size "${TRAIN_SAMPLES}"
    --flat_validation_sample_size 0
    --flat_epochs "${EPOCHS}"
    --flat_epoch_eval_interval 0
    --flat_transform_lr "${TRANSFORM_LR}"
    --flat_lwc_lr "${LWC_LR}"
    --flat_lac_lr "${LAC_LR}"
    --flat_weight_decay "${WEIGHT_DECAY}"
    --flat_diag_alpha "${DIAG_ALPHA}"
    --flat_init_lwc_logit "${INIT_LWC_LOGIT}"
    --flat_init_lac_logit "${INIT_LAC_LOGIT}"
    --flat_normalize_mse_gradient
    --omni_final_objective lfq_ce
    --omni_lfq_token_scope sid_slots
    --omni_lfq_vocab_scope s_abc
    --omni_lfq_slot_weights 1 1 1
    --omni_lfq_loss_weight "${LFQ_LOSS_WEIGHT}"
    --omni_lfq_boundary_loss_weight "${BOUNDARY_WEIGHT}"
    --omni_lfq_boundary_topk "${BOUNDARY_TOPK}"
    --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}"
    --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}"
    --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}"
)

# Loading a trained LFQ checkpoint is an inference-only operation. Keep the
# quantization/runtime configuration identical to training, but do not pass the
# LFQ objective or optimizer settings: those options request another training
# objective and are intentionally rejected together with --flat_load_checkpoint_dir.
EVAL_MODEL_ARGS=(
    --mode flatquant_core
    --layers all
    --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
    --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --flat_lac
    "${TRANSFORM_FLAG}"
    "${LAC_FLAG}"
    --flat_transform_kind kronecker
    --flat_transform_init random_orthogonal
    --flat_train_sample_size "${TRAIN_SAMPLES}"
    --flat_validation_sample_size 0
    --flat_diag_alpha "${DIAG_ALPHA}"
    --flat_init_lwc_logit "${INIT_LWC_LOGIT}"
    --flat_init_lac_logit "${INIT_LAC_LOGIT}"
)

run_training() {
    if checkpoint_complete "${CHECKPOINT_DIR}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[training] complete checkpoints found; skipping."
        return
    fi
    if checkpoint_dir_has_files "${CHECKPOINT_DIR}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[training] partial checkpoints found at ${CHECKPOINT_DIR}." >&2
        echo "Move them aside or set OVERWRITE=1." >&2
        return 3
    fi
    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        --task "${TASK}"
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --device cuda:0
        --calib_sample_size "${CALIB_SAMPLES}"
        --flat_finetune_checkpoint_dir "${SOURCE_CHECKPOINT_DIR}"
        --calibration_only
        --output_dir "${OUTPUT_DIR}"
        "${COMMON_MODEL_ARGS[@]}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi
    echo "[training] task=${TASK} physical_gpu=${TRAIN_GPU} output=${OUTPUT_DIR}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${TRAIN_GPU}" "${command[@]}"
        return
    fi
    mkdir -p "${OUTPUT_DIR}"
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
    if ! checkpoint_complete "${CHECKPOINT_DIR}"; then
        echo "[training] expected checkpoints 0-${FINAL_LAYER} were not produced." >&2
        return 4
    fi
}

run_evaluation() {
    if [[ "${RUN_EVAL}" != "1" ]]; then
        echo "[evaluation] skipped because RUN_EVAL=${RUN_EVAL}."
        return
    fi
    if [[ -s "${EVAL_OUTPUT_DIR}/eval_results.json" && "${EVAL_OVERWRITE}" != "1" ]]; then
        echo "[evaluation] complete result found; skipping."
        return
    fi
    if [[ "${EVAL_OVERWRITE}" != "1" && ( -d "${EVAL_OUTPUT_DIR}" || -d "${EVAL_OUTPUT_DIR}.shards" ) ]]; then
        echo "[evaluation] partial output found at ${EVAL_OUTPUT_DIR}." >&2
        echo "Move it aside or set EVAL_OVERWRITE=1." >&2
        return 3
    fi
    echo "[evaluation] task=${TASK} physical_gpus=${GPUS} output=${EVAL_OUTPUT_DIR}"
    env \
        TASK="${TASK}" \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        OVERWRITE="${EVAL_OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/flat_quant/run_sharded_eval_cuda.sh \
            "${EVAL_MODEL_ARGS[@]}" \
            --flat_load_checkpoint_dir "${CHECKPOINT_DIR}" \
            --calib_sample_size 128 \
            --eval_sample_size "${EVAL_SAMPLE_SIZE}"
}

echo "[protocol] Qwen3-1.7B ${TASK} ${QUANT_LABEL} per-output-channel FlatQuant final alignment"
echo "[protocol] source=${SOURCE_CHECKPOINT_DIR}"
echo "[protocol] train=[0,${TRAIN_SAMPLES}) epochs=${EPOCHS} seed=42"
echo "[protocol] learn_transform=${LEARN_TRANSFORM} learn_lac=${LEARN_LAC} train_lwc=1"
echo "[protocol] lfq_weight=${LFQ_LOSS_WEIGHT} boundary_weight=${BOUNDARY_WEIGHT}"
echo "[protocol] output=${OUTPUT_DIR}"

run_training
run_evaluation

echo "[done] checkpoints=${CHECKPOINT_DIR}"
if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "[done] evaluation=${EVAL_OUTPUT_DIR}/eval_results.json"
fi
