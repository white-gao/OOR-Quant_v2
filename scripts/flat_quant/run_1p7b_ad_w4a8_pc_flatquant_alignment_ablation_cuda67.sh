#!/usr/bin/env bash
set -euo pipefail

# Result-oriented FlatQuant per-channel comparison (W4A8 by default), executed in priority
# order 3 -> 2 -> 1:
#   3) ABC CE + 0.3 boundary, jointly optimize matrix/diagonal + LWC/LAC;
#   2) ABC CE + 0.3 boundary, freeze matrix/diagonal and optimize LWC/LAC;
#   1) reuse the completed formal FlatQuant-MSE AD-full baseline.
#
# Arms 3 and 2 both inherit layers 0-26 and the layer-27 initialization from
# exactly the same complete MSE checkpoint. All 1024 AD calibration records
# participate in their final-block backpropagation. The calibration parquet has
# exactly 1024 records, so this protocol deliberately has no calibration
# held-out split; the full AD benchmark is the downstream comparison.

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
NUM_LAYERS="${NUM_LAYERS:-28}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1024}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1024}"
EPOCHS="${EPOCHS:-15}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.3}"
TOPK="${TOPK:-32}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-32}"
TIE_THRESHOLD="${TIE_THRESHOLD:-0.01}"
GAP_SCALE="${GAP_SCALE:-1.0}"
TRANSFORM_LR="${TRANSFORM_LR:-5e-3}"
LWC_LR="${LWC_LR:-5e-2}"
LAC_LR="${LAC_LR:-5e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DIAG_ALPHA="${DIAG_ALPHA:-0.5}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
INIT_LAC_LOGIT="${INIT_LAC_LOGIT:-4.0}"
QUANT_LABEL="${QUANT_LABEL:-w4a8}"
WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-fp4_e2m1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_ad_${QUANT_LABEL}_pc_prefix128_final512_heldout512_epoch15}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-${BASE_RUN_ROOT}/final_train512_heldout512/1.7B/ad/flatquant_calibration}"
BASELINE_EVAL_RESULT="${BASELINE_EVAL_RESULT:-${BASE_RUN_ROOT}/ad_full_eval/eval_results.json}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment/1p7b_ad_${QUANT_LABEL}_pc_from_mse_train${TRAIN_SAMPLES}_epoch${EPOCHS}}"

JOINT_OUTPUT_DIR="${RUN_ROOT}/exp3_abc_boundary${BOUNDARY_WEIGHT}_joint"
FROZEN_OUTPUT_DIR="${RUN_ROOT}/exp2_abc_boundary${BOUNDARY_WEIGHT}_frozen_transform"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
JOINT_CHECKPOINT_DIR="${JOINT_OUTPUT_DIR}/${MODEL_NAME}/ad/flatquant_calibration"
FROZEN_CHECKPOINT_DIR="${FROZEN_OUTPUT_DIR}/${MODEL_NAME}/ad/flatquant_calibration"
JOINT_EVAL_DIR="${RUN_ROOT}/exp3_abc_boundary${BOUNDARY_WEIGHT}_joint_ad_full_eval"
FROZEN_EVAL_DIR="${RUN_ROOT}/exp2_abc_boundary${BOUNDARY_WEIGHT}_frozen_transform_ad_full_eval"
FINAL_LAYER=$((NUM_LAYERS - 1))

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}

validate_boolean RUN_EVAL "${RUN_EVAL}"
validate_boolean OVERWRITE "${OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"
if (( NUM_LAYERS < 1 )); then
    echo "NUM_LAYERS must be positive." >&2
    exit 2
fi
if (( CALIB_SAMPLES != 1024 || TRAIN_SAMPLES != 1024 )); then
    echo "This protocol requires CALIB_SAMPLES=TRAIN_SAMPLES=1024." >&2
    echo "Got calibration=${CALIB_SAMPLES}, train=${TRAIN_SAMPLES}." >&2
    exit 2
fi
if [[ "${EVAL_SAMPLE_SIZE}" != "full" ]]; then
    echo "EVAL_SAMPLE_SIZE must be full so arms 3/2 match the reused arm-1 result." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS="," read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} != 2 )); then
    echo "This launcher requires exactly two GPUs; got GPUS=${GPUS}." >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "Duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
done
TRAIN_GPU="${GPU_IDS[0]}"

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

if ! checkpoint_complete "${BASE_CHECKPOINT_DIR}"; then
    echo "The shared MSE initialization checkpoint is missing or incomplete:" >&2
    echo "  ${BASE_CHECKPOINT_DIR}" >&2
    exit 3
fi
if [[ ! -s "${BASELINE_EVAL_RESULT}" ]]; then
    echo "The reused experiment-1 AD-full result is missing:" >&2
    echo "  ${BASELINE_EVAL_RESULT}" >&2
    exit 3
fi

COMMON_TRAIN_ARGS=(
    --task ad
    --mode flatquant_core
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --layers all
    --calib_sample_size "${CALIB_SAMPLES}"
    --flat_train_sample_size "${TRAIN_SAMPLES}"
    --flat_validation_sample_size 0
    --flat_finetune_checkpoint_dir "${BASE_CHECKPOINT_DIR}"
    --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --flat_lac
    --flat_transform_kind kronecker
    --flat_transform_init random_orthogonal
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
    --omni_lfq_loss_weight 1
    --omni_lfq_boundary_loss_weight "${BOUNDARY_WEIGHT}"
    --omni_lfq_boundary_topk "${TOPK}"
    --omni_lfq_boundary_negative_count "${NEGATIVE_COUNT}"
    --omni_lfq_boundary_tie_threshold "${TIE_THRESHOLD}"
    --omni_lfq_boundary_gap_scale "${GAP_SCALE}"
    --calibration_only
)

run_training_arm() {
    local label="$1"
    local output_dir="$2"
    local checkpoint_dir="$3"
    local transform_flag="$4"

    if checkpoint_complete "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping training."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        echo "Move the partial output aside or rerun with OVERWRITE=1." >&2
        return 3
    fi

    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        "${COMMON_TRAIN_ARGS[@]}"
        --output_dir "${output_dir}"
        "${transform_flag}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi
    echo "[${label}] physical_gpu=${TRAIN_GPU} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${TRAIN_GPU}" "${command[@]}"
        return
    fi
    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    if ! checkpoint_complete "${checkpoint_dir}"; then
        echo "[${label}] expected 0-${FINAL_LAYER} checkpoints were not produced." >&2
        return 4
    fi
}

run_full_eval() {
    local label="$1"
    local checkpoint_dir="$2"
    local output_dir="$3"

    if [[ -s "${output_dir}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete AD-full result found; skipping evaluation."
        return
    fi
    if [[ "${OVERWRITE}" != "1" && ( -d "${output_dir}" || -d "${output_dir}.shards" ) ]]; then
        echo "[${label}] partial evaluation output found at ${output_dir}." >&2
        echo "Move it aside or rerun with OVERWRITE=1." >&2
        return 3
    fi

    echo "[${label}] physical_gpus=${GPUS} output=${output_dir}"
    env \
        TASK=ad \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        OUTPUT_DIR="${output_dir}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/flat_quant/run_sharded_eval_cuda.sh \
            --mode flatquant_core \
            --layers all \
            --weight_quant_format "${WEIGHT_QUANT_FORMAT}" \
            --activation_quant_format fp8_e4m3fn \
            --weight_quant_scheme symmetric \
            --weight_group_size 0 \
            --omni_lwc \
            --flat_lac \
            --flat_learn_transform \
            --flat_transform_kind kronecker \
            --flat_transform_init random_orthogonal \
            --flat_train_sample_size "${TRAIN_SAMPLES}" \
            --flat_validation_sample_size 0 \
            --flat_diag_alpha "${DIAG_ALPHA}" \
            --flat_load_checkpoint_dir "${checkpoint_dir}" \
            --calib_sample_size 128 \
            --eval_sample_size full
}

echo "[protocol] Qwen3-1.7B AD FlatQuant ${QUANT_LABEL} per-output-channel"
echo "[protocol] weight_format=${WEIGHT_QUANT_FORMAT}, activation_format=fp8_e4m3fn"
echo "[protocol] priority order=3 joint -> 2 frozen -> 1 reused MSE baseline"
echo "[protocol] arms 3/2 train=[0,1024), validation=none, epochs=${EPOCHS}, seed=42"
echo "[protocol] shared initialization=${BASE_CHECKPOINT_DIR}"
echo "[protocol] experiment-1 reused result=${BASELINE_EVAL_RESULT}"
echo "[protocol] run_root=${RUN_ROOT}"

echo
echo "[3/3] Joint matrix/diagonal + LWC/LAC, ABC CE + ${BOUNDARY_WEIGHT} boundary."
run_training_arm \
    "experiment 3 training" \
    "${JOINT_OUTPUT_DIR}" \
    "${JOINT_CHECKPOINT_DIR}" \
    --flat_learn_transform
if [[ "${RUN_EVAL}" == "1" ]]; then
    run_full_eval "experiment 3 AD-full" "${JOINT_CHECKPOINT_DIR}" "${JOINT_EVAL_DIR}"
fi

echo
echo "[2/3] Frozen matrix/diagonal, train LWC/LAC with the same objective."
run_training_arm \
    "experiment 2 training" \
    "${FROZEN_OUTPUT_DIR}" \
    "${FROZEN_CHECKPOINT_DIR}" \
    --no-flat_learn_transform
if [[ "${RUN_EVAL}" == "1" ]]; then
    run_full_eval "experiment 2 AD-full" "${FROZEN_CHECKPOINT_DIR}" "${FROZEN_EVAL_DIR}"
fi

echo
echo "[1/3] Reuse completed formal FlatQuant-MSE baseline; no continuation or evaluation job."
echo "[experiment 1] result=${BASELINE_EVAL_RESULT}"

echo "[done] experiment-3 checkpoints=${JOINT_CHECKPOINT_DIR}"
echo "[done] experiment-2 checkpoints=${FROZEN_CHECKPOINT_DIR}"
if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "[done] experiment-3 evaluation=${JOINT_EVAL_DIR}/eval_results.json"
    echo "[done] experiment-2 evaluation=${FROZEN_EVAL_DIR}/eval_results.json"
fi
echo "[done] experiment-1 reused evaluation=${BASELINE_EVAL_RESULT}"
