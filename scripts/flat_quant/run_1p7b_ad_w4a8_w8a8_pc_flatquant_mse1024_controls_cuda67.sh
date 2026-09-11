#!/usr/bin/env bash
set -euo pipefail

# Serial FlatQuant MSE-1024 continuation controls for W4A8 and W8A8.
#
# Each arm starts from its completed formal FlatQuant-MSE checkpoint, restores
# layers 0-26 without training, and continues layer 27 for 15 epochs on all
# 1024 AD calibration samples. Matrix/diagonal transforms stay frozen; only
# LWC/LAC parameters are optimized with hidden-state MSE. A two-GPU AD-full
# evaluation follows each training arm.

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
LWC_LR="${LWC_LR:-5e-2}"
LAC_LR="${LAC_LR:-5e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DIAG_ALPHA="${DIAG_ALPHA:-0.5}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
INIT_LAC_LOGIT="${INIT_LAC_LOGIT:-4.0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

W4_BASE_RUN_ROOT="${W4_BASE_RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_ad_w4a8_pc_prefix128_final512_heldout512_epoch15}"
W8_BASE_RUN_ROOT="${W8_BASE_RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_ad_w8a8_pc_prefix128_final512_heldout512_epoch15}"
W4_BASE_CHECKPOINT_DIR="${W4_BASE_CHECKPOINT_DIR:-${W4_BASE_RUN_ROOT}/final_train512_heldout512/1.7B/ad/flatquant_calibration}"
W8_BASE_CHECKPOINT_DIR="${W8_BASE_CHECKPOINT_DIR:-${W8_BASE_RUN_ROOT}/final_train512_heldout512/1.7B/ad/flatquant_calibration}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-${ARTIFACTS_ROOT}/results/flat_quant/task_alignment}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
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
    echo "This control requires CALIB_SAMPLES=TRAIN_SAMPLES=1024." >&2
    echo "Got calibration=${CALIB_SAMPLES}, train=${TRAIN_SAMPLES}." >&2
    exit 2
fi
if [[ "${EVAL_SAMPLE_SIZE}" != "full" ]]; then
    echo "EVAL_SAMPLE_SIZE must be full for comparison with existing AD-full arms." >&2
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

run_training() {
    local quant_label="$1"
    local weight_format="$2"
    local base_checkpoint_dir="$3"
    local output_dir="$4"
    local checkpoint_dir="$5"

    if ! checkpoint_complete "${base_checkpoint_dir}"; then
        echo "[${quant_label}] source FlatQuant-MSE checkpoint is missing or incomplete:" >&2
        echo "  ${base_checkpoint_dir}" >&2
        return 3
    fi
    if checkpoint_complete "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${quant_label} training] complete checkpoints found; skipping training."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${quant_label} training] partial checkpoints found at ${checkpoint_dir}." >&2
        echo "Move the partial output aside or rerun with OVERWRITE=1." >&2
        return 3
    fi

    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        --task ad
        --mode flatquant_core
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --device cuda:0
        --layers all
        --calib_sample_size "${CALIB_SAMPLES}"
        --flat_train_sample_size "${TRAIN_SAMPLES}"
        --flat_validation_sample_size 0
        --flat_finetune_checkpoint_dir "${base_checkpoint_dir}"
        --weight_quant_format "${weight_format}"
        --activation_quant_format fp8_e4m3fn
        --weight_quant_scheme symmetric
        --weight_group_size 0
        --omni_lwc
        --flat_lac
        --flat_transform_kind kronecker
        --flat_transform_init random_orthogonal
        --no-flat_learn_transform
        --flat_epochs "${EPOCHS}"
        --flat_epoch_eval_interval 0
        --flat_lwc_lr "${LWC_LR}"
        --flat_lac_lr "${LAC_LR}"
        --flat_weight_decay "${WEIGHT_DECAY}"
        --flat_diag_alpha "${DIAG_ALPHA}"
        --flat_init_lwc_logit "${INIT_LWC_LOGIT}"
        --flat_init_lac_logit "${INIT_LAC_LOGIT}"
        --flat_normalize_mse_gradient
        --omni_final_objective mse
        --calibration_only
        --output_dir "${output_dir}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi

    echo "[${quant_label} training] physical_gpu=${TRAIN_GPU} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${TRAIN_GPU}" "${command[@]}"
        return
    fi
    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    if ! checkpoint_complete "${checkpoint_dir}"; then
        echo "[${quant_label} training] expected checkpoints 0-${FINAL_LAYER} were not produced." >&2
        return 4
    fi
}

run_full_eval() {
    local quant_label="$1"
    local weight_format="$2"
    local checkpoint_dir="$3"
    local output_dir="$4"

    if [[ -s "${output_dir}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[${quant_label} evaluation] complete AD-full result found; skipping evaluation."
        return
    fi
    if [[ "${OVERWRITE}" != "1" && ( -d "${output_dir}" || -d "${output_dir}.shards" ) ]]; then
        echo "[${quant_label} evaluation] partial output found at ${output_dir}." >&2
        echo "Move it aside or rerun with OVERWRITE=1." >&2
        return 3
    fi

    echo "[${quant_label} evaluation] physical_gpus=${GPUS} output=${output_dir}"
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
            --weight_quant_format "${weight_format}" \
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
            --omni_final_objective mse \
            --calib_sample_size 128 \
            --eval_sample_size full
}

run_control() {
    local quant_label="$1"
    local weight_format="$2"
    local base_checkpoint_dir="$3"
    local run_root="${RUN_ROOT_BASE}/1p7b_ad_${quant_label}_pc_from_mse_train${TRAIN_SAMPLES}_epoch${EPOCHS}"
    local output_dir="${run_root}/mse1024_frozen_transform_control"
    local checkpoint_dir="${output_dir}/${MODEL_NAME}/ad/flatquant_calibration"
    local eval_dir="${run_root}/mse1024_frozen_transform_control_ad_full_eval"

    echo
    echo "[${quant_label}] FP8 activation, per-output-channel weight, MSE-1024 continuation"
    echo "[${quant_label}] source=${base_checkpoint_dir}"
    echo "[${quant_label}] transforms=frozen, trainable=LWC/LAC, final_layer=${FINAL_LAYER}"
    run_training \
        "${quant_label}" \
        "${weight_format}" \
        "${base_checkpoint_dir}" \
        "${output_dir}" \
        "${checkpoint_dir}"
    if [[ "${RUN_EVAL}" == "1" ]]; then
        run_full_eval "${quant_label}" "${weight_format}" "${checkpoint_dir}" "${eval_dir}"
    fi
    echo "[${quant_label} done] checkpoints=${checkpoint_dir}"
    if [[ "${RUN_EVAL}" == "1" ]]; then
        echo "[${quant_label} done] evaluation=${eval_dir}/eval_results.json"
    fi
}

echo "[protocol] Qwen3-1.7B AD FlatQuant MSE-1024 frozen-transform controls"
echo "[protocol] order=W4A8 -> W8A8"
echo "[protocol] train=[0,1024), validation=none, epochs=${EPOCHS}, seed=42"
echo "[protocol] training_gpu=${TRAIN_GPU}, evaluation_gpus=${GPUS}"

run_control "w4a8" "fp4_e2m1" "${W4_BASE_CHECKPOINT_DIR}"
run_control "w8a8" "fp8_e4m3fn" "${W8_BASE_CHECKPOINT_DIR}"

echo
echo "[done] W4A8 and W8A8 FlatQuant MSE-1024 controls completed."
