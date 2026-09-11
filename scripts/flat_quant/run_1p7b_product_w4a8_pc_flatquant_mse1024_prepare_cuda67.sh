#!/usr/bin/env bash
set -euo pipefail

# Prepare the strict Product W4A8 FlatQuant MSE-1024 checkpoint in resumable
# stages matching the AD protocol:
#   prefix: layers 0-26, MSE, 128 records, joint transforms/LWC/LAC;
#   mse512: layer 27, 512 train + 512 held-out, joint optimization;
#   mse1024: layer 27 continuation on all 1024, frozen transforms, LWC/LAC.
# Select stages independently with RUN_STAGES, e.g. RUN_STAGES=prefix.

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
PREFIX_SAMPLES="${PREFIX_SAMPLES:-128}"
FINAL_SAMPLES="${FINAL_SAMPLES:-1024}"
MSE512_TRAIN_SAMPLES="${MSE512_TRAIN_SAMPLES:-512}"
MSE512_HELDOUT_SAMPLES="${MSE512_HELDOUT_SAMPLES:-512}"
EPOCHS="${EPOCHS:-15}"
TRANSFORM_LR="${TRANSFORM_LR:-5e-3}"
LWC_LR="${LWC_LR:-5e-2}"
LAC_LR="${LAC_LR:-5e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DIAG_ALPHA="${DIAG_ALPHA:-0.5}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
INIT_LAC_LOGIT="${INIT_LAC_LOGIT:-4.0}"
RUN_STAGES="${RUN_STAGES:-prefix mse512 mse1024 eval_mse1024}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/generalization/1p7b_product_w4a8_pc_prefix128_mse512_mse1024_epoch15}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
PREFIX_OUTPUT_DIR="${RUN_ROOT}/prefix_calib${PREFIX_SAMPLES}"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/product/flatquant_calibration"
MSE512_OUTPUT_DIR="${RUN_ROOT}/final_mse_train512_heldout512"
MSE512_CHECKPOINT_DIR="${MSE512_OUTPUT_DIR}/${MODEL_NAME}/product/flatquant_calibration"
MSE1024_OUTPUT_DIR="${RUN_ROOT}/final_mse1024_frozen_transform"
MSE1024_CHECKPOINT_DIR="${MSE1024_OUTPUT_DIR}/${MODEL_NAME}/product/flatquant_calibration"
MSE1024_EVAL_DIR="${RUN_ROOT}/mse1024_product_${EVAL_SAMPLE_SIZE}_eval"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

if (( NUM_LAYERS < 2 || PREFIX_SAMPLES != 128 || FINAL_SAMPLES != 1024 || MSE512_TRAIN_SAMPLES + MSE512_HELDOUT_SAMPLES != FINAL_SAMPLES )); then
    echo "Protocol requires prefix=128 and a 512/512 split of 1024 final records." >&2
    exit 2
fi
for value in "${OVERWRITE}" "${DRY_RUN}"; do
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "OVERWRITE and DRY_RUN must be 0 or 1." >&2
        exit 2
    fi
done

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} != 2 )) || [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "This launcher requires two distinct GPUs; got GPUS=${GPUS}." >&2
    exit 2
fi
CALIB_GPU="${GPU_IDS[0]}"

want_prefix=0
want_mse512=0
want_mse1024=0
want_eval=0
for stage in ${RUN_STAGES}; do
    case "${stage}" in
        prefix) want_prefix=1 ;;
        mse512) want_mse512=1 ;;
        mse1024) want_mse1024=1 ;;
        eval_mse1024) want_eval=1 ;;
        *) echo "Unknown RUN_STAGES entry: ${stage}." >&2; exit 2 ;;
    esac
done

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

print_command() {
    printf "  %q" "$@"
    printf "\n"
}

checkpoint_range_complete() {
    local checkpoint_dir="$1" first_layer="$2" last_layer="$3"
    local layer_idx checkpoint_name
    for ((layer_idx = first_layer; layer_idx <= last_layer; layer_idx++)); do
        printf -v checkpoint_name "layer_%02d.pt" "${layer_idx}"
        if [[ ! -s "${checkpoint_dir}/${checkpoint_name}" ]]; then
            return 1
        fi
    done
}

checkpoint_dir_has_files() {
    compgen -G "$1/layer_*.pt" >/dev/null
}

BASE_ARGS=(
    --task product
    --mode flatquant_core
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --weight_quant_format fp4_e2m1
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --flat_lac
    --flat_learn_lac
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
    --omni_final_objective mse
)

run_stage() {
    local label="$1" output_dir="$2" checkpoint_dir="$3" first_layer="$4" last_layer="$5"
    shift 5
    if checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        return 3
    fi
    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        "${BASE_ARGS[@]}"
        --output_dir "${output_dir}"
        --calibration_only
        "$@"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi
    echo "[${label}] physical_gpu=${CALIB_GPU} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${CALIB_GPU}" "${command[@]}"
        return
    fi
    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${CALIB_GPU}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}"
}

if (( want_prefix == 1 )); then
    run_stage \
        "prefix" "${PREFIX_OUTPUT_DIR}" "${PREFIX_CHECKPOINT_DIR}" \
        0 "${PREFIX_LAST_LAYER}" \
        --flat_learn_transform \
        --layers "0-${PREFIX_LAST_LAYER}" \
        --calib_sample_size "${PREFIX_SAMPLES}"
fi

if (( want_mse512 == 1 )); then
    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${PREFIX_CHECKPOINT_DIR}" 0 "${PREFIX_LAST_LAYER}"; then
        echo "mse512 requires a complete prefix: ${PREFIX_CHECKPOINT_DIR}" >&2
        exit 3
    fi
    run_stage \
        "mse512" "${MSE512_OUTPUT_DIR}" "${MSE512_CHECKPOINT_DIR}" \
        0 "${FINAL_LAYER}" \
        --flat_learn_transform \
        --layers all \
        --calib_sample_size "${FINAL_SAMPLES}" \
        --flat_train_sample_size "${MSE512_TRAIN_SAMPLES}" \
        --flat_validation_sample_size "${MSE512_HELDOUT_SAMPLES}" \
        --flat_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}"
fi

if (( want_mse1024 == 1 )); then
    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${MSE512_CHECKPOINT_DIR}" 0 "${FINAL_LAYER}"; then
        echo "mse1024 requires a complete mse512 checkpoint: ${MSE512_CHECKPOINT_DIR}" >&2
        exit 3
    fi
    run_stage \
        "mse1024" "${MSE1024_OUTPUT_DIR}" "${MSE1024_CHECKPOINT_DIR}" \
        0 "${FINAL_LAYER}" \
        --no-flat_learn_transform \
        --layers all \
        --calib_sample_size "${FINAL_SAMPLES}" \
        --flat_train_sample_size "${FINAL_SAMPLES}" \
        --flat_validation_sample_size 0 \
        --flat_finetune_checkpoint_dir "${MSE512_CHECKPOINT_DIR}"
fi

if (( want_eval == 1 )); then
    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${MSE1024_CHECKPOINT_DIR}" 0 "${FINAL_LAYER}"; then
        echo "eval_mse1024 requires: ${MSE1024_CHECKPOINT_DIR}" >&2
        exit 3
    fi
    if [[ -s "${MSE1024_EVAL_DIR}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[eval_mse1024] complete result found; skipping."
    elif [[ "${OVERWRITE}" != "1" && ( -d "${MSE1024_EVAL_DIR}" || -d "${MSE1024_EVAL_DIR}.shards" ) ]]; then
        echo "[eval_mse1024] partial output found at ${MSE1024_EVAL_DIR}." >&2
        exit 3
    else
        echo "[eval_mse1024] physical_gpus=${GPUS} output=${MSE1024_EVAL_DIR}"
        env \
            TASK=product \
            GPUS="${GPUS}" \
            MODEL_PATH="${MODEL_PATH}" \
            DATA_DIR="${DATA_DIR}" \
            OUTPUT_DIR="${MSE1024_EVAL_DIR}" \
            PYTHON_BIN="${PYTHON_BIN}" \
            OVERWRITE="${OVERWRITE}" \
            DRY_RUN="${DRY_RUN}" \
            bash scripts/flat_quant/run_sharded_eval_cuda.sh \
                --mode flatquant_core \
                --layers all \
                --weight_quant_format fp4_e2m1 \
                --activation_quant_format fp8_e4m3fn \
                --weight_quant_scheme symmetric \
                --weight_group_size 0 \
                --omni_lwc \
                --flat_lac \
                --flat_learn_lac \
                --no-flat_learn_transform \
                --flat_transform_kind kronecker \
                --flat_transform_init random_orthogonal \
                --flat_train_sample_size "${FINAL_SAMPLES}" \
                --flat_validation_sample_size 0 \
                --flat_diag_alpha "${DIAG_ALPHA}" \
                --omni_final_objective mse \
                --flat_load_checkpoint_dir "${MSE1024_CHECKPOINT_DIR}" \
                --calib_sample_size "${PREFIX_SAMPLES}" \
                --eval_sample_size "${EVAL_SAMPLE_SIZE}"
    fi
fi

echo "[done] Product MSE preparation root=${RUN_ROOT}"
echo "[done] mse1024 checkpoints=${MSE1024_CHECKPOINT_DIR}"
