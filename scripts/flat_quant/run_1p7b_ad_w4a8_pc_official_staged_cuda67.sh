#!/usr/bin/env bash
set -euo pipefail

# Formal fake-quant FlatQuant reproduction for Qwen3-1.7B OneRec tasks.
# Stage 1 trains layers 0-26 on calibration records [0, 128).
# Stage 2 restores that exact prefix and trains layer 27 on [0, 512), while
# [512, 1024) is held out for MSE diagnostics only. Final-epoch parameters are
# retained in both stages. Full-task evaluation is sharded over GPUS by default.

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
GPUS="${GPUS:-6,7}"
WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-fp4_e2m1}"
ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT:-fp8_e4m3fn}"
QUANT_LABEL="${QUANT_LABEL:-w4a8}"
NUM_LAYERS="${NUM_LAYERS:-28}"
PREFIX_CALIB_SAMPLES="${PREFIX_CALIB_SAMPLES:-128}"
FINAL_CALIB_SAMPLES="${FINAL_CALIB_SAMPLES:-1024}"
FINAL_TRAIN_SAMPLES="${FINAL_TRAIN_SAMPLES:-512}"
FINAL_HELDOUT_SAMPLES="${FINAL_HELDOUT_SAMPLES:-512}"
EPOCHS="${EPOCHS:-15}"
TRANSFORM_LR="${TRANSFORM_LR:-5e-3}"
LWC_LR="${LWC_LR:-5e-2}"
LAC_LR="${LAC_LR:-5e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DIAG_ALPHA="${DIAG_ALPHA:-0.5}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
INIT_LAC_LOGIT="${INIT_LAC_LOGIT:-4.0}"
SMOKE="${SMOKE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_${TASK}_${QUANT_LABEL}_pc_prefix${PREFIX_CALIB_SAMPLES}_final${FINAL_TRAIN_SAMPLES}_heldout${FINAL_HELDOUT_SAMPLES}_epoch${EPOCHS}}"
PREFIX_OUTPUT_DIR="${RUN_ROOT}/prefix_calib${PREFIX_CALIB_SAMPLES}"
FINAL_OUTPUT_DIR="${RUN_ROOT}/final_train${FINAL_TRAIN_SAMPLES}_heldout${FINAL_HELDOUT_SAMPLES}"
EVAL_OUTPUT_DIR="${RUN_ROOT}/${TASK}_${EVAL_SAMPLE_SIZE}_eval"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/${TASK}/flatquant_calibration"
FINAL_CHECKPOINT_DIR="${FINAL_OUTPUT_DIR}/${MODEL_NAME}/${TASK}/flatquant_calibration"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}

validate_boolean SMOKE "${SMOKE}"
validate_boolean RUN_EVAL "${RUN_EVAL}"
validate_boolean OVERWRITE "${OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"
case "${TASK}" in
    ad|product|video|label_pred) ;;
    *) echo "Unsupported TASK=${TASK}." >&2; exit 2 ;;
esac
if (( NUM_LAYERS < 2 )); then
    echo "NUM_LAYERS must be at least 2." >&2
    exit 2
fi
if (( PREFIX_CALIB_SAMPLES <= 0 || FINAL_TRAIN_SAMPLES <= 0 || FINAL_HELDOUT_SAMPLES <= 0 )); then
    echo "Calibration split sizes must be positive." >&2
    exit 2
fi
if (( FINAL_TRAIN_SAMPLES + FINAL_HELDOUT_SAMPLES != FINAL_CALIB_SAMPLES )); then
    echo "FINAL_TRAIN_SAMPLES + FINAL_HELDOUT_SAMPLES must equal FINAL_CALIB_SAMPLES." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS="," read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "GPUS must contain at least one GPU ID." >&2
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
CALIB_GPU="${GPU_IDS[0]}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

print_command() {
    printf "  %q" "$@"
    printf "\n"
}

checkpoint_range_complete() {
    local checkpoint_dir="$1"
    local first_layer="$2"
    local last_layer="$3"
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

COMMON_ARGS=(
    --task "${TASK}"
    --mode flatquant_core
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
    --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --flat_lac
    --flat_learn_lac
    --flat_learn_transform
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
)

run_stage() {
    local label="$1"
    local output_dir="$2"
    local checkpoint_dir="$3"
    local first_layer="$4"
    local last_layer="$5"
    shift 5

    if checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        echo "Move the partial output aside or set OVERWRITE=1." >&2
        exit 3
    fi

    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        "${COMMON_ARGS[@]}"
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
    if ! checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}"; then
        echo "[${label}] expected checkpoints were not produced." >&2
        exit 4
    fi
}

echo "[protocol] formal FlatQuant fake-quant, Qwen3-1.7B, task=${TASK}, seed=42"
echo "[protocol] weight=${WEIGHT_QUANT_FORMAT} activation=${ACTIVATION_QUANT_FORMAT} / per-output-channel weight scale"
echo "[protocol] SVD-Cayley matrices + three diagonals + two-sided LWC + two-sided LAC"
echo "[protocol] AdamW lr=${TRANSFORM_LR}/${LWC_LR}/${LAC_LR}, wd=${WEIGHT_DECAY}, epochs=${EPOCHS}, best_epoch=off"
echo "[protocol] calibration_gpu=${CALIB_GPU} evaluation_gpus=${GPUS}"
echo "[protocol] run_root=${RUN_ROOT}"

if [[ "${SMOKE}" == "1" ]]; then
    SMOKE_OUTPUT_DIR="${RUN_ROOT}/smoke_layer0_calib4_epoch1"
    SMOKE_CHECKPOINT_DIR="${SMOKE_OUTPUT_DIR}/${MODEL_NAME}/${TASK}/flatquant_calibration"
    EPOCHS=1
    COMMON_ARGS=(
        --task "${TASK}"
        --mode flatquant_core
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --device cuda:0
        --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
        --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
        --weight_quant_scheme symmetric
        --weight_group_size 0
        --omni_lwc
        --flat_lac
        --flat_learn_lac
        --flat_learn_transform
        --flat_transform_kind kronecker
        --flat_transform_init random_orthogonal
        --flat_epochs 1
        --flat_epoch_eval_interval 0
        --flat_transform_lr "${TRANSFORM_LR}"
        --flat_lwc_lr "${LWC_LR}"
        --flat_lac_lr "${LAC_LR}"
        --flat_weight_decay "${WEIGHT_DECAY}"
        --flat_diag_alpha "${DIAG_ALPHA}"
        --flat_init_lwc_logit "${INIT_LWC_LOGIT}"
        --flat_init_lac_logit "${INIT_LAC_LOGIT}"
        --flat_normalize_mse_gradient
    )
    run_stage "smoke" "${SMOKE_OUTPUT_DIR}" "${SMOKE_CHECKPOINT_DIR}" 0 0 --layers 0 --calib_sample_size 4
    echo "[done] smoke checkpoint=${SMOKE_CHECKPOINT_DIR}/layer_00.pt"
    exit 0
fi

PREFIX_STAGE_ARGS=(
    --layers "0-${PREFIX_LAST_LAYER}"
    --calib_sample_size "${PREFIX_CALIB_SAMPLES}"
)
PREFIX_STAGE=(
    "1/2 prefix"
    "${PREFIX_OUTPUT_DIR}"
    "${PREFIX_CHECKPOINT_DIR}"
    0
    "${PREFIX_LAST_LAYER}"
)
run_stage "${PREFIX_STAGE[@]}" "${PREFIX_STAGE_ARGS[@]}"

FINAL_STAGE_ARGS=(
    --layers all
    --calib_sample_size "${FINAL_CALIB_SAMPLES}"
    --flat_train_sample_size "${FINAL_TRAIN_SAMPLES}"
    --flat_validation_sample_size "${FINAL_HELDOUT_SAMPLES}"
    --flat_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}"
)
FINAL_STAGE=(
    "2/2 final"
    "${FINAL_OUTPUT_DIR}"
    "${FINAL_CHECKPOINT_DIR}"
    0
    "${FINAL_LAYER}"
)
run_stage "${FINAL_STAGE[@]}" "${FINAL_STAGE_ARGS[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
    if (( ${#GPU_IDS[@]} < 2 )); then
        echo "RUN_EVAL=1 requires at least two GPUs in GPUS." >&2
        exit 2
    fi
    echo "[evaluation] physical_gpus=${GPUS} output=${EVAL_OUTPUT_DIR}"
    EVAL_COMMAND=(
        bash scripts/flat_quant/run_sharded_eval_cuda.sh
        --mode flatquant_core
        --layers all
        --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
        --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
        --weight_quant_scheme symmetric
        --weight_group_size 0
        --omni_lwc
        --flat_lac
        --flat_learn_lac
        --flat_learn_transform
        --flat_transform_kind kronecker
        --flat_transform_init random_orthogonal
        --flat_train_sample_size "${FINAL_TRAIN_SAMPLES}"
        --flat_validation_sample_size "${FINAL_HELDOUT_SAMPLES}"
        --flat_diag_alpha "${DIAG_ALPHA}"
        --flat_load_checkpoint_dir "${FINAL_CHECKPOINT_DIR}"
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}"
        --eval_sample_size "${EVAL_SAMPLE_SIZE}"
    )
    EVAL_ENV=(
        TASK="${TASK}"
        GPUS="${GPUS}"
        MODEL_PATH="${MODEL_PATH}"
        DATA_DIR="${DATA_DIR}"
        OUTPUT_DIR="${EVAL_OUTPUT_DIR}"
        PYTHON_BIN="${PYTHON_BIN}"
        OVERWRITE="${OVERWRITE}"
        DRY_RUN="${DRY_RUN}"
    )
    env "${EVAL_ENV[@]}" "${EVAL_COMMAND[@]}"
fi

echo "[done] prefix checkpoints=${PREFIX_CHECKPOINT_DIR}"
echo "[done] final checkpoints=${FINAL_CHECKPOINT_DIR}"
if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "[done] evaluation=${EVAL_OUTPUT_DIR}/eval_results.json"
else
    echo "[done] evaluation skipped; rerun with RUN_EVAL=1 to evaluate checkpoints."
fi
