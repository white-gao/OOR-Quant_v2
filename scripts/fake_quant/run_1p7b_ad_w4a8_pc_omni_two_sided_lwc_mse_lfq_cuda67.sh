#!/usr/bin/env bash
set -euo pipefail

# OmniQuant FP two-sided LWC parameter-capacity control:
#   shared layers 0-26 MSE prefix on 128 records;
#   layer 27 MSE and ABC+0.3*boundary arms on all 1024 records.
# The FP4 codebook and symmetric scale are unchanged; only the positive and
# negative pre-QDQ clipping factors become independent.

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
EPOCHS="${EPOCHS:-20}"
LWC_LR="${LWC_LR:-1e-2}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.3}"
RUN_ARMS="${RUN_ARMS:-mse lfq}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_w4a8_pc_two_sided_lwc_prefix${PREFIX_SAMPLES}_final${FINAL_SAMPLES}}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
PREFIX_OUTPUT_DIR="${RUN_ROOT}/shared_mse_prefix_calib${PREFIX_SAMPLES}"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
MSE_OUTPUT_DIR="${RUN_ROOT}/mse_final${FINAL_SAMPLES}"
MSE_CHECKPOINT_DIR="${MSE_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
MSE_EVAL_DIR="${RUN_ROOT}/mse_ad_${EVAL_SAMPLE_SIZE}"
LFQ_OUTPUT_DIR="${RUN_ROOT}/abc_boundary_w0p3_final${FINAL_SAMPLES}"
LFQ_CHECKPOINT_DIR="${LFQ_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
LFQ_EVAL_DIR="${RUN_ROOT}/abc_boundary_w0p3_ad_${EVAL_SAMPLE_SIZE}"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

validate_boolean() {
    local name="$1" value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}
validate_boolean RUN_EVAL "${RUN_EVAL}"
validate_boolean OVERWRITE "${OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"
if (( NUM_LAYERS < 2 || PREFIX_SAMPLES != 128 || FINAL_SAMPLES != 1024 )); then
    echo "This control requires 28-style layers, prefix=128 and final=1024." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} != 2 )); then
    echo "This launcher requires exactly two GPUs; got GPUS=${GPUS}." >&2
    exit 2
fi
if [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" || ! "${GPU_IDS[0]}" =~ ^[0-9]+$ || ! "${GPU_IDS[1]}" =~ ^[0-9]+$ ]]; then
    echo "GPUS must contain two distinct numeric IDs." >&2
    exit 2
fi
PREFIX_GPU="${GPU_IDS[0]}"
MSE_GPU="${GPU_IDS[0]}"
LFQ_GPU="${GPU_IDS[1]}"

want_mse=0
want_lfq=0
for arm in ${RUN_ARMS}; do
    case "${arm}" in
        mse) want_mse=1 ;;
        lfq) want_lfq=1 ;;
        *) echo "RUN_ARMS supports only 'mse' and 'lfq'; got ${arm}." >&2; exit 2 ;;
    esac
done
if (( want_mse == 0 && want_lfq == 0 )); then
    echo "RUN_ARMS must select at least one arm." >&2
    exit 2
fi

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

COMMON_ARGS=(
    --task ad
    --mode omniquant
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --weight_quant_format fp4_e2m1
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --omni_symmetric_lwc_mode two_sided
    --weight_group_size 0
    --omni_lwc
    --omni_let_mode none
    --omni_epochs "${EPOCHS}"
    --omni_epoch_eval_interval 0
    --omni_lwc_lr "${LWC_LR}"
    --omni_init_lwc_logit "${INIT_LWC_LOGIT}"
)

run_calibration() {
    local label="$1" gpu="$2" output_dir="$3" checkpoint_dir="$4" first_layer="$5" last_layer="$6"
    shift 6
    if checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        return 3
    fi
    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        "${COMMON_ARGS[@]}"
        --output_dir "${output_dir}"
        --calibration_only
        "$@"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi
    echo "[${label}] physical_gpu=${gpu} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${gpu}" "${command[@]}"
        return
    fi
    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}"
}

run_eval() {
    local label="$1" checkpoint_dir="$2" output_dir="$3" objective="$4" boundary_weight="$5"
    if [[ "${RUN_EVAL}" != "1" ]]; then
        return
    fi
    if [[ -s "${output_dir}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete evaluation found; skipping."
        return
    fi
    if [[ "${OVERWRITE}" != "1" && ( -d "${output_dir}" || -d "${output_dir}.shards" ) ]]; then
        echo "[${label}] partial evaluation found at ${output_dir}." >&2
        return 3
    fi
    echo "[${label}] evaluation_gpus=${GPUS} output=${output_dir}"
    env \
        TASK=ad \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        OUTPUT_DIR="${output_dir}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh \
            --mode omniquant \
            --layers all \
            --weight_quant_format fp4_e2m1 \
            --activation_quant_format fp8_e4m3fn \
            --weight_quant_scheme symmetric \
            --omni_symmetric_lwc_mode two_sided \
            --weight_group_size 0 \
            --omni_lwc \
            --omni_let_mode none \
            --omni_final_objective "${objective}" \
            --omni_lfq_loss_weight 1.0 \
            --omni_lfq_boundary_loss_weight "${boundary_weight}" \
            --omni_load_checkpoint_dir "${checkpoint_dir}" \
            --calib_sample_size "${PREFIX_SAMPLES}" \
            --eval_sample_size "${EVAL_SAMPLE_SIZE}"
}

echo "[protocol] OmniQuant W4A8 per-output-channel FP two-sided LWC"
echo "[protocol] prefix=[0,${PREFIX_SAMPLES}) final=[0,${FINAL_SAMPLES}) epochs=${EPOCHS} seed=42"
echo "[protocol] codebook=FP4-E2M1 symmetric; clipping=independent upper/lower"
echo "[protocol] run_arms=${RUN_ARMS} run_root=${RUN_ROOT}"

run_calibration \
    "prefix MSE" "${PREFIX_GPU}" \
    "${PREFIX_OUTPUT_DIR}" "${PREFIX_CHECKPOINT_DIR}" \
    0 "${PREFIX_LAST_LAYER}" \
    --layers "0-${PREFIX_LAST_LAYER}" \
    --calib_sample_size "${PREFIX_SAMPLES}" \
    --omni_final_objective mse

pids=()
labels=()
if (( want_mse == 1 )); then
    run_calibration \
        "final MSE" "${MSE_GPU}" \
        "${MSE_OUTPUT_DIR}" "${MSE_CHECKPOINT_DIR}" \
        0 "${FINAL_LAYER}" \
        --layers all \
        --calib_sample_size "${FINAL_SAMPLES}" \
        --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}" \
        --omni_final_objective mse &
    pids+=("$!")
    labels+=("MSE")
fi
if (( want_lfq == 1 )); then
    run_calibration \
        "final LFQ" "${LFQ_GPU}" \
        "${LFQ_OUTPUT_DIR}" "${LFQ_CHECKPOINT_DIR}" \
        0 "${FINAL_LAYER}" \
        --layers all \
        --calib_sample_size "${FINAL_SAMPLES}" \
        --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}" \
        --omni_final_objective lfq_ce \
        --omni_lfq_loss_weight 1.0 \
        --omni_lfq_boundary_loss_weight "${BOUNDARY_WEIGHT}" &
    pids+=("$!")
    labels+=("LFQ")
fi
failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
        echo "[${labels[${index}]}] calibration failed." >&2
        failed=1
    fi
done
if (( failed != 0 )); then
    exit 4
fi

if (( want_mse == 1 )); then
    run_eval "MSE" "${MSE_CHECKPOINT_DIR}" "${MSE_EVAL_DIR}" mse 0
fi
if (( want_lfq == 1 )); then
    run_eval "LFQ" "${LFQ_CHECKPOINT_DIR}" "${LFQ_EVAL_DIR}" lfq_ce "${BOUNDARY_WEIGHT}"
fi

echo "[done] run_root=${RUN_ROOT}"
