#!/usr/bin/env bash
set -euo pipefail

# Controlled global-LET training comparison on OneRec-1.7B AD:
#   shared layers 0-26: FP4-W/FP8-A, LWC + learned LET (ones init), MSE
#   layer-27 arm 1:     MSE
#   layer-27 arm 2:     ABC-LFQ
#   layer-27 arm 3:     ABC-LFQ + boundary (weight 0.3)
#
# The fixed calibration order is split into train [0, 512) and held-out
# [512, 1024). The MSE arm uses the same first 512 training records. All arms
# restore the exact same global-LWC+LET prefix and independently initialize the
# final block with LWC logit 4.0 and LET scale 1.
#
# Usage:
#   bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_global_lwc_let_ones_three_arm.sh
#
# Useful overrides:
#   GPUS=4,5 bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_global_lwc_let_ones_three_arm.sh
#   DRY_RUN=1 bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_global_lwc_let_ones_three_arm.sh
#   OVERWRITE=1 bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_global_lwc_let_ones_three_arm.sh
#   RUN_DIAGNOSTICS=0 bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_global_lwc_let_ones_three_arm.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
GPUS="${GPUS:-6,7}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1024}"
PREFIX_CALIB_SAMPLES="${PREFIX_CALIB_SAMPLES:-128}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-512}"
HELDOUT_SAMPLES="${HELDOUT_SAMPLES:-512}"
NUM_LAYERS="${NUM_LAYERS:-28}"
EPOCHS="${EPOCHS:-20}"
LWC_LR="${LWC_LR:-1e-2}"
LET_LR="${LET_LR:-5e-3}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.3}"
BOUNDARY_TOPK="${BOUNDARY_TOPK:-32}"
BOUNDARY_NEGATIVES="${BOUNDARY_NEGATIVES:-32}"
BOUNDARY_TIE_THRESHOLD="${BOUNDARY_TIE_THRESHOLD:-0.01}"
BOUNDARY_GAP_SCALE="${BOUNDARY_GAP_SCALE:-1.0}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
DIAGNOSTIC_OVERWRITE="${DIAGNOSTIC_OVERWRITE:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

RUN_ROOT="${RUN_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_fp4w_fp8a_global_lwc_let_ones_abc_boundary_prefix${PREFIX_CALIB_SAMPLES}_final${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}}"
PREFIX_OUTPUT_DIR="${PREFIX_OUTPUT_DIR:-${RUN_ROOT}/shared_mse_lwc_let_ones_prefix_calib${PREFIX_CALIB_SAMPLES}}"
MSE_OUTPUT_DIR="${MSE_OUTPUT_DIR:-${RUN_ROOT}/mse_lwc_let_ones_control_train${TRAIN_SAMPLES}}"
ABC_OUTPUT_DIR="${ABC_OUTPUT_DIR:-${RUN_ROOT}/abc_lfq_lwc_let_ones_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}}"
BOUNDARY_OUTPUT_DIR="${BOUNDARY_OUTPUT_DIR:-${RUN_ROOT}/abc_lfq_boundary_w${BOUNDARY_LOSS_WEIGHT}_lwc_let_ones_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}}"
DIAGNOSTIC_OUTPUT_PATH="${DIAGNOSTIC_OUTPUT_PATH:-${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_global_lwc_let_ones_diagnostics.json}"
DIAGNOSTIC_LOG_PATH="${DIAGNOSTIC_LOG_PATH:-${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_diagnostics.log}"

for boolean_name in OVERWRITE DRY_RUN RUN_DIAGNOSTICS DIAGNOSTIC_OVERWRITE; do
    boolean_value="${!boolean_name}"
    if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
        echo "${boolean_name} must be 0 or 1; got ${boolean_value}." >&2
        exit 2
    fi
done
if (( TRAIN_SAMPLES + HELDOUT_SAMPLES != CALIB_SAMPLES )); then
    echo "TRAIN_SAMPLES + HELDOUT_SAMPLES must equal CALIB_SAMPLES." >&2
    exit 2
fi
if (( PREFIX_CALIB_SAMPLES <= 0 || PREFIX_CALIB_SAMPLES > TRAIN_SAMPLES )); then
    echo "PREFIX_CALIB_SAMPLES must be in [1, TRAIN_SAMPLES]." >&2
    exit 2
fi
if (( NUM_LAYERS < 2 )); then
    echo "NUM_LAYERS must be at least 2." >&2
    exit 2
fi
if (( BOOTSTRAP_SAMPLES <= 0 )); then
    echo "BOOTSTRAP_SAMPLES must be positive." >&2
    exit 2
fi
for numeric_arg in "${LWC_LR}" "${LET_LR}" "${BOUNDARY_LOSS_WEIGHT}"; do
    if ! "${PYTHON_BIN}" -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and x > 0.0))' "${numeric_arg}"; then
        echo "LWC_LR, LET_LR, and BOUNDARY_LOSS_WEIGHT must be finite and positive." >&2
        exit 2
    fi
done

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "GPUS must contain at least one GPU ID." >&2
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
GPU_COUNT=${#GPU_IDS[@]}
PREFIX_GPU=${GPU_IDS[0]}

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
MSE_CHECKPOINT_DIR="${MSE_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
ABC_CHECKPOINT_DIR="${ABC_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
BOUNDARY_CHECKPOINT_DIR="${BOUNDARY_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

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

COMMON_ARGS=(
    --task ad
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --mode omniquant
    --weight_quant_format fp4_e2m1
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --omni_let_mode learned
    --omni_let_init ones
    --omni_epochs "${EPOCHS}"
    --omni_epoch_eval_interval 0
    --omni_lwc_lr "${LWC_LR}"
    --omni_let_lr "${LET_LR}"
    --omni_init_lwc_logit "${INIT_LWC_LOGIT}"
    --calibration_only
)

run_calibration_stage() {
    local label="$1"
    local physical_gpu="$2"
    local output_dir="$3"
    local checkpoint_dir="$4"
    local first_layer="$5"
    local last_layer="$6"
    shift 6

    if checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        echo "Set OVERWRITE=1 or move the partial output aside before retrying." >&2
        return 3
    fi

    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        "${COMMON_ARGS[@]}"
        --output_dir "${output_dir}"
        "$@"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi

    echo "[${label}] physical_gpu=${physical_gpu} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${physical_gpu}" "${command[@]}"
        return
    fi

    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${physical_gpu}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    if ! checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}"; then
        echo "[${label}] expected checkpoints for layers ${first_layer}-${last_layer} were not produced." >&2
        return 4
    fi
}

echo "[protocol] model=${MODEL_PATH} task=ad seed=42 gpus=${GPUS}"
echo "[protocol] deployment-matched FP4-E2M1-W/FP8-E4M3-A, symmetric LWC + global learned LET, LET-init=ones"
echo "[protocol] prefix layers=0-${PREFIX_LAST_LAYER} MSE calib=[0,${PREFIX_CALIB_SAMPLES})"
echo "[protocol] final layer=${FINAL_LAYER} train=[0,${TRAIN_SAMPLES}) heldout=[${TRAIN_SAMPLES},${CALIB_SAMPLES})"
echo "[protocol] epochs=${EPOCHS} lwc_lr=${LWC_LR} let_lr=${LET_LR} boundary_weight=${BOUNDARY_LOSS_WEIGHT}"
echo "[protocol] results_root=${RUN_ROOT}"

run_calibration_stage \
    "1/5 shared-global-LET-prefix" \
    "${PREFIX_GPU}" \
    "${PREFIX_OUTPUT_DIR}" \
    "${PREFIX_CHECKPOINT_DIR}" \
    0 "${PREFIX_LAST_LAYER}" \
    --layers "0-${PREFIX_LAST_LAYER}" \
    --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
    --omni_final_objective mse

LFQ_SPLIT_ARGS=(
    --layers all
    --calib_sample_size "${CALIB_SAMPLES}"
    --omni_train_sample_size "${TRAIN_SAMPLES}"
    --omni_validation_sample_size "${HELDOUT_SAMPLES}"
    --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}"
    --omni_final_objective lfq_ce
    --omni_lfq_token_scope sid_slots
    --omni_lfq_vocab_scope s_abc
    --omni_lfq_slot_weights 1 1 1
    --omni_lfq_boundary_topk "${BOUNDARY_TOPK}"
    --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}"
    --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}"
    --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}"
)

run_final_arm() {
    local arm_index="$1"
    local physical_gpu="$2"
    case "${arm_index}" in
        0)
            run_calibration_stage \
                "2/5 global-LET-mse-control" \
                "${physical_gpu}" \
                "${MSE_OUTPUT_DIR}" \
                "${MSE_CHECKPOINT_DIR}" \
                0 "${FINAL_LAYER}" \
                --layers all \
                --calib_sample_size "${TRAIN_SAMPLES}" \
                --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}" \
                --omni_final_objective mse
            ;;
        1)
            run_calibration_stage \
                "3/5 global-LET-abc-lfq" \
                "${physical_gpu}" \
                "${ABC_OUTPUT_DIR}" \
                "${ABC_CHECKPOINT_DIR}" \
                0 "${FINAL_LAYER}" \
                "${LFQ_SPLIT_ARGS[@]}" \
                --omni_lfq_loss_weight 1.0 \
                --omni_lfq_boundary_loss_weight 0.0
            ;;
        2)
            run_calibration_stage \
                "4/5 global-LET-abc-lfq-boundary" \
                "${physical_gpu}" \
                "${BOUNDARY_OUTPUT_DIR}" \
                "${BOUNDARY_CHECKPOINT_DIR}" \
                0 "${FINAL_LAYER}" \
                "${LFQ_SPLIT_ARGS[@]}" \
                --omni_lfq_loss_weight 1.0 \
                --omni_lfq_boundary_loss_weight "${BOUNDARY_LOSS_WEIGHT}"
            ;;
        *)
            echo "Unknown final-arm index: ${arm_index}" >&2
            return 2
            ;;
    esac
}

run_branch_worker() {
    local worker_index="$1"
    local physical_gpu="$2"
    local arm_index
    for ((arm_index = worker_index; arm_index < 3; arm_index += GPU_COUNT)); do
        run_final_arm "${arm_index}" "${physical_gpu}"
    done
}

echo "[branches] scheduling three matched final-layer arms across GPUS=${GPUS}"
if [[ "${DRY_RUN}" == "1" ]]; then
    for worker_index in "${!GPU_IDS[@]}"; do
        if (( worker_index >= 3 )); then
            break
        fi
        run_branch_worker "${worker_index}" "${GPU_IDS[${worker_index}]}"
    done
else
    BRANCH_PIDS=()
    for worker_index in "${!GPU_IDS[@]}"; do
        if (( worker_index >= 3 )); then
            break
        fi
        run_branch_worker "${worker_index}" "${GPU_IDS[${worker_index}]}" &
        BRANCH_PIDS+=("$!")
    done
    branch_failure=0
    for branch_pid in "${BRANCH_PIDS[@]}"; do
        if ! wait "${branch_pid}"; then
            branch_failure=1
        fi
    done
    if (( branch_failure != 0 )); then
        echo "At least one final-layer arm failed." >&2
        exit 5
    fi
fi

run_heldout_diagnostics() {
    if [[ "${RUN_DIAGNOSTICS}" != "1" ]]; then
        echo "[5/5 held-out-diagnostics] disabled by RUN_DIAGNOSTICS=0."
        return
    fi
    if [[ -s "${DIAGNOSTIC_OUTPUT_PATH}" && "${OVERWRITE}" != "1" && "${DIAGNOSTIC_OVERWRITE}" != "1" ]]; then
        echo "[5/5 held-out-diagnostics] existing result found; skipping: ${DIAGNOSTIC_OUTPUT_PATH}"
        return
    fi

    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.evaluate_lfq_boundary_diagnostics
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --task ad
        --calib_sample_size "${CALIB_SAMPLES}"
        --prefix_calib_sample_size "${PREFIX_CALIB_SAMPLES}"
        --train_sample_size "${TRAIN_SAMPLES}"
        --heldout_sample_size "${HELDOUT_SAMPLES}"
        --mse_checkpoint_dir "${MSE_CHECKPOINT_DIR}"
        --abc_checkpoint_dir "${ABC_CHECKPOINT_DIR}"
        --boundary_checkpoint_dir "${BOUNDARY_CHECKPOINT_DIR}"
        --expected_boundary_lfq_loss_weight 1.0
        --expected_omni_let_mode learned
        --expected_omni_let_init ones
        --topk "${BOUNDARY_TOPK}"
        --negative_count "${BOUNDARY_NEGATIVES}"
        --tie_threshold "${BOUNDARY_TIE_THRESHOLD}"
        --gap_scale "${BOUNDARY_GAP_SCALE}"
        --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
        --seed 42
        --dtype bfloat16
        --device cuda:0
        --output_path "${DIAGNOSTIC_OUTPUT_PATH}"
    )
    if [[ "${OVERWRITE}" == "1" || "${DIAGNOSTIC_OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi

    echo "[5/5 held-out-diagnostics] physical_gpu=${PREFIX_GPU} output=${DIAGNOSTIC_OUTPUT_PATH}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${PREFIX_GPU}" "${command[@]}"
        return
    fi

    mkdir -p "${RUN_ROOT}"
    env CUDA_VISIBLE_DEVICES="${PREFIX_GPU}" "${command[@]}" 2>&1 | tee "${DIAGNOSTIC_LOG_PATH}"
    if [[ ! -s "${DIAGNOSTIC_OUTPUT_PATH}" ]]; then
        echo "[5/5 held-out-diagnostics] expected output was not produced." >&2
        return 6
    fi
}

run_heldout_diagnostics

echo "[done] shared prefix: ${PREFIX_CHECKPOINT_DIR}"
echo "[done] MSE control:   ${MSE_CHECKPOINT_DIR}"
echo "[done] ABC-LFQ:       ${ABC_CHECKPOINT_DIR}"
echo "[done] ABC+boundary:  ${BOUNDARY_CHECKPOINT_DIR}"
if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    echo "[done] diagnostics:   ${DIAGNOSTIC_OUTPUT_PATH}"
fi
