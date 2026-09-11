#!/usr/bin/env bash
set -euo pipefail

# Controlled OneRec AD experiment:
# shared FP4-W/FP8-A MSE/LWC prefix -> MSE, ABC, ABC+boundary arms.
# The fixed calibration order is split into train [0, 512) and held-out [512, 1024).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-${GPU:-0}}"
MODEL_PATH="${MODEL_PATH:-/root/dataDisk/guowei/models/1.7B}"
DATA_DIR="${DATA_DIR:-/root/dataDisk/guowei/data/onerec_data/benchmark_data}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1024}"
PREFIX_CALIB_SAMPLES="${PREFIX_CALIB_SAMPLES:-128}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-512}"
HELDOUT_SAMPLES="${HELDOUT_SAMPLES:-512}"
NUM_LAYERS="${NUM_LAYERS:-28}"
EPOCHS="${EPOCHS:-20}"
LWC_LR="${LWC_LR:-1e-2}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.1}"
BOUNDARY_ONLY="${BOUNDARY_ONLY:-0}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-0}"
FINAL_ARMS="${FINAL_ARMS:-mse,abc,boundary}"
BOUNDARY_TOPK="${BOUNDARY_TOPK:-32}"
BOUNDARY_NEGATIVES="${BOUNDARY_NEGATIVES:-32}"
BOUNDARY_TIE_THRESHOLD="${BOUNDARY_TIE_THRESHOLD:-0.01}"
BOUNDARY_GAP_SCALE="${BOUNDARY_GAP_SCALE:-1.0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
DIAGNOSTICS_ONLY="${DIAGNOSTICS_ONLY:-0}"
DIAGNOSTIC_OVERWRITE="${DIAGNOSTIC_OVERWRITE:-0}"

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_STRING"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "GPUS must contain at least one GPU ID." >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID in GPUS=$GPUS: $gpu_id" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[$gpu_id]:-}" ]]; then
        echo "Duplicate GPU ID in GPUS=$GPUS: $gpu_id" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu_id]=1
done
GPU_COUNT=${#GPU_IDS[@]}
CALIB_GPU=${GPU_IDS[0]}

ARM_STRING="${FINAL_ARMS//[[:space:]]/}"
IFS=',' read -r -a REQUESTED_ARMS <<< "$ARM_STRING"
declare -a FINAL_ARM_INDICES=()
declare -A SEEN_ARMS=()
for arm_name in "${REQUESTED_ARMS[@]}"; do
    case "$arm_name" in
        mse) arm_index=0 ;;
        abc) arm_index=1 ;;
        boundary) arm_index=2 ;;
        *)
            echo "Unknown FINAL_ARMS entry: $arm_name (expected mse,abc,boundary)." >&2
            exit 2
            ;;
    esac
    if [[ -z "${SEEN_ARMS[$arm_name]:-}" ]]; then
        FINAL_ARM_INDICES+=("$arm_index")
        SEEN_ARMS[$arm_name]=1
    fi
done
if (( ${#FINAL_ARM_INDICES[@]} == 0 )); then
    echo "FINAL_ARMS must select at least one of mse,abc,boundary." >&2
    exit 2
fi
if [[ ! "$WEIGHT_GROUP_SIZE" =~ ^[0-9]+$ ]]; then
    echo "WEIGHT_GROUP_SIZE must be 0 or a positive integer." >&2
    exit 2
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
GROUP_TAG=""
if (( WEIGHT_GROUP_SIZE > 0 )); then
    GROUP_TAG="_g${WEIGHT_GROUP_SIZE}"
fi
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts/results/fake_quant/recommender/1p7b_ad_fp4w_fp8a_lwc${GROUP_TAG}_abc_boundary_prefix${PREFIX_CALIB_SAMPLES}_final${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}}"
PREFIX_OUTPUT_DIR="${RUN_ROOT}/shared_mse_lwc_prefix_calib${PREFIX_CALIB_SAMPLES}"
MSE_OUTPUT_DIR="${RUN_ROOT}/mse_lwc_control_train${TRAIN_SAMPLES}"
ABC_OUTPUT_DIR="${RUN_ROOT}/abc_lfq_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}"
if [[ "$BOUNDARY_ONLY" == "1" ]]; then
    BOUNDARY_LFQ_LOSS_WEIGHT=0.0
    BOUNDARY_STAGE_NAME="boundary-only"
    BOUNDARY_OUTPUT_DIR="${RUN_ROOT}/boundary_only_w${BOUNDARY_LOSS_WEIGHT}_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}"
    DEFAULT_DIAGNOSTIC_OUTPUT="${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_diagnostics_boundary_only_w${BOUNDARY_LOSS_WEIGHT}.json"
    DEFAULT_DIAGNOSTIC_LOG="${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_diagnostics_boundary_only_w${BOUNDARY_LOSS_WEIGHT}.log"
elif [[ "$BOUNDARY_ONLY" == "0" ]]; then
    BOUNDARY_LFQ_LOSS_WEIGHT=1.0
    BOUNDARY_STAGE_NAME="abc-lfq-boundary"
    BOUNDARY_OUTPUT_DIR="${RUN_ROOT}/abc_lfq_boundary_w${BOUNDARY_LOSS_WEIGHT}_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}"
    DEFAULT_DIAGNOSTIC_OUTPUT="${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_diagnostics.json"
    DEFAULT_DIAGNOSTIC_LOG="${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_diagnostics.log"
else
    echo "BOUNDARY_ONLY must be 0 or 1." >&2
    exit 2
fi
DIAGNOSTIC_OUTPUT="${DIAGNOSTIC_OUTPUT:-$DEFAULT_DIAGNOSTIC_OUTPUT}"
DIAGNOSTIC_LOG="${DIAGNOSTIC_LOG:-$DEFAULT_DIAGNOSTIC_LOG}"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
MSE_CHECKPOINT_DIR="${MSE_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
ABC_CHECKPOINT_DIR="${ABC_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
BOUNDARY_CHECKPOINT_DIR="${BOUNDARY_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"

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
if ! "$PYTHON_BIN" -c 'import math,sys; value=float(sys.argv[1]); sys.exit(not (math.isfinite(value) and value > 0.0))' "$BOUNDARY_LOSS_WEIGHT"; then
    echo "BOUNDARY_LOSS_WEIGHT must be finite and positive." >&2
    exit 2
fi
if [[ "$DIAGNOSTICS_ONLY" != "0" && "$DIAGNOSTICS_ONLY" != "1" ]]; then
    echo "DIAGNOSTICS_ONLY must be 0 or 1." >&2
    exit 2
fi
if [[ "$DIAGNOSTIC_OVERWRITE" != "0" && "$DIAGNOSTIC_OVERWRITE" != "1" ]]; then
    echo "DIAGNOSTIC_OVERWRITE must be 0 or 1." >&2
    exit 2
fi
if [[ "$DIAGNOSTICS_ONLY" == "1" && "$RUN_DIAGNOSTICS" != "1" ]]; then
    echo "DIAGNOSTICS_ONLY=1 requires RUN_DIAGNOSTICS=1." >&2
    exit 2
fi

FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

checkpoint_range_complete() {
    local checkpoint_dir="$1"
    local first_layer="$2"
    local last_layer="$3"
    local layer_idx
    for ((layer_idx = first_layer; layer_idx <= last_layer; layer_idx++)); do
        if [[ ! -s "${checkpoint_dir}/layer_$(printf '%02d' "$layer_idx").pt" ]]; then
            return 1
        fi
    done
    return 0
}

checkpoint_dir_has_files() {
    compgen -G "$1/layer_*.pt" >/dev/null
}

print_command() {
    printf '  %q' "$@"
    printf '\n'
}

run_calibration_stage() {
    local stage_name="$1"
    local stage_gpu="$2"
    local output_dir="$3"
    local checkpoint_dir="$4"
    local first_layer="$5"
    local last_layer="$6"
    shift 6
    if checkpoint_range_complete "$checkpoint_dir" "$first_layer" "$last_layer" && [[ "$OVERWRITE" != "1" ]]; then
        echo "[$stage_name] complete checkpoints found; skipping."
        return
    fi
    if checkpoint_dir_has_files "$checkpoint_dir" && [[ "$OVERWRITE" != "1" ]]; then
        echo "[$stage_name] partial checkpoints found at $checkpoint_dir; set OVERWRITE=1 or move them aside." >&2
        exit 3
    fi
    mkdir -p "$output_dir"
    local command=("$PYTHON_BIN" -u -m fake_quant.run_m1_onerec_ad "${COMMON_ARGS[@]}" --output_dir "$output_dir" "$@")
    if [[ "$OVERWRITE" == "1" ]]; then
        command+=(--overwrite)
    fi
    echo "[$stage_name] GPU=$stage_gpu output=$output_dir"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=$stage_gpu" "${command[@]}"
        return
    fi
    env CUDA_VISIBLE_DEVICES="$stage_gpu" "${command[@]}" 2>&1 | tee "$output_dir/train.log"
    if ! checkpoint_range_complete "$checkpoint_dir" "$first_layer" "$last_layer"; then
        echo "[$stage_name] expected checkpoints were not all produced." >&2
        exit 4
    fi
}

COMMON_ARGS=(
    --task ad
    --mode omniquant
    --model_path "$MODEL_PATH"
    --data_dir "$DATA_DIR"
    --device cuda:0
    --weight_quant_format fp4_e2m1
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --weight_group_size "$WEIGHT_GROUP_SIZE"
    --omni_lwc
    --omni_let_mode none
    --omni_epochs "$EPOCHS"
    --omni_epoch_eval_interval 0
    --omni_lwc_lr "$LWC_LR"
    --omni_init_lwc_logit "$INIT_LWC_LOGIT"
    --calibration_only
)

echo "[protocol] model=$MODEL_PATH task=ad seed=42"
echo "[protocol] gpus=$GPUS prefix_and_diagnostics_gpu=$CALIB_GPU"
echo "[protocol] prefix_calib=[0,$PREFIX_CALIB_SAMPLES) final_train=[0,$TRAIN_SAMPLES)"
echo "[protocol] heldout=[$TRAIN_SAMPLES,$CALIB_SAMPLES)"
echo "[protocol] FP4-E2M1-W/FP8-E4M3-A symmetric LWC-only, weight_group_size=${WEIGHT_GROUP_SIZE:-per_channel}, epochs=$EPOCHS"
echo "[protocol] final_arms=$FINAL_ARMS"
echo "[protocol] boundary_arm=$BOUNDARY_STAGE_NAME lfq_weight=$BOUNDARY_LFQ_LOSS_WEIGHT boundary_weight=$BOUNDARY_LOSS_WEIGHT"
echo "[protocol] run_root=$RUN_ROOT"

if [[ "$DIAGNOSTICS_ONLY" == "1" ]]; then
    echo "[training] skipped because DIAGNOSTICS_ONLY=1."
else
    run_calibration_stage "1/5 shared-prefix" "$CALIB_GPU" "$PREFIX_OUTPUT_DIR" "$PREFIX_CHECKPOINT_DIR" 0 "$PREFIX_LAST_LAYER" --layers "0-$PREFIX_LAST_LAYER" --calib_sample_size "$PREFIX_CALIB_SAMPLES" --omni_final_objective mse
fi

LFQ_SPLIT_ARGS=(
    --layers all
    --calib_sample_size "$CALIB_SAMPLES"
    --omni_train_sample_size "$TRAIN_SAMPLES"
    --omni_validation_sample_size "$HELDOUT_SAMPLES"
    --omni_prefix_checkpoint_dir "$PREFIX_CHECKPOINT_DIR"
    --omni_final_objective lfq_ce
    --omni_lfq_token_scope sid_slots
    --omni_lfq_vocab_scope s_abc
    --omni_lfq_slot_weights 1 1 1
    --omni_lfq_boundary_topk "$BOUNDARY_TOPK"
    --omni_lfq_boundary_negative_count "$BOUNDARY_NEGATIVES"
    --omni_lfq_boundary_tie_threshold "$BOUNDARY_TIE_THRESHOLD"
    --omni_lfq_boundary_gap_scale "$BOUNDARY_GAP_SCALE"
)

run_final_arm() {
    local arm_index="$1"
    local arm_gpu="$2"
    case "$arm_index" in
        0)
            run_calibration_stage "2/5 mse-control" "$arm_gpu" "$MSE_OUTPUT_DIR" "$MSE_CHECKPOINT_DIR" 0 "$FINAL_LAYER" --layers all --calib_sample_size "$TRAIN_SAMPLES" --omni_prefix_checkpoint_dir "$PREFIX_CHECKPOINT_DIR" --omni_final_objective mse
            ;;
        1)
            run_calibration_stage "3/5 abc-lfq" "$arm_gpu" "$ABC_OUTPUT_DIR" "$ABC_CHECKPOINT_DIR" 0 "$FINAL_LAYER" "${LFQ_SPLIT_ARGS[@]}" --omni_lfq_loss_weight 1.0 --omni_lfq_boundary_loss_weight 0.0
            ;;
        2)
            run_calibration_stage "4/5 $BOUNDARY_STAGE_NAME" "$arm_gpu" "$BOUNDARY_OUTPUT_DIR" "$BOUNDARY_CHECKPOINT_DIR" 0 "$FINAL_LAYER" "${LFQ_SPLIT_ARGS[@]}" --omni_lfq_loss_weight "$BOUNDARY_LFQ_LOSS_WEIGHT" --omni_lfq_boundary_loss_weight "$BOUNDARY_LOSS_WEIGHT"
            ;;
        *)
            echo "Unknown final-arm index: $arm_index" >&2
            return 2
            ;;
    esac
}

run_branch_worker() {
    local worker_index="$1"
    local worker_gpu="$2"
    local arm_position
    local arm_index
    for ((arm_position = worker_index; arm_position < ${#FINAL_ARM_INDICES[@]}; arm_position += GPU_COUNT)); do
        arm_index=${FINAL_ARM_INDICES[$arm_position]}
        run_final_arm "$arm_index" "$worker_gpu"
    done
}

if [[ "$DIAGNOSTICS_ONLY" != "1" ]]; then
    echo "[branches] scheduling ${#FINAL_ARM_INDICES[@]} final-layer arm(s) across GPUS=$GPUS"
    if [[ "$DRY_RUN" == "1" ]]; then
        for worker_index in "${!GPU_IDS[@]}"; do
            if (( worker_index >= ${#FINAL_ARM_INDICES[@]} )); then
                break
            fi
            run_branch_worker "$worker_index" "${GPU_IDS[$worker_index]}"
        done
    else
        BRANCH_PIDS=()
        for worker_index in "${!GPU_IDS[@]}"; do
            if (( worker_index >= ${#FINAL_ARM_INDICES[@]} )); then
                break
            fi
            run_branch_worker "$worker_index" "${GPU_IDS[$worker_index]}" &
            BRANCH_PIDS+=("$!")
        done
        branch_failure=0
        for branch_pid in "${BRANCH_PIDS[@]}"; do
            if ! wait "$branch_pid"; then
                branch_failure=1
            fi
        done
        if (( branch_failure != 0 )); then
            echo "At least one final-layer branch failed." >&2
            exit 5
        fi
    fi
fi


if [[ "$RUN_DIAGNOSTICS" != "1" ]]; then
    echo "[diagnostics] skipped because RUN_DIAGNOSTICS=$RUN_DIAGNOSTICS."
    exit 0
fi

DIAGNOSTIC_COMMAND=(
    "$PYTHON_BIN" -u -m fake_quant.evaluate_lfq_boundary_diagnostics
    --model_path "$MODEL_PATH"
    --data_dir "$DATA_DIR"
    --task ad
    --calib_sample_size "$CALIB_SAMPLES"
    --prefix_calib_sample_size "$PREFIX_CALIB_SAMPLES"
    --train_sample_size "$TRAIN_SAMPLES"
    --heldout_sample_size "$HELDOUT_SAMPLES"
    --mse_checkpoint_dir "$MSE_CHECKPOINT_DIR"
    --abc_checkpoint_dir "$ABC_CHECKPOINT_DIR"
    --boundary_checkpoint_dir "$BOUNDARY_CHECKPOINT_DIR"
    --expected_boundary_lfq_loss_weight "$BOUNDARY_LFQ_LOSS_WEIGHT"
    --topk "$BOUNDARY_TOPK"
    --negative_count "$BOUNDARY_NEGATIVES"
    --tie_threshold "$BOUNDARY_TIE_THRESHOLD"
    --gap_scale "$BOUNDARY_GAP_SCALE"
    --bootstrap_samples "$BOOTSTRAP_SAMPLES"
    --seed 42
    --dtype bfloat16
    --device cuda:0
    --output_path "$DIAGNOSTIC_OUTPUT"
)
if [[ "$OVERWRITE" == "1" || "$DIAGNOSTIC_OVERWRITE" == "1" ]]; then
    DIAGNOSTIC_COMMAND+=(--overwrite)
fi

if [[ -s "$DIAGNOSTIC_OUTPUT" && "$OVERWRITE" != "1" && "$DIAGNOSTIC_OVERWRITE" != "1" ]]; then
    echo "[diagnostics] output exists; skipping: $DIAGNOSTIC_OUTPUT"
elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[5/5 diagnostics] fixed held-out CE/KL/head-rank/top-32/boundary metrics on GPU=$CALIB_GPU"
    print_command env "CUDA_VISIBLE_DEVICES=$CALIB_GPU" "${DIAGNOSTIC_COMMAND[@]}"
else
    mkdir -p "$RUN_ROOT"
    env CUDA_VISIBLE_DEVICES="$CALIB_GPU" "${DIAGNOSTIC_COMMAND[@]}" 2>&1 | tee "$DIAGNOSTIC_LOG"
fi

echo "[done] $RUN_ROOT"
