#!/usr/bin/env bash
set -euo pipefail

# Train one group-wise OmniQuant-LWC experiment:
#   layers 0--26: MSE prefix on calibration samples [0, 128)
#   layer 27: ABC CE + 0.3 boundary on [0, 512), held-out [512, 1024)
# After training, report the final checkpoint's fixed-state block-output MSE,
# then evaluate CE/KL, retention, agreement, and boundary diagnostics on the
# untouched held-out half.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/root/dataDisk/guowei/models/1.7B}"
DATA_DIR="${DATA_DIR:-/root/dataDisk/guowei/data/onerec_data/benchmark_data}"
GPUS="${GPUS:-${GPU:-0}}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1024}"
PREFIX_CALIB_SAMPLES="${PREFIX_CALIB_SAMPLES:-128}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-512}"
HELDOUT_SAMPLES="${HELDOUT_SAMPLES:-512}"
NUM_LAYERS="${NUM_LAYERS:-28}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-128}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.3}"
BOUNDARY_TOPK="${BOUNDARY_TOPK:-32}"
BOUNDARY_NEGATIVES="${BOUNDARY_NEGATIVES:-32}"
BOUNDARY_TIE_THRESHOLD="${BOUNDARY_TIE_THRESHOLD:-0.01}"
BOUNDARY_GAP_SCALE="${BOUNDARY_GAP_SCALE:-1.0}"
RUN_HELDOUT_DIAGNOSTICS="${RUN_HELDOUT_DIAGNOSTICS:-1}"
DIAGNOSTIC_OVERWRITE="${DIAGNOSTIC_OVERWRITE:-0}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! "$WEIGHT_GROUP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "WEIGHT_GROUP_SIZE must be a positive integer." >&2
    exit 2
fi

DIAGNOSTIC_GPU="${DIAGNOSTIC_GPU:-${GPUS%%,*}}"
DIAGNOSTIC_GPU="${DIAGNOSTIC_GPU//[[:space:]]/}"
if [[ ! "$DIAGNOSTIC_GPU" =~ ^[0-9]+$ ]]; then
    echo "DIAGNOSTIC_GPU must resolve to one physical GPU ID." >&2
    exit 2
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts/results/fake_quant/recommender/1p7b_ad_fp4w_fp8a_lwc_g${WEIGHT_GROUP_SIZE}_abc_boundary_prefix${PREFIX_CALIB_SAMPLES}_final${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}}"
BOUNDARY_OUTPUT_DIR="${RUN_ROOT}/abc_lfq_boundary_w${BOUNDARY_LOSS_WEIGHT}_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}"
FINAL_LAYER=$((NUM_LAYERS - 1))
BOUNDARY_CHECKPOINT_DIR="${BOUNDARY_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
FINAL_CHECKPOINT="${BOUNDARY_CHECKPOINT_DIR}/layer_$(printf '%02d' "$FINAL_LAYER").pt"
DIAGNOSTIC_OUTPUT="${DIAGNOSTIC_OUTPUT:-${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_groupwise_g${WEIGHT_GROUP_SIZE}_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}.json}"
DIAGNOSTIC_LOG="${DIAGNOSTIC_LOG:-${RUN_ROOT}/heldout${HELDOUT_SAMPLES}_groupwise_g${WEIGHT_GROUP_SIZE}_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}.log}"

export MODEL_PATH
export DATA_DIR
export GPUS
export CALIB_SAMPLES
export PREFIX_CALIB_SAMPLES
export TRAIN_SAMPLES
export HELDOUT_SAMPLES
export NUM_LAYERS
export WEIGHT_GROUP_SIZE
export BOUNDARY_LOSS_WEIGHT
export BOUNDARY_TOPK
export BOUNDARY_NEGATIVES
export BOUNDARY_TIE_THRESHOLD
export BOUNDARY_GAP_SCALE
export RUN_ROOT
export FINAL_ARMS=boundary
export BOUNDARY_ONLY=0
export RUN_DIAGNOSTICS=0

echo "[groupwise-g${WEIGHT_GROUP_SIZE}] ABC CE + ${BOUNDARY_LOSS_WEIGHT} boundary"
echo "[groupwise-g${WEIGHT_GROUP_SIZE}] run_root=$RUN_ROOT"

bash "${SCRIPT_DIR}/run_1p7b_ad_fp4w_fp8a_lwc_abc_boundary_calib1024.sh"

DIAGNOSTIC_COMMAND=(
    "$PYTHON_BIN" -u -m fake_quant.evaluate_lfq_boundary_diagnostics
    --model_path "$MODEL_PATH"
    --data_dir "$DATA_DIR"
    --task ad
    --calib_sample_size "$CALIB_SAMPLES"
    --prefix_calib_sample_size "$PREFIX_CALIB_SAMPLES"
    --train_sample_size "$TRAIN_SAMPLES"
    --heldout_sample_size "$HELDOUT_SAMPLES"
    --single_checkpoint_dir "$BOUNDARY_CHECKPOINT_DIR"
    --single_method_name "groupwise_g${WEIGHT_GROUP_SIZE}_abc_boundary"
    --expected_boundary_lfq_loss_weight 1.0
    --expected_boundary_loss_weight "$BOUNDARY_LOSS_WEIGHT"
    --expected_omni_let_mode none
    --expected_omni_let_init smoothquant
    --topk "$BOUNDARY_TOPK"
    --negative_count "$BOUNDARY_NEGATIVES"
    --tie_threshold "$BOUNDARY_TIE_THRESHOLD"
    --gap_scale "$BOUNDARY_GAP_SCALE"
    --seed 42
    --dtype bfloat16
    --device cuda:0
    --output_path "$DIAGNOSTIC_OUTPUT"
)
if [[ "$OVERWRITE" == "1" || "$DIAGNOSTIC_OVERWRITE" == "1" ]]; then
    DIAGNOSTIC_COMMAND+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$RUN_HELDOUT_DIAGNOSTICS" == "1" ]]; then
        echo "[held-out diagnostics dry-run] GPU=$DIAGNOSTIC_GPU"
        printf '  %q' env "CUDA_VISIBLE_DEVICES=$DIAGNOSTIC_GPU" "${DIAGNOSTIC_COMMAND[@]}"
        printf '\n'
    fi
    exit 0
fi

if [[ ! -s "$FINAL_CHECKPOINT" ]]; then
    echo "Missing final checkpoint for MSE summary: $FINAL_CHECKPOINT" >&2
    exit 4
fi

"$PYTHON_BIN" - "$FINAL_CHECKPOINT" <<'PY'
import sys

import torch

checkpoint_path = sys.argv[1]
try:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(checkpoint_path, map_location="cpu")

initial = float(state["initial_mse_loss"])
final = float(state["final_mse_loss"])
delta = final - initial
relative = 100.0 * delta / initial if initial != 0.0 else float("nan")
direction = "decreased" if delta < 0.0 else (
    "unchanged" if delta == 0.0 else "increased"
)
print(
    f"[groupwise-mse-summary] layer={state['layer_idx']} "
    f"initial_mse={initial:.8e} final_mse={final:.8e} "
    f"delta={delta:+.8e} relative={relative:+.4f}% result={direction}"
)
PY

if [[ "$RUN_HELDOUT_DIAGNOSTICS" != "1" ]]; then
    echo "[held-out diagnostics] skipped because RUN_HELDOUT_DIAGNOSTICS=$RUN_HELDOUT_DIAGNOSTICS."
elif [[ -s "$DIAGNOSTIC_OUTPUT" && "$OVERWRITE" != "1" && "$DIAGNOSTIC_OVERWRITE" != "1" ]]; then
    echo "[held-out diagnostics] output exists; skipping: $DIAGNOSTIC_OUTPUT"
else
    mkdir -p "$RUN_ROOT"
    echo "[held-out diagnostics] GPU=$DIAGNOSTIC_GPU output=$DIAGNOSTIC_OUTPUT"
    env CUDA_VISIBLE_DEVICES="$DIAGNOSTIC_GPU" "${DIAGNOSTIC_COMMAND[@]}" 2>&1 | tee "$DIAGNOSTIC_LOG"
fi

echo "[done] checkpoint=$FINAL_CHECKPOINT"
echo "[done] heldout_json=$DIAGNOSTIC_OUTPUT"
