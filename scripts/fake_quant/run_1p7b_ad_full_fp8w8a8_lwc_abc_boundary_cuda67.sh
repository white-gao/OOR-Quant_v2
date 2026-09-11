#!/usr/bin/env bash
set -euo pipefail

# Serial controlled comparison on OneRec-1.7B AD-full:
#   1. OmniQuant-LWC FP8-W/FP8-A + ABC-LFQ.
#   2. OmniQuant-LWC FP8-W/FP8-A + ABC-LFQ + boundary (weight 0.3).
#
# Layers 0-26 share one MSE/LWC prefix trained on calibration samples [0, 128).
# Layer 27 is independently reinitialized and trained on all 1024 calibration
# samples for each arm.  The two full AD evaluations run serially; each one is
# round-robin sharded across physical GPUs 6 and 7 by default.
#
# Usage after activating the benchmark environment:
#   bash scripts/fake_quant/run_1p7b_ad_full_fp8w8a8_lwc_abc_boundary_cuda67.sh
#
# Useful overrides:
#   DRY_RUN=1 bash scripts/fake_quant/run_1p7b_ad_full_fp8w8a8_lwc_abc_boundary_cuda67.sh
#   OVERWRITE=1 bash scripts/fake_quant/run_1p7b_ad_full_fp8w8a8_lwc_abc_boundary_cuda67.sh

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
PREFIX_CALIB_SAMPLES="${PREFIX_CALIB_SAMPLES:-128}"
FINAL_CALIB_SAMPLES="${FINAL_CALIB_SAMPLES:-1024}"
NUM_LAYERS="${NUM_LAYERS:-28}"
EPOCHS="${EPOCHS:-20}"
LWC_LR="${LWC_LR:-1e-2}"
INIT_LWC_LOGIT="${INIT_LWC_LOGIT:-4.0}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.3}"
BOUNDARY_TOPK="${BOUNDARY_TOPK:-32}"
BOUNDARY_NEGATIVES="${BOUNDARY_NEGATIVES:-32}"
BOUNDARY_TIE_THRESHOLD="${BOUNDARY_TIE_THRESHOLD:-0.01}"
BOUNDARY_GAP_SCALE="${BOUNDARY_GAP_SCALE:-1.0}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_full_fp8w8a8_lwc_abc_boundary_prefix${PREFIX_CALIB_SAMPLES}_final${FINAL_CALIB_SAMPLES}}"
PREFIX_OUTPUT_DIR="${PREFIX_OUTPUT_DIR:-${RESULTS_ROOT}/shared_mse_lwc_prefix_calib${PREFIX_CALIB_SAMPLES}}"
ABC_CALIB_DIR="${ABC_CALIB_DIR:-${RESULTS_ROOT}/abc_lfq_final${FINAL_CALIB_SAMPLES}_calibration}"
ABC_EVAL_DIR="${ABC_EVAL_DIR:-${RESULTS_ROOT}/abc_lfq_ad_full}"
BOUNDARY_CALIB_DIR="${BOUNDARY_CALIB_DIR:-${RESULTS_ROOT}/abc_lfq_boundary_w${BOUNDARY_LOSS_WEIGHT}_final${FINAL_CALIB_SAMPLES}_calibration}"
BOUNDARY_EVAL_DIR="${BOUNDARY_EVAL_DIR:-${RESULTS_ROOT}/abc_lfq_boundary_w${BOUNDARY_LOSS_WEIGHT}_ad_full}"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
ABC_CHECKPOINT_DIR="${ABC_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
BOUNDARY_CHECKPOINT_DIR="${BOUNDARY_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}." >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1; got ${DRY_RUN}." >&2
    exit 2
fi
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
if ! "${PYTHON_BIN}" -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and x > 0.0))' "${BOUNDARY_LOSS_WEIGHT}"; then
    echo "BOUNDARY_LOSS_WEIGHT must be finite and positive." >&2
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
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    WRITE_ARGS+=(--overwrite)
fi

COMMON_OMNI_ARGS=(
    --task ad
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --mode omniquant
    --weight_quant_format fp8_e4m3fn
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --weight_group_size 0
    --omni_lwc
    --omni_let_mode none
    --omni_epochs "${EPOCHS}"
    --omni_epoch_eval_interval 0
    --omni_lwc_lr "${LWC_LR}"
    --omni_init_lwc_logit "${INIT_LWC_LOGIT}"
)

LFQ_ARGS=(
    --layers all
    --omni_final_objective lfq_ce
    --omni_lfq_token_scope sid_slots
    --omni_lfq_vocab_scope s_abc
    --omni_lfq_slot_weights 1 1 1
    --omni_lfq_loss_weight 1.0
    --omni_lfq_boundary_topk "${BOUNDARY_TOPK}"
    --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}"
    --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}"
    --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}"
)

print_command() {
    printf '  %q' "$@"
    printf '\n'
}

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

run_calibration_stage() {
    local label="$1"
    local output_dir="$2"
    local checkpoint_dir="$3"
    local first_layer="$4"
    local last_layer="$5"
    shift 5

    if checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping calibration."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}; set OVERWRITE=1 or move them aside." >&2
        exit 3
    fi

    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        "${COMMON_OMNI_ARGS[@]}"
        --output_dir "${output_dir}"
        --calibration_only
        "${WRITE_ARGS[@]}"
        "$@"
    )
    echo "[${label}] physical_gpu=${CALIB_GPU} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${CALIB_GPU}" "${command[@]}"
        return
    fi

    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${CALIB_GPU}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    if ! checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}"; then
        echo "[${label}] expected checkpoint range ${first_layer}-${last_layer} was not produced." >&2
        exit 4
    fi
}

run_full_evaluation() {
    local label="$1"
    local output_dir="$2"
    local checkpoint_dir="$3"
    local boundary_weight="$4"

    if [[ -s "${output_dir}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete eval_results.json found; skipping full evaluation."
        return
    fi
    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}"; then
        echo "[${label}] checkpoint set is incomplete: ${checkpoint_dir}" >&2
        exit 4
    fi

    echo "[${label}] evaluation_gpus=${GPUS} output=${output_dir}"
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
        --weight_quant_format fp8_e4m3fn \
        --activation_quant_format fp8_e4m3fn \
        --weight_quant_scheme symmetric \
        --weight_group_size 0 \
        --omni_lwc \
        --omni_let_mode none \
        --omni_final_objective lfq_ce \
        --omni_lfq_token_scope sid_slots \
        --omni_lfq_vocab_scope s_abc \
        --omni_lfq_slot_weights 1 1 1 \
        --omni_lfq_loss_weight 1.0 \
        --omni_lfq_boundary_loss_weight "${boundary_weight}" \
        --omni_lfq_boundary_topk "${BOUNDARY_TOPK}" \
        --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}" \
        --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}" \
        --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}" \
        --omni_load_checkpoint_dir "${checkpoint_dir}" \
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}"
}

compare_eval_results() {
    local abc_path="${ABC_EVAL_DIR}/eval_results.json"
    local boundary_path="${BOUNDARY_EVAL_DIR}/eval_results.json"
    "${PYTHON_BIN}" - "${abc_path}" "${boundary_path}" "${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

abc_path, boundary_path, model_name = map(str, sys.argv[1:])

def load_metrics(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload[model_name]["ad"]["test"]

abc = load_metrics(abc_path)
boundary = load_metrics(boundary_path)
metric_names = (
    "pass@1", "pass@4", "pass@8", "pass@16", "pass@32",
    "recall@1", "recall@4", "recall@8", "recall@16", "recall@32",
    "pid_pass@1", "pid_pass@4", "pid_pass@8", "pid_pass@16", "pid_pass@32",
    "pid_recall@1", "pid_recall@4", "pid_recall@8", "pid_recall@16", "pid_recall@32",
)
print("[comparison] metric             ABC-LFQ       ABC+boundary   delta(pp)")
for name in metric_names:
    if name not in abc or name not in boundary:
        continue
    abc_value = float(abc[name])
    boundary_value = float(boundary[name])
    print(
        f"[comparison] {name:<17} "
        f"{100.0 * abc_value:>10.4f}%   "
        f"{100.0 * boundary_value:>10.4f}%   "
        f"{100.0 * (boundary_value - abc_value):>+9.4f}"
    )
PY
}

echo "[protocol] model=${MODEL_PATH} task=ad gpus=${GPUS} calibration_gpu=${CALIB_GPU}"
echo "[protocol] deployment-matched FP8-E4M3-W/FP8-E4M3-A, symmetric OmniQuant-LWC, LET disabled"
echo "[protocol] shared prefix layers=0-${PREFIX_LAST_LAYER} calib=[0,${PREFIX_CALIB_SAMPLES})"
echo "[protocol] final layer=${FINAL_LAYER} calib=[0,${FINAL_CALIB_SAMPLES}) for both independent arms"
echo "[protocol] ABC weight=1.0; boundary arm weight=${BOUNDARY_LOSS_WEIGHT}; no B/C prefix expansion"
echo "[protocol] methods execute serially; each AD-${EVAL_SAMPLE_SIZE} evaluation is sharded across GPUS=${GPUS}"
echo "[protocol] results_root=${RESULTS_ROOT}"

echo "Stage 1/5: shared MSE/LWC prefix on ${PREFIX_CALIB_SAMPLES} samples."
run_calibration_stage \
    "shared-prefix" \
    "${PREFIX_OUTPUT_DIR}" \
    "${PREFIX_CHECKPOINT_DIR}" \
    0 "${PREFIX_LAST_LAYER}" \
    --layers "0-${PREFIX_LAST_LAYER}" \
    --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
    --omni_final_objective mse

echo "Stage 2/5: ABC-LFQ final-layer calibration on all ${FINAL_CALIB_SAMPLES} samples."
run_calibration_stage \
    "abc-lfq" \
    "${ABC_CALIB_DIR}" \
    "${ABC_CHECKPOINT_DIR}" \
    0 "${FINAL_LAYER}" \
    "${LFQ_ARGS[@]}" \
    --calib_sample_size "${FINAL_CALIB_SAMPLES}" \
    --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}" \
    --omni_lfq_boundary_loss_weight 0.0

echo "Stage 3/5: ABC-LFQ AD-${EVAL_SAMPLE_SIZE} evaluation."
run_full_evaluation "abc-lfq" "${ABC_EVAL_DIR}" "${ABC_CHECKPOINT_DIR}" 0.0

echo "Stage 4/5: ABC-LFQ + boundary final-layer calibration on all ${FINAL_CALIB_SAMPLES} samples."
run_calibration_stage \
    "abc-lfq-boundary" \
    "${BOUNDARY_CALIB_DIR}" \
    "${BOUNDARY_CHECKPOINT_DIR}" \
    0 "${FINAL_LAYER}" \
    "${LFQ_ARGS[@]}" \
    --calib_sample_size "${FINAL_CALIB_SAMPLES}" \
    --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}" \
    --omni_lfq_boundary_loss_weight "${BOUNDARY_LOSS_WEIGHT}"

echo "Stage 5/5: ABC-LFQ + boundary AD-${EVAL_SAMPLE_SIZE} evaluation."
run_full_evaluation "abc-lfq-boundary" "${BOUNDARY_EVAL_DIR}" "${BOUNDARY_CHECKPOINT_DIR}" "${BOUNDARY_LOSS_WEIGHT}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[comparison] skipped in DRY_RUN mode."
else
    compare_eval_results
fi

echo "[done] ABC-LFQ:          ${ABC_EVAL_DIR}/eval_results.json"
echo "[done] ABC-LFQ+boundary: ${BOUNDARY_EVAL_DIR}/eval_results.json"
