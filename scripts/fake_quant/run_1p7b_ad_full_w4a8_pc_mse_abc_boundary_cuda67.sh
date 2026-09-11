#!/usr/bin/env bash
set -euo pipefail

# Complete the two missing W4A8 per-output-channel learned arms:
#   1. OmniQuant-LWC with final-layer MSE.
#   2. OmniQuant-LWC with ABC-CE + 0.3 boundary.
#
# Reuse the completed deployment-matched per-channel Layer 0-26 prefix trained
# on calibration records [0, 128). Independently train Layer 27 on all 1024
# calibration records. The final-layer jobs run on cards 6 and 7 concurrently;
# the two full-AD evaluations then run serially and are sharded over both cards.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
DEFAULT_PYTHON_BIN="/home/guowei/miniconda3/envs/benchmark/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
    DEFAULT_PYTHON_BIN="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
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
CALIB_OVERWRITE="${CALIB_OVERWRITE:-0}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_full_w4a8_per_channel_prefix${PREFIX_CALIB_SAMPLES}_final${FINAL_CALIB_SAMPLES}}"
PREFIX_OUTPUT_DIR="${PREFIX_OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_fp4w_fp8a_lwc_abc_boundary_prefix128_final512_heldout512/shared_mse_lwc_prefix_calib128}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
PREFIX_CHECKPOINT_DIR="${PREFIX_CHECKPOINT_DIR:-${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration}"

MSE_CALIB_DIR="${MSE_CALIB_DIR:-${RESULTS_ROOT}/omniquant_lwc_mse_final${FINAL_CALIB_SAMPLES}_calibration}"
MSE_CHECKPOINT_DIR="${MSE_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
MSE_EVAL_DIR="${MSE_EVAL_DIR:-${RESULTS_ROOT}/omniquant_lwc_mse_ad_${EVAL_SAMPLE_SIZE}}"
BOUNDARY_CALIB_DIR="${BOUNDARY_CALIB_DIR:-${RESULTS_ROOT}/omniquant_lwc_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}_final${FINAL_CALIB_SAMPLES}_calibration}"
BOUNDARY_CHECKPOINT_DIR="${BOUNDARY_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
BOUNDARY_EVAL_DIR="${BOUNDARY_EVAL_DIR:-${RESULTS_ROOT}/omniquant_lwc_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}_ad_${EVAL_SAMPLE_SIZE}}"

PC_RTN_RESULT="${PC_RTN_RESULT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_product_full_fp4w_fp8a_rtn_smoothquant_deployment_matched/rtn_fp4w_fp8a_ad_full/eval_results.json}"
G128_W4_ROOT="${G128_W4_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_full_w4w8a8_g128_core6_prefix128_final1024/w4a8_g128}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_ROOT}/w4a8_per_channel_vs_g128_summary.json}"

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}

validate_boolean CALIB_OVERWRITE "${CALIB_OVERWRITE}"
validate_boolean EVAL_OVERWRITE "${EVAL_OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"
if (( PREFIX_CALIB_SAMPLES != 128 || FINAL_CALIB_SAMPLES != 1024 )); then
    echo "This controlled comparison requires prefix=128 and final=1024." >&2
    exit 2
fi
if (( NUM_LAYERS < 2 )); then
    echo "NUM_LAYERS must be at least 2." >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and abs(x - 0.3) < 1e-12))' "${BOUNDARY_LOSS_WEIGHT}"; then
    echo "This controlled comparison requires BOUNDARY_LOSS_WEIGHT=0.3." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "GPUS must contain at least two GPU IDs; got ${GPUS}." >&2
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
MSE_GPU="${GPU_IDS[0]}"
BOUNDARY_GPU="${GPU_IDS[1]}"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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

if ! checkpoint_range_complete "${PREFIX_CHECKPOINT_DIR}" 0 "${PREFIX_LAST_LAYER}"; then
    echo "The per-channel prefix is incomplete: ${PREFIX_CHECKPOINT_DIR}" >&2
    echo "Expected non-empty checkpoints for layers 0-${PREFIX_LAST_LAYER}." >&2
    exit 3
fi

COMMON_ARGS=(
    --task ad
    --mode omniquant
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --device cuda:0
    --weight_quant_format fp4_e2m1
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
    --omni_final_objective lfq_ce
    --omni_lfq_token_scope sid_slots
    --omni_lfq_vocab_scope s_abc
    --omni_lfq_slot_weights 1 1 1
    --omni_lfq_loss_weight 1.0
    --omni_lfq_boundary_loss_weight "${BOUNDARY_LOSS_WEIGHT}"
    --omni_lfq_boundary_topk "${BOUNDARY_TOPK}"
    --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}"
    --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}"
    --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}"
)

run_final_calibration() {
    local label="$1"
    local physical_gpu="$2"
    local output_dir="$3"
    local checkpoint_dir="$4"
    shift 4

    if checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}" && [[ "${CALIB_OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping calibration."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${CALIB_OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        echo "Move the partial output aside or set CALIB_OVERWRITE=1." >&2
        return 4
    fi

    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        "${COMMON_ARGS[@]}"
        --output_dir "${output_dir}"
        --calibration_only
        --layers all
        --calib_sample_size "${FINAL_CALIB_SAMPLES}"
        --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}"
        "$@"
    )
    if [[ "${CALIB_OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi

    echo "[${label}] physical_gpu=${physical_gpu} output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${physical_gpu}" "${command[@]}"
        return
    fi
    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${physical_gpu}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    if ! checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}"; then
        echo "[${label}] expected checkpoints 0-${FINAL_LAYER} were not produced." >&2
        return 5
    fi
}

run_evaluation() {
    local label="$1"
    local output_dir="$2"
    local checkpoint_dir="$3"
    shift 3

    if [[ -s "${output_dir}/eval_results.json" && "${EVAL_OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete eval_results.json found; skipping evaluation."
        return
    fi
    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}"; then
        echo "[${label}] incomplete checkpoint set: ${checkpoint_dir}" >&2
        return 5
    fi

    echo "[${label}] evaluation_gpus=${GPUS} output=${output_dir}"
    TASK=ad \
    GPUS="${GPUS}" \
    MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${output_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OVERWRITE="${EVAL_OVERWRITE}" \
    DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh \
        --mode omniquant \
        --weight_quant_format fp4_e2m1 \
        --activation_quant_format fp8_e4m3fn \
        --weight_quant_scheme symmetric \
        --weight_group_size 0 \
        --omni_lwc \
        --omni_let_mode none \
        --omni_load_checkpoint_dir "${checkpoint_dir}" \
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}" \
        "$@"
}

write_summary() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[summary] skipped in DRY_RUN mode."
        return
    fi
    if [[ "${EVAL_SAMPLE_SIZE}" != "full" ]]; then
        echo "[summary] skipped because reference results are AD-full."
        return
    fi

    local g128_rtn="${G128_W4_ROOT}/rtn_ad_full/eval_results.json"
    local g128_mse="${G128_W4_ROOT}/omniquant_lwc_mse_ad_full/eval_results.json"
    local g128_boundary="${G128_W4_ROOT}/omniquant_lwc_abc_boundary_w0.3_ad_full/eval_results.json"
    local pc_mse="${MSE_EVAL_DIR}/eval_results.json"
    local pc_boundary="${BOUNDARY_EVAL_DIR}/eval_results.json"
    local required
    for required in "${PC_RTN_RESULT}" "${pc_mse}" "${pc_boundary}" "${g128_rtn}" "${g128_mse}" "${g128_boundary}"; do
        if [[ ! -s "${required}" ]]; then
            echo "[summary] missing result; comparison not written: ${required}" >&2
            return
        fi
    done

    mkdir -p "$(dirname "${SUMMARY_PATH}")"
    "${PYTHON_BIN}" - \
        "${SUMMARY_PATH}" "${MODEL_NAME}" \
        pc_rtn "${PC_RTN_RESULT}" \
        pc_omniquant_lwc_mse "${pc_mse}" \
        pc_abc_boundary_w0.3 "${pc_boundary}" \
        g128_rtn "${g128_rtn}" \
        g128_omniquant_lwc_mse "${g128_mse}" \
        g128_abc_boundary_w0.3 "${g128_boundary}" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
model_name = sys.argv[2]
pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))
methods = {}
for label, path_text in pairs:
    path = Path(path_text)
    metrics = json.loads(path.read_text(encoding="utf-8"))[model_name]["ad"]["test"]
    methods[label] = {"eval_results_path": str(path.resolve()), "metrics": metrics}

payload = {
    "protocol": {
        "model": model_name,
        "task": "ad",
        "weight_quant_format": "fp4_e2m1",
        "activation_quant_format": "fp8_e4m3fn",
        "prefix_calib_sample_size": 128,
        "final_calib_sample_size": 1024,
        "boundary_loss_weight": 0.3,
    },
    "methods": methods,
}
temporary = summary_path.with_name(f".{summary_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
os.replace(temporary, summary_path)

metric_names = ("pass@1", "pass@32", "recall@1", "recall@32", "pid_pass@1", "pid_pass@32", "pid_recall@1", "pid_recall@32")
print("[comparison] method                       " + " ".join(f"{name:>13}" for name in metric_names))
for label, _ in pairs:
    metrics = methods[label]["metrics"]
    cells = ["          n/a" if metrics.get(name) is None else f"{100.0 * float(metrics[name]):>12.4f}%" for name in metric_names]
    print(f"[comparison] {label:<28} " + " ".join(cells))
print(f"[comparison] output={summary_path}")
PY
}

echo "[protocol] model=${MODEL_PATH} task=ad seed=42"
echo "[protocol] FP4-E2M1-W/FP8-E4M3-A symmetric, weight_group_size=0 (per-output-channel)"
echo "[protocol] shared prefix layers=0-${PREFIX_LAST_LAYER} calib=[0,${PREFIX_CALIB_SAMPLES}) source=${PREFIX_CHECKPOINT_DIR}"
echo "[protocol] final layer=${FINAL_LAYER} calib=[0,${FINAL_CALIB_SAMPLES}) epochs=${EPOCHS} best_epoch=disabled"
echo "[protocol] final calibration GPUs: MSE=${MSE_GPU}, ABC+boundary=${BOUNDARY_GPU}"
echo "[protocol] full evaluation GPUs=${GPUS}; results_root=${RESULTS_ROOT}"

echo "[1/4] Launching the two independent final-layer calibrations."
run_final_calibration \
    "per-channel MSE-LWC" "${MSE_GPU}" \
    "${MSE_CALIB_DIR}" "${MSE_CHECKPOINT_DIR}" \
    --omni_final_objective mse &
mse_pid=$!
run_final_calibration \
    "per-channel ABC+boundary" "${BOUNDARY_GPU}" \
    "${BOUNDARY_CALIB_DIR}" "${BOUNDARY_CHECKPOINT_DIR}" \
    "${LFQ_ARGS[@]}" &
boundary_pid=$!

calibration_failed=0
if ! wait "${mse_pid}"; then
    echo "[calibration] MSE-LWC arm failed." >&2
    calibration_failed=1
fi
if ! wait "${boundary_pid}"; then
    echo "[calibration] ABC+boundary arm failed." >&2
    calibration_failed=1
fi
if (( calibration_failed != 0 )); then
    exit 5
fi

echo "[2/4] Evaluating per-channel MSE-LWC on AD-${EVAL_SAMPLE_SIZE}."
run_evaluation \
    "per-channel MSE-LWC" "${MSE_EVAL_DIR}" "${MSE_CHECKPOINT_DIR}" \
    --omni_final_objective mse

echo "[3/4] Evaluating per-channel ABC+0.3 boundary on AD-${EVAL_SAMPLE_SIZE}."
run_evaluation \
    "per-channel ABC+boundary" "${BOUNDARY_EVAL_DIR}" "${BOUNDARY_CHECKPOINT_DIR}" \
    "${LFQ_ARGS[@]}"

echo "[4/4] Writing the W4A8 per-channel versus g128 comparison."
write_summary

echo "[done] per-channel MSE-LWC:       ${MSE_EVAL_DIR}/eval_results.json"
echo "[done] per-channel ABC+boundary:  ${BOUNDARY_EVAL_DIR}/eval_results.json"
echo "[done] comparison summary:        ${SUMMARY_PATH}"
