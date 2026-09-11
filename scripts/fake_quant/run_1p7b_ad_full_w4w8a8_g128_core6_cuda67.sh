#!/usr/bin/env bash
set -euo pipefail

# Run the 1.7B AD-full g128 core matrix:
#   FP4-W/FP8-A: RTN, OmniQuant-LWC MSE, OmniQuant-LWC ABC+0.3 boundary.
#   FP8-W/FP8-A: RTN, OmniQuant-LWC MSE, OmniQuant-LWC ABC+0.3 boundary.
#
# Both learned arms of one precision share a deployment-matched MSE/LWC prefix
# trained on calibration records [0, 128). Layer 27 is independently optimized
# on all 1024 calibration records. RTN has no learned calibration. Full AD
# evaluations are serial and round-robin sharded across all GPUs. The two
# final-layer calibrations of one precision run concurrently on the first two
# GPUs after their shared prefix is ready.
#
# The completed W4 g128 Layer 0--26 prefix is reused by default.
#
# Usage:
#   bash scripts/fake_quant/run_1p7b_ad_full_w4w8a8_g128_core6_cuda67.sh
#
# Useful overrides:
#   DRY_RUN=1 bash scripts/fake_quant/run_1p7b_ad_full_w4w8a8_g128_core6_cuda67.sh
#   GPUS=4,5 bash scripts/fake_quant/run_1p7b_ad_full_w4w8a8_g128_core6_cuda67.sh
#   EVAL_SAMPLE_SIZE=3000 bash scripts/fake_quant/run_1p7b_ad_full_w4w8a8_g128_core6_cuda67.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
DEFAULT_PYTHON_BIN="/home/guowei/miniconda3/envs/benchmark/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
    DEFAULT_PYTHON_BIN="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
GPUS="${GPUS:-6,7}"

WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-128}"
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
CALIB_OVERWRITE="${CALIB_OVERWRITE:-${OVERWRITE}}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-${OVERWRITE}}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_full_w4w8a8_g128_core6_prefix${PREFIX_CALIB_SAMPLES}_final${FINAL_CALIB_SAMPLES}}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
FINAL_LAYER=$((NUM_LAYERS - 1))
PREFIX_LAST_LAYER=$((FINAL_LAYER - 1))

# Reuse the already completed W4 g128 prefix unless explicitly redirected.
W4_PREFIX_OUTPUT_DIR="${W4_PREFIX_OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad_fp4w_fp8a_lwc_g128_abc_boundary_prefix128_final512_heldout512/shared_mse_lwc_prefix_calib128}"
W8_PREFIX_OUTPUT_DIR="${W8_PREFIX_OUTPUT_DIR:-${RESULTS_ROOT}/w8a8_g128/shared_mse_lwc_prefix_calib${PREFIX_CALIB_SAMPLES}}"
W4_PREFIX_CHECKPOINT_DIR="${W4_PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
W8_PREFIX_CHECKPOINT_DIR="${W8_PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"

W4_ROOT="${RESULTS_ROOT}/w4a8_g128"
W8_ROOT="${RESULTS_ROOT}/w8a8_g128"

W4_RTN_EVAL_DIR="${W4_ROOT}/rtn_ad_${EVAL_SAMPLE_SIZE}"
W4_MSE_CALIB_DIR="${W4_ROOT}/omniquant_lwc_mse_final${FINAL_CALIB_SAMPLES}_calibration"
W4_MSE_CHECKPOINT_DIR="${W4_MSE_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
W4_MSE_EVAL_DIR="${W4_ROOT}/omniquant_lwc_mse_ad_${EVAL_SAMPLE_SIZE}"
W4_BOUNDARY_CALIB_DIR="${W4_ROOT}/omniquant_lwc_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}_final${FINAL_CALIB_SAMPLES}_calibration"
W4_BOUNDARY_CHECKPOINT_DIR="${W4_BOUNDARY_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
W4_BOUNDARY_EVAL_DIR="${W4_ROOT}/omniquant_lwc_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}_ad_${EVAL_SAMPLE_SIZE}"

W8_RTN_EVAL_DIR="${W8_ROOT}/rtn_ad_${EVAL_SAMPLE_SIZE}"
W8_MSE_CALIB_DIR="${W8_ROOT}/omniquant_lwc_mse_final${FINAL_CALIB_SAMPLES}_calibration"
W8_MSE_CHECKPOINT_DIR="${W8_MSE_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
W8_MSE_EVAL_DIR="${W8_ROOT}/omniquant_lwc_mse_ad_${EVAL_SAMPLE_SIZE}"
W8_BOUNDARY_CALIB_DIR="${W8_ROOT}/omniquant_lwc_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}_final${FINAL_CALIB_SAMPLES}_calibration"
W8_BOUNDARY_CHECKPOINT_DIR="${W8_BOUNDARY_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
W8_BOUNDARY_EVAL_DIR="${W8_ROOT}/omniquant_lwc_abc_boundary_w${BOUNDARY_LOSS_WEIGHT}_ad_${EVAL_SAMPLE_SIZE}"

SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_ROOT}/core6_eval_summary.json}"

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}

validate_boolean OVERWRITE "${OVERWRITE}"
validate_boolean CALIB_OVERWRITE "${CALIB_OVERWRITE}"
validate_boolean EVAL_OVERWRITE "${EVAL_OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"

if [[ "${WEIGHT_GROUP_SIZE}" != "128" ]]; then
    echo "This controlled launcher requires WEIGHT_GROUP_SIZE=128; got ${WEIGHT_GROUP_SIZE}." >&2
    exit 2
fi
if (( PREFIX_CALIB_SAMPLES != 128 || FINAL_CALIB_SAMPLES != 1024 )); then
    echo "This controlled launcher requires PREFIX_CALIB_SAMPLES=128 and FINAL_CALIB_SAMPLES=1024." >&2
    echo "Got prefix=${PREFIX_CALIB_SAMPLES}, final=${FINAL_CALIB_SAMPLES}." >&2
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
if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" -c 'import torch, transformers, pyfiglet'
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
AUX_CALIB_GPU="${GPU_IDS[1]}"

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

run_calibration_stage() {
    local label="$1"
    local physical_gpu="$2"
    local weight_format="$3"
    local output_dir="$4"
    local checkpoint_dir="$5"
    local first_layer="$6"
    local last_layer="$7"
    shift 7

    if checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}" && [[ "${CALIB_OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete checkpoints found; skipping calibration."
        return
    fi
    if checkpoint_dir_has_files "${checkpoint_dir}" && [[ "${CALIB_OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial checkpoints found at ${checkpoint_dir}." >&2
        echo "Move the partial stage aside, or rerun with CALIB_OVERWRITE=1." >&2
        return 3
    fi

    local command=(
        "${PYTHON_BIN}" -u -m fake_quant.run_m1_onerec_ad
        --task ad
        --mode omniquant
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --device cuda:0
        --output_dir "${output_dir}"
        --weight_quant_format "${weight_format}"
        --activation_quant_format fp8_e4m3fn
        --weight_quant_scheme symmetric
        --weight_group_size "${WEIGHT_GROUP_SIZE}"
        --omni_lwc
        --omni_let_mode none
        --omni_epochs "${EPOCHS}"
        --omni_epoch_eval_interval 0
        --omni_lwc_lr "${LWC_LR}"
        --omni_init_lwc_logit "${INIT_LWC_LOGIT}"
        --calibration_only
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
    if ! checkpoint_range_complete "${checkpoint_dir}" "${first_layer}" "${last_layer}"; then
        echo "[${label}] expected checkpoints ${first_layer}-${last_layer} were not produced." >&2
        return 4
    fi
}

run_sharded_stage() {
    local label="$1"
    local output_dir="$2"
    shift 2

    if [[ -s "${output_dir}/eval_results.json" && "${EVAL_OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete eval_results.json found; skipping evaluation."
        return
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
        bash scripts/fake_quant/run_sharded_eval_cuda.sh "$@"
}

run_rtn_evaluation() {
    local label="$1"
    local weight_format="$2"
    local output_dir="$3"
    run_sharded_stage \
        "${label}" \
        "${output_dir}" \
        --mode baseline_qdq \
        --weight_quant_format "${weight_format}" \
        --activation_quant_format fp8_e4m3fn \
        --weight_quant_scheme symmetric \
        --weight_group_size "${WEIGHT_GROUP_SIZE}" \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}"
}

run_omni_evaluation() {
    local label="$1"
    local weight_format="$2"
    local output_dir="$3"
    local checkpoint_dir="$4"
    local objective="$5"
    local boundary_weight="$6"

    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}"; then
        echo "[${label}] incomplete checkpoint set: ${checkpoint_dir}" >&2
        return 4
    fi

    local args=(
        --mode omniquant
        --weight_quant_format "${weight_format}"
        --activation_quant_format fp8_e4m3fn
        --weight_quant_scheme symmetric
        --weight_group_size "${WEIGHT_GROUP_SIZE}"
        --omni_lwc
        --omni_let_mode none
        --omni_final_objective "${objective}"
        --omni_load_checkpoint_dir "${checkpoint_dir}"
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}"
        --eval_sample_size "${EVAL_SAMPLE_SIZE}"
    )
    if [[ "${objective}" == "lfq_ce" ]]; then
        args+=(
            --omni_lfq_token_scope sid_slots
            --omni_lfq_vocab_scope s_abc
            --omni_lfq_slot_weights 1 1 1
            --omni_lfq_loss_weight 1.0
            --omni_lfq_boundary_loss_weight "${boundary_weight}"
            --omni_lfq_boundary_topk "${BOUNDARY_TOPK}"
            --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}"
            --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}"
            --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}"
        )
    fi
    run_sharded_stage "${label}" "${output_dir}" "${args[@]}"
}

run_precision_matrix() {
    local precision_label="$1"
    local weight_format="$2"
    local prefix_output_dir="$3"
    local prefix_checkpoint_dir="$4"
    local rtn_eval_dir="$5"
    local mse_calib_dir="$6"
    local mse_checkpoint_dir="$7"
    local mse_eval_dir="$8"
    local boundary_calib_dir="$9"
    local boundary_checkpoint_dir="${10}"
    local boundary_eval_dir="${11}"

    echo
    echo "[${precision_label}] Experiment 1/3: RTN-g128 AD-${EVAL_SAMPLE_SIZE}."
    run_rtn_evaluation \
        "${precision_label} RTN-g128" \
        "${weight_format}" \
        "${rtn_eval_dir}"

    echo
    echo "[${precision_label}] Support stage: shared Layer 0-${PREFIX_LAST_LAYER} MSE/LWC prefix."
    run_calibration_stage \
        "${precision_label} shared-prefix" \
        "${CALIB_GPU}" \
        "${weight_format}" \
        "${prefix_output_dir}" \
        "${prefix_checkpoint_dir}" \
        0 "${PREFIX_LAST_LAYER}" \
        --layers "0-${PREFIX_LAST_LAYER}" \
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
        --omni_final_objective mse

    echo
    echo "[${precision_label}] Experiments 2/3 and 3/3: parallel final-layer calibration."
    run_calibration_stage \
        "${precision_label} OmniQuant-LWC-MSE" \
        "${CALIB_GPU}" \
        "${weight_format}" \
        "${mse_calib_dir}" \
        "${mse_checkpoint_dir}" \
        0 "${FINAL_LAYER}" \
        --layers all \
        --calib_sample_size "${FINAL_CALIB_SAMPLES}" \
        --omni_prefix_checkpoint_dir "${prefix_checkpoint_dir}" \
        --omni_final_objective mse &
    local mse_pid=$!

    run_calibration_stage \
        "${precision_label} ABC+boundary" \
        "${AUX_CALIB_GPU}" \
        "${weight_format}" \
        "${boundary_calib_dir}" \
        "${boundary_checkpoint_dir}" \
        0 "${FINAL_LAYER}" \
        --layers all \
        --calib_sample_size "${FINAL_CALIB_SAMPLES}" \
        --omni_prefix_checkpoint_dir "${prefix_checkpoint_dir}" \
        --omni_final_objective lfq_ce \
        --omni_lfq_token_scope sid_slots \
        --omni_lfq_vocab_scope s_abc \
        --omni_lfq_slot_weights 1 1 1 \
        --omni_lfq_loss_weight 1.0 \
        --omni_lfq_boundary_loss_weight "${BOUNDARY_LOSS_WEIGHT}" \
        --omni_lfq_boundary_topk "${BOUNDARY_TOPK}" \
        --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}" \
        --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}" \
        --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}" &
    local boundary_pid=$!

    local calibration_failed=0
    if ! wait "${mse_pid}"; then
        echo "[${precision_label}] MSE final-layer calibration failed." >&2
        calibration_failed=1
    fi
    if ! wait "${boundary_pid}"; then
        echo "[${precision_label}] ABC+boundary final-layer calibration failed." >&2
        calibration_failed=1
    fi
    if (( calibration_failed != 0 )); then
        return 5
    fi

    echo
    echo "[${precision_label}] Experiment 2/3: OmniQuant-LWC-MSE AD-${EVAL_SAMPLE_SIZE}."
    run_omni_evaluation \
        "${precision_label} OmniQuant-LWC-MSE" \
        "${weight_format}" \
        "${mse_eval_dir}" \
        "${mse_checkpoint_dir}" \
        mse \
        0.0

    echo
    echo "[${precision_label}] Experiment 3/3: ABC+boundary AD-${EVAL_SAMPLE_SIZE}."
    run_omni_evaluation \
        "${precision_label} ABC+boundary" \
        "${weight_format}" \
        "${boundary_eval_dir}" \
        "${boundary_checkpoint_dir}" \
        lfq_ce \
        "${BOUNDARY_LOSS_WEIGHT}"
}

print_summary() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[summary] skipped in DRY_RUN mode."
        return
    fi

    mkdir -p "$(dirname "${SUMMARY_PATH}")"
    "${PYTHON_BIN}" - \
        "${SUMMARY_PATH}" \
        "${MODEL_NAME}" \
        w4a8_rtn_g128 "${W4_RTN_EVAL_DIR}/eval_results.json" \
        w4a8_omniquant_lwc_mse_g128 "${W4_MSE_EVAL_DIR}/eval_results.json" \
        w4a8_abc_boundary_g128 "${W4_BOUNDARY_EVAL_DIR}/eval_results.json" \
        w8a8_rtn_g128 "${W8_RTN_EVAL_DIR}/eval_results.json" \
        w8a8_omniquant_lwc_mse_g128 "${W8_MSE_EVAL_DIR}/eval_results.json" \
        w8a8_abc_boundary_g128 "${W8_BOUNDARY_EVAL_DIR}/eval_results.json" <<'PY'
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload[model_name]["ad"]["test"]
    methods[label] = {
        "eval_results_path": str(path.resolve()),
        "metrics": metrics,
    }

summary = {
    "protocol": {
        "model": model_name,
        "task": "ad",
        "weight_group_size": 128,
        "activation_quant_format": "fp8_e4m3fn",
        "prefix_calib_sample_size": int(os.environ.get("PREFIX_CALIB_SAMPLES", "128")),
        "final_calib_sample_size": int(os.environ.get("FINAL_CALIB_SAMPLES", "1024")),
        "boundary_loss_weight": float(os.environ.get("BOUNDARY_LOSS_WEIGHT", "0.3")),
    },
    "methods": methods,
}
temporary_path = summary_path.with_name(f".{summary_path.name}.tmp")
temporary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
os.replace(temporary_path, summary_path)

metric_names = (
    "pass@1",
    "pass@32",
    "recall@1",
    "recall@32",
    "pid_pass@1",
    "pid_pass@32",
    "pid_recall@1",
    "pid_recall@32",
)
print("[core6 summary] method                              " + " ".join(f"{name:>13}" for name in metric_names))
for label, _path in pairs:
    metrics = methods[label]["metrics"]
    cells = []
    for name in metric_names:
        value = metrics.get(name)
        cells.append("          n/a" if value is None else f"{100.0 * float(value):>12.4f}%")
    print(f"[core6 summary] {label:<35} " + " ".join(cells))
print(f"[core6 summary] output={summary_path}")
PY
}

echo "[protocol] model=${MODEL_PATH} task=ad seed=42"
echo "[protocol] matrix=W4A8/W8A8 x RTN/MSE-LWC/ABC+0.3-boundary"
echo "[protocol] weight_group_size=${WEIGHT_GROUP_SIZE} activation=FP8-E4M3 symmetric shared-input QDQ"
echo "[protocol] OmniQuant=LWC-only LET=off epochs=${EPOCHS} best_epoch=disabled"
echo "[protocol] prefix layers=0-${PREFIX_LAST_LAYER} calib=[0,${PREFIX_CALIB_SAMPLES})"
echo "[protocol] final layer=${FINAL_LAYER} calib=[0,${FINAL_CALIB_SAMPLES})"
echo "[protocol] calibration_gpus=${CALIB_GPU},${AUX_CALIB_GPU}; evaluation_gpus=${GPUS}"
echo "[protocol] results_root=${RESULTS_ROOT}"
echo "[protocol] W4 prefix source=${W4_PREFIX_CHECKPOINT_DIR}"

run_precision_matrix \
    W4A8 \
    fp4_e2m1 \
    "${W4_PREFIX_OUTPUT_DIR}" \
    "${W4_PREFIX_CHECKPOINT_DIR}" \
    "${W4_RTN_EVAL_DIR}" \
    "${W4_MSE_CALIB_DIR}" \
    "${W4_MSE_CHECKPOINT_DIR}" \
    "${W4_MSE_EVAL_DIR}" \
    "${W4_BOUNDARY_CALIB_DIR}" \
    "${W4_BOUNDARY_CHECKPOINT_DIR}" \
    "${W4_BOUNDARY_EVAL_DIR}"

run_precision_matrix \
    W8A8 \
    fp8_e4m3fn \
    "${W8_PREFIX_OUTPUT_DIR}" \
    "${W8_PREFIX_CHECKPOINT_DIR}" \
    "${W8_RTN_EVAL_DIR}" \
    "${W8_MSE_CALIB_DIR}" \
    "${W8_MSE_CHECKPOINT_DIR}" \
    "${W8_MSE_EVAL_DIR}" \
    "${W8_BOUNDARY_CALIB_DIR}" \
    "${W8_BOUNDARY_CHECKPOINT_DIR}" \
    "${W8_BOUNDARY_EVAL_DIR}"

export PREFIX_CALIB_SAMPLES
export FINAL_CALIB_SAMPLES
export BOUNDARY_LOSS_WEIGHT
print_summary

echo "[done] six g128 AD-${EVAL_SAMPLE_SIZE} experiments completed."
echo "[done] summary=${SUMMARY_PATH}"
