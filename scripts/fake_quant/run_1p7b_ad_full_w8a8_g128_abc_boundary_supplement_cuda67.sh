#!/usr/bin/env bash
set -euo pipefail

# Supplement the completed W8A8-g128 core matrix with two final-layer arms:
#   1. ABC CE only (boundary weight 0.0).
#   2. ABC CE + 0.5 boundary.
#
# Both arms reuse the existing deployment-matched Layer 0--26 MSE/LWC prefix
# trained on calibration records [0, 128). They independently initialize and
# optimize Layer 27 on the same 1,024 records with seed 42. The two calibration
# jobs run concurrently on the first two GPUs; AD-full evaluations then run
# serially, with each evaluation sharded over every GPU in GPUS.
#
# Usage:
#   bash scripts/fake_quant/run_1p7b_ad_full_w8a8_g128_abc_boundary_supplement_cuda67.sh
#
# Useful overrides:
#   DRY_RUN=1 bash scripts/fake_quant/run_1p7b_ad_full_w8a8_g128_abc_boundary_supplement_cuda67.sh
#   GPUS=4,5 bash scripts/fake_quant/run_1p7b_ad_full_w8a8_g128_abc_boundary_supplement_cuda67.sh

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
REFERENCE_BOUNDARY_LOSS_WEIGHT="${REFERENCE_BOUNDARY_LOSS_WEIGHT:-0.3}"
EXTRA_BOUNDARY_LOSS_WEIGHT="${EXTRA_BOUNDARY_LOSS_WEIGHT:-0.5}"
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
W8_ROOT="${RESULTS_ROOT}/w8a8_g128"

PREFIX_OUTPUT_DIR="${W8_PREFIX_OUTPUT_DIR:-${W8_ROOT}/shared_mse_lwc_prefix_calib${PREFIX_CALIB_SAMPLES}}"
PREFIX_CHECKPOINT_DIR="${PREFIX_OUTPUT_DIR}/${MODEL_NAME}/ad/omniquant_calibration"

RTN_EVAL_DIR="${W8_ROOT}/rtn_ad_${EVAL_SAMPLE_SIZE}"
MSE_EVAL_DIR="${W8_ROOT}/omniquant_lwc_mse_ad_${EVAL_SAMPLE_SIZE}"
ABC_ONLY_CALIB_DIR="${W8_ROOT}/omniquant_lwc_abc_only_final${FINAL_CALIB_SAMPLES}_calibration"
ABC_ONLY_CHECKPOINT_DIR="${ABC_ONLY_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
ABC_ONLY_EVAL_DIR="${W8_ROOT}/omniquant_lwc_abc_only_ad_${EVAL_SAMPLE_SIZE}"
REFERENCE_BOUNDARY_EVAL_DIR="${W8_ROOT}/omniquant_lwc_abc_boundary_w${REFERENCE_BOUNDARY_LOSS_WEIGHT}_ad_${EVAL_SAMPLE_SIZE}"
EXTRA_BOUNDARY_CALIB_DIR="${W8_ROOT}/omniquant_lwc_abc_boundary_w${EXTRA_BOUNDARY_LOSS_WEIGHT}_final${FINAL_CALIB_SAMPLES}_calibration"
EXTRA_BOUNDARY_CHECKPOINT_DIR="${EXTRA_BOUNDARY_CALIB_DIR}/${MODEL_NAME}/ad/omniquant_calibration"
EXTRA_BOUNDARY_EVAL_DIR="${W8_ROOT}/omniquant_lwc_abc_boundary_w${EXTRA_BOUNDARY_LOSS_WEIGHT}_ad_${EVAL_SAMPLE_SIZE}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_ROOT}/w8a8_g128_abc_boundary_sweep_summary.json}"

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}

validate_positive_float() {
    local name="$1"
    local value="$2"
    if ! "${PYTHON_BIN}" -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and x > 0.0))' "${value}"; then
        echo "${name} must be finite and positive; got ${value}." >&2
        exit 2
    fi
}

validate_boolean OVERWRITE "${OVERWRITE}"
validate_boolean CALIB_OVERWRITE "${CALIB_OVERWRITE}"
validate_boolean EVAL_OVERWRITE "${EVAL_OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"
validate_positive_float REFERENCE_BOUNDARY_LOSS_WEIGHT "${REFERENCE_BOUNDARY_LOSS_WEIGHT}"
validate_positive_float EXTRA_BOUNDARY_LOSS_WEIGHT "${EXTRA_BOUNDARY_LOSS_WEIGHT}"

if [[ "${WEIGHT_GROUP_SIZE}" != "128" ]]; then
    echo "This controlled launcher requires WEIGHT_GROUP_SIZE=128; got ${WEIGHT_GROUP_SIZE}." >&2
    exit 2
fi
if (( PREFIX_CALIB_SAMPLES != 128 || FINAL_CALIB_SAMPLES != 1024 )); then
    echo "This controlled launcher requires PREFIX_CALIB_SAMPLES=128 and FINAL_CALIB_SAMPLES=1024." >&2
    echo "Got prefix=${PREFIX_CALIB_SAMPLES}, final=${FINAL_CALIB_SAMPLES}." >&2
    exit 2
fi
if (( NUM_LAYERS < 2 )); then
    echo "NUM_LAYERS must be at least 2." >&2
    exit 2
fi
if [[ "${REFERENCE_BOUNDARY_LOSS_WEIGHT}" != "0.3" ]]; then
    echo "The retained reference arm must use REFERENCE_BOUNDARY_LOSS_WEIGHT=0.3." >&2
    exit 2
fi
if [[ "${EXTRA_BOUNDARY_LOSS_WEIGHT}" != "0.5" ]]; then
    echo "This supplement requires EXTRA_BOUNDARY_LOSS_WEIGHT=0.5." >&2
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
ABC_ONLY_GPU="${GPU_IDS[0]}"
EXTRA_BOUNDARY_GPU="${GPU_IDS[1]}"

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

run_final_calibration() {
    local label="$1"
    local physical_gpu="$2"
    local output_dir="$3"
    local checkpoint_dir="$4"
    local boundary_weight="$5"

    if checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}" && [[ "${CALIB_OVERWRITE}" != "1" ]]; then
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
        --weight_quant_format fp8_e4m3fn
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
        --layers all
        --calib_sample_size "${FINAL_CALIB_SAMPLES}"
        --omni_prefix_checkpoint_dir "${PREFIX_CHECKPOINT_DIR}"
        --omni_final_objective lfq_ce
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

run_abc_evaluation() {
    local label="$1"
    local output_dir="$2"
    local checkpoint_dir="$3"
    local boundary_weight="$4"

    if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${checkpoint_dir}" 0 "${FINAL_LAYER}"; then
        echo "[${label}] incomplete checkpoint set: ${checkpoint_dir}" >&2
        return 4
    fi

    run_sharded_stage \
        "${label}" \
        "${output_dir}" \
        --mode omniquant \
        --weight_quant_format fp8_e4m3fn \
        --activation_quant_format fp8_e4m3fn \
        --weight_quant_scheme symmetric \
        --weight_group_size "${WEIGHT_GROUP_SIZE}" \
        --omni_lwc \
        --omni_let_mode none \
        --omni_final_objective lfq_ce \
        --omni_load_checkpoint_dir "${checkpoint_dir}" \
        --calib_sample_size "${PREFIX_CALIB_SAMPLES}" \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}" \
        --omni_lfq_token_scope sid_slots \
        --omni_lfq_vocab_scope s_abc \
        --omni_lfq_slot_weights 1 1 1 \
        --omni_lfq_loss_weight 1.0 \
        --omni_lfq_boundary_loss_weight "${boundary_weight}" \
        --omni_lfq_boundary_topk "${BOUNDARY_TOPK}" \
        --omni_lfq_boundary_negative_count "${BOUNDARY_NEGATIVES}" \
        --omni_lfq_boundary_tie_threshold "${BOUNDARY_TIE_THRESHOLD}" \
        --omni_lfq_boundary_gap_scale "${BOUNDARY_GAP_SCALE}"
}

require_existing_result() {
    local label="$1"
    local path="$2"
    if [[ ! -s "${path}" ]]; then
        echo "[summary] missing retained ${label} result: ${path}" >&2
        return 4
    fi
}

write_summary() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[summary] skipped in DRY_RUN mode."
        return
    fi

    require_existing_result "RTN" "${RTN_EVAL_DIR}/eval_results.json"
    require_existing_result "MSE-LWC" "${MSE_EVAL_DIR}/eval_results.json"
    require_existing_result "ABC-only" "${ABC_ONLY_EVAL_DIR}/eval_results.json"
    require_existing_result "ABC+0.3 boundary" "${REFERENCE_BOUNDARY_EVAL_DIR}/eval_results.json"
    require_existing_result "ABC+0.5 boundary" "${EXTRA_BOUNDARY_EVAL_DIR}/eval_results.json"

    mkdir -p "$(dirname "${SUMMARY_PATH}")"
    "${PYTHON_BIN}" - \
        "${SUMMARY_PATH}" \
        "${MODEL_NAME}" \
        w8a8_rtn_g128 "${RTN_EVAL_DIR}/eval_results.json" \
        w8a8_omniquant_lwc_mse_g128 "${MSE_EVAL_DIR}/eval_results.json" \
        w8a8_abc_only_g128 "${ABC_ONLY_EVAL_DIR}/eval_results.json" \
        w8a8_abc_boundary_w0.3_g128 "${REFERENCE_BOUNDARY_EVAL_DIR}/eval_results.json" \
        w8a8_abc_boundary_w0.5_g128 "${EXTRA_BOUNDARY_EVAL_DIR}/eval_results.json" <<'PY'
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
    pass_values = [float(metrics[f"pass@{k}"]) for k in (1, 4, 8, 16, 32)]
    methods[label] = {
        "eval_results_path": str(path.resolve()),
        "metrics": metrics,
        "first_hit_rank_buckets": {
            "rank_1": pass_values[0],
            "rank_2_4": pass_values[1] - pass_values[0],
            "rank_5_8": pass_values[2] - pass_values[1],
            "rank_9_16": pass_values[3] - pass_values[2],
            "rank_17_32": pass_values[4] - pass_values[3],
        },
    }

summary = {
    "protocol": {
        "model": model_name,
        "task": "ad",
        "weight_quant_format": "fp8_e4m3fn",
        "activation_quant_format": "fp8_e4m3fn",
        "weight_group_size": 128,
        "prefix_calib_sample_size": int(os.environ.get("PREFIX_CALIB_SAMPLES", "128")),
        "final_calib_sample_size": int(os.environ.get("FINAL_CALIB_SAMPLES", "1024")),
        "abc_ce_loss_weight": 1.0,
        "boundary_loss_weights": [0.0, 0.3, 0.5],
        "boundary_topk": int(os.environ.get("BOUNDARY_TOPK", "32")),
        "epochs": int(os.environ.get("EPOCHS", "20")),
        "seed": 42,
        "best_epoch": "disabled",
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
    "pass@4",
    "pass@8",
    "pass@16",
    "pass@32",
    "recall@32",
    "pid_pass@32",
    "pid_recall@32",
)
print("[W8 sweep summary] method                              " + " ".join(f"{name:>13}" for name in metric_names))
for label, _path in pairs:
    metrics = methods[label]["metrics"]
    cells = []
    for name in metric_names:
        value = metrics.get(name)
        cells.append("          n/a" if value is None else f"{100.0 * float(value):>12.4f}%")
    print(f"[W8 sweep summary] {label:<35} " + " ".join(cells))
print(f"[W8 sweep summary] output={summary_path}")
PY
}

echo "[protocol] model=${MODEL_PATH} task=ad seed=42"
echo "[protocol] supplement=W8A8-g128 ABC-only and ABC+0.5-boundary"
echo "[protocol] retained_reference=ABC+${REFERENCE_BOUNDARY_LOSS_WEIGHT}-boundary"
echo "[protocol] weight_group_size=${WEIGHT_GROUP_SIZE} activation=FP8-E4M3 symmetric shared-input QDQ"
echo "[protocol] OmniQuant=LWC-only LET=off epochs=${EPOCHS} best_epoch=disabled"
echo "[protocol] prefix layers=0-${PREFIX_LAST_LAYER} calib=[0,${PREFIX_CALIB_SAMPLES})"
echo "[protocol] final layer=${FINAL_LAYER} calib=[0,${FINAL_CALIB_SAMPLES})"
echo "[protocol] calibration_gpus=${ABC_ONLY_GPU},${EXTRA_BOUNDARY_GPU}; evaluation_gpus=${GPUS}"
echo "[protocol] prefix_checkpoint=${PREFIX_CHECKPOINT_DIR}"
echo "[protocol] results_root=${RESULTS_ROOT}"

if [[ "${DRY_RUN}" != "1" ]] && ! checkpoint_range_complete "${PREFIX_CHECKPOINT_DIR}" 0 "${PREFIX_LAST_LAYER}"; then
    echo "The completed W8A8-g128 prefix is missing or incomplete: ${PREFIX_CHECKPOINT_DIR}" >&2
    echo "Run the core6 launcher first, or set W8_PREFIX_OUTPUT_DIR to a complete prefix." >&2
    exit 4
fi

echo
echo "[calibration 1/2] ABC-only on physical GPU ${ABC_ONLY_GPU}."
run_final_calibration \
    "W8A8-g128 ABC-only" \
    "${ABC_ONLY_GPU}" \
    "${ABC_ONLY_CALIB_DIR}" \
    "${ABC_ONLY_CHECKPOINT_DIR}" \
    0.0 &
abc_only_pid=$!

echo "[calibration 2/2] ABC+0.5 boundary on physical GPU ${EXTRA_BOUNDARY_GPU}."
run_final_calibration \
    "W8A8-g128 ABC+0.5-boundary" \
    "${EXTRA_BOUNDARY_GPU}" \
    "${EXTRA_BOUNDARY_CALIB_DIR}" \
    "${EXTRA_BOUNDARY_CHECKPOINT_DIR}" \
    "${EXTRA_BOUNDARY_LOSS_WEIGHT}" &
extra_boundary_pid=$!

calibration_failed=0
if ! wait "${abc_only_pid}"; then
    echo "[W8A8-g128] ABC-only final-layer calibration failed." >&2
    calibration_failed=1
fi
if ! wait "${extra_boundary_pid}"; then
    echo "[W8A8-g128] ABC+0.5 final-layer calibration failed." >&2
    calibration_failed=1
fi
if (( calibration_failed != 0 )); then
    exit 5
fi

echo
echo "[evaluation 1/2] ABC-only AD-${EVAL_SAMPLE_SIZE}."
run_abc_evaluation \
    "W8A8-g128 ABC-only" \
    "${ABC_ONLY_EVAL_DIR}" \
    "${ABC_ONLY_CHECKPOINT_DIR}" \
    0.0

echo
echo "[evaluation 2/2] ABC+0.5 boundary AD-${EVAL_SAMPLE_SIZE}."
run_abc_evaluation \
    "W8A8-g128 ABC+0.5-boundary" \
    "${EXTRA_BOUNDARY_EVAL_DIR}" \
    "${EXTRA_BOUNDARY_CHECKPOINT_DIR}" \
    "${EXTRA_BOUNDARY_LOSS_WEIGHT}"

export PREFIX_CALIB_SAMPLES
export FINAL_CALIB_SAMPLES
export BOUNDARY_TOPK
export EPOCHS
write_summary

echo "[done] W8A8-g128 ABC-only and ABC+0.5-boundary AD-${EVAL_SAMPLE_SIZE} completed."
echo "[done] ABC-only result=${ABC_ONLY_EVAL_DIR}/eval_results.json"
echo "[done] ABC+0.5 result=${EXTRA_BOUNDARY_EVAL_DIR}/eval_results.json"
echo "[done] sweep summary=${SUMMARY_PATH}"
