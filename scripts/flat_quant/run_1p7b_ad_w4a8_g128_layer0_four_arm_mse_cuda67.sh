#!/usr/bin/env bash
set -euo pipefail

# Controlled layer-0 FlatQuant-core ablation:
#   1. RTN              : fixed identity transform, fixed clipping.
#   2. LWC-only         : fixed identity transform, learned clipping.
#   3. LT-only          : learned transform, fixed clipping.
#   4. LT + LWC         : learned transform and learned clipping.
#
# All arms use the same FlatQuant training harness, calibration order, optimizer
# family, cosine schedule, seed, and QDQ implementation. The final 64 of the
# loaded 128 records are diagnostic held-out samples: they never participate in
# backpropagation or best-epoch selection.

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
LAYER="${LAYER:-0}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-64}"
HELDOUT_SAMPLES="${HELDOUT_SAMPLES:-64}"
EPOCHS="${EPOCHS:-15}"
TRANSFORM_LR="${TRANSFORM_LR:-5e-3}"
LWC_LR="${LWC_LR:-5e-2}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-128}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/ablation/1p7b_ad_w4a8_g128_layer${LAYER}_calib${CALIB_SAMPLES}_train${TRAIN_SAMPLES}_heldout${HELDOUT_SAMPLES}_epoch${EPOCHS}}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_ROOT}/four_arm_mse_summary.json}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
}

validate_boolean OVERWRITE "${OVERWRITE}"
validate_boolean DRY_RUN "${DRY_RUN}"
if (( CALIB_SAMPLES <= 1 || TRAIN_SAMPLES <= 0 || HELDOUT_SAMPLES <= 0 )); then
    echo "Calibration, train, and held-out sizes must be positive." >&2
    exit 2
fi
if (( TRAIN_SAMPLES + HELDOUT_SAMPLES != CALIB_SAMPLES )); then
    echo "This controlled split requires TRAIN_SAMPLES + HELDOUT_SAMPLES = CALIB_SAMPLES." >&2
    exit 2
fi
if [[ "${WEIGHT_GROUP_SIZE}" != "128" ]]; then
    echo "This launcher requires WEIGHT_GROUP_SIZE=128." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "GPUS must contain at least two physical GPU IDs; got ${GPUS}." >&2
    exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: ${gpu_id}" >&2
        exit 2
    fi
done
GPU_A="${GPU_IDS[0]}"
GPU_B="${GPU_IDS[1]}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

print_command() {
    printf '  %q' "$@"
    printf '\n'
}

run_arm() {
    local label="$1"
    local slug="$2"
    local gpu="$3"
    local learn_transform="$4"
    local use_lwc="$5"
    local output_dir="${RESULTS_ROOT}/${slug}"
    local result_dir="${output_dir}/${MODEL_NAME}/ad"
    local config_path="${result_dir}/flatquant_core_config.json"
    local checkpoint_path="${result_dir}/flatquant_calibration/layer_$(printf '%02d' "${LAYER}").pt"

    if [[ -s "${config_path}" && -s "${checkpoint_path}" && "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete result found; skipping."
        return
    fi
    if [[ ( -e "${config_path}" || -e "${checkpoint_path}" ) && "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] partial result found under ${output_dir}." >&2
        echo "Move it aside or rerun with OVERWRITE=1." >&2
        return 4
    fi

    local transform_flag="--flat_learn_transform"
    local lwc_flag="--omni_lwc"
    if [[ "${learn_transform}" != "1" ]]; then
        transform_flag="--no-flat_learn_transform"
    fi
    if [[ "${use_lwc}" != "1" ]]; then
        lwc_flag="--no-omni_lwc"
    fi

    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        --task ad
        --mode flatquant_core
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --output_dir "${output_dir}"
        --device cuda:0
        --layers "${LAYER}"
        --calib_sample_size "${CALIB_SAMPLES}"
        --weight_quant_format fp4_e2m1
        --activation_quant_format fp8_e4m3fn
        --weight_quant_scheme symmetric
        --weight_group_size "${WEIGHT_GROUP_SIZE}"
        "${transform_flag}"
        "${lwc_flag}"
        --no-flat_lac
        --omni_let_mode none
        --flat_transform_init identity
        --flat_epochs "${EPOCHS}"
        --flat_train_sample_size "${TRAIN_SAMPLES}"
        --flat_validation_sample_size "${HELDOUT_SAMPLES}"
        --flat_epoch_eval_interval 1
        --flat_transform_lr "${TRANSFORM_LR}"
        --flat_lwc_lr "${LWC_LR}"
        --flat_normalize_mse_gradient
        --calibration_only
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        command+=(--overwrite)
    fi

    echo "[${label}] gpu=${gpu} transform=${learn_transform} lwc=${use_lwc}"
    echo "[${label}] output=${output_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=${gpu}" "${command[@]}"
        return
    fi
    mkdir -p "${output_dir}"
    env CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    if [[ ! -s "${config_path}" || ! -s "${checkpoint_path}" ]]; then
        echo "[${label}] expected config/checkpoint was not produced." >&2
        return 5
    fi
}

run_wave() {
    local first_label="$1"
    local first_slug="$2"
    local first_transform="$3"
    local first_lwc="$4"
    local second_label="$5"
    local second_slug="$6"
    local second_transform="$7"
    local second_lwc="$8"

    run_arm "${first_label}" "${first_slug}" "${GPU_A}" "${first_transform}" "${first_lwc}" &
    local first_pid=$!
    run_arm "${second_label}" "${second_slug}" "${GPU_B}" "${second_transform}" "${second_lwc}" &
    local second_pid=$!
    local failed=0
    if ! wait "${first_pid}"; then
        failed=1
    fi
    if ! wait "${second_pid}"; then
        failed=1
    fi
    if (( failed != 0 )); then
        return 5
    fi
}

write_summary() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[summary] skipped in DRY_RUN mode."
        return
    fi
    mkdir -p "$(dirname "${SUMMARY_PATH}")"
    "${PYTHON_BIN}" - "${SUMMARY_PATH}" "${MODEL_NAME}" "${RESULTS_ROOT}" "${LAYER}" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
model_name = sys.argv[2]
results_root = Path(sys.argv[3])
layer_key = str(int(sys.argv[4]))
specs = (
    ("rtn", False, False),
    ("lwc_only", False, True),
    ("lt_only", True, False),
    ("lt_lwc", True, True),
)
arms = {}
for slug, learn_transform, use_lwc in specs:
    config_path = (
        results_root / slug / model_name / "ad" / "flatquant_core_config.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    item = payload["baseline_summaries"][layer_key]
    arms[slug] = {
        "learn_transform": learn_transform,
        "use_lwc": use_lwc,
        "config_path": str(config_path.resolve()),
        "initial_train_mse": item["initial_mse_loss"],
        "final_train_mse": item["final_mse_loss"],
        "initial_heldout_mse": item["initial_validation_mse_loss"],
        "final_heldout_mse": item["final_validation_mse_loss"],
        "best_epoch": item["best_epoch"],
        "trainable_transform_parameters": item["trainable_transform_parameters"],
        "trainable_lwc_parameters": item["trainable_lwc_parameters"],
    }
    arms[slug]["train_mse_reduction"] = (
        arms[slug]["initial_train_mse"] - arms[slug]["final_train_mse"]
    )
    arms[slug]["heldout_mse_reduction"] = (
        arms[slug]["initial_heldout_mse"] - arms[slug]["final_heldout_mse"]
    )

summary = {
    "protocol": {
        "model": model_name,
        "task": "ad",
        "layer": int(sys.argv[4]),
        "weight_quant_format": "fp4_e2m1",
        "activation_quant_format": "fp8_e4m3fn",
        "weight_group_size": 128,
        "calib_sample_size": int(os.environ.get("CALIB_SAMPLES", "128")),
        "train_sample_size": int(os.environ.get("TRAIN_SAMPLES", "64")),
        "heldout_sample_size": int(os.environ.get("HELDOUT_SAMPLES", "64")),
        "epochs": int(os.environ.get("EPOCHS", "15")),
        "seed": 42,
        "best_epoch_selected_by": "fixed_training_mse",
        "heldout_used_for_selection": False,
    },
    "arms": arms,
}
summary["heldout_interaction"] = (
    arms["lt_lwc"]["heldout_mse_reduction"]
    - arms["lt_only"]["heldout_mse_reduction"]
    - arms["lwc_only"]["heldout_mse_reduction"]
)
temporary = summary_path.with_name(f".{summary_path.name}.tmp")
temporary.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
os.replace(temporary, summary_path)

print("[four-arm] arm       train_initial    train_final  heldout_initial  heldout_final best")
for slug, _learn_transform, _use_lwc in specs:
    item = arms[slug]
    print(
        f"[four-arm] {slug:<9} "
        f"{item['initial_train_mse']:>13.6e} "
        f"{item['final_train_mse']:>13.6e} "
        f"{item['initial_heldout_mse']:>16.6e} "
        f"{item['final_heldout_mse']:>13.6e} "
        f"{str(item['best_epoch']):>4}"
    )
print(f"[four-arm] heldout_interaction={summary['heldout_interaction']:+.6e}")
print(f"[four-arm] summary={summary_path}")
PY
}

echo "[protocol] model=${MODEL_PATH} task=ad layer=${LAYER} seed=42"
echo "[protocol] W4A8-g128 calib=${CALIB_SAMPLES} train=${TRAIN_SAMPLES} heldout=${HELDOUT_SAMPLES}"
echo "[protocol] epochs=${EPOCHS} transform_lr=${TRANSFORM_LR} lwc_lr=${LWC_LR}"
echo "[protocol] best epoch uses training MSE only; held-out is diagnostic only"
echo "[protocol] gpus=${GPU_A},${GPU_B} results=${RESULTS_ROOT}"

echo
echo "[wave 1/2] RTN and LWC-only"
run_wave RTN rtn 0 0 LWC-only lwc_only 0 1

echo
echo "[wave 2/2] LT-only and LT+LWC"
run_wave LT-only lt_only 1 0 LT+LWC lt_lwc 1 1

export CALIB_SAMPLES TRAIN_SAMPLES HELDOUT_SAMPLES EPOCHS
write_summary
echo "[done] FlatQuant four-arm layer-${LAYER} MSE ablation completed."
