#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/onerec_data/benchmark_data}"
GPU="${GPU:-7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE="${SMOKE:-0}"
OVERWRITE="${OVERWRITE:-0}"
TRANSFORM_INIT="${TRANSFORM_INIT:-identity}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-128}"

if [[ "${SMOKE}" != "0" && "${SMOKE}" != "1" ]]; then
    echo "SMOKE must be 0 or 1; got ${SMOKE}." >&2
    exit 2
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1; got ${OVERWRITE}." >&2
    exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "GPU must be one physical GPU index; got ${GPU}." >&2
    exit 2
fi

if [[ "${SMOKE}" == "1" ]]; then
    LAYERS="${LAYERS:-0}"
    CALIB_SAMPLES="${CALIB_SAMPLES:-4}"
    EPOCHS="${EPOCHS:-1}"
    RUN_NAME="smoke_layer0_calib${CALIB_SAMPLES}_epoch${EPOCHS}"
else
    LAYERS="${LAYERS:-all}"
    CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
    EPOCHS="${EPOCHS:-15}"
    RUN_NAME="all_layers_calib${CALIB_SAMPLES}_epoch${EPOCHS}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/results/flat_quant/recommender/1p7b_ad_w4a8_g128_core/${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

COMMAND=(
    "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
    --task ad
    --model_path "${MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --device cuda:0
    --mode flatquant_core
    --layers "${LAYERS}"
    --calib_sample_size "${CALIB_SAMPLES}"
    --weight_quant_format fp4_e2m1
    --activation_quant_format fp8_e4m3fn
    --weight_quant_scheme symmetric
    --weight_group_size "${WEIGHT_GROUP_SIZE}"
    --omni_lwc
    --no-flat_lac
    --flat_transform_init "${TRANSFORM_INIT}"
    --flat_epochs "${EPOCHS}"
    --flat_transform_lr "${FLAT_TRANSFORM_LR:-5e-3}"
    --flat_lwc_lr "${FLAT_LWC_LR:-5e-2}"
    --flat_init_lwc_logit "${FLAT_INIT_LWC_LOGIT:-4.0}"
    --flat_normalize_mse_gradient
    --calibration_only
)
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMAND+=(--overwrite)
fi

echo "[flatquant-core] model=${MODEL_PATH} task=ad physical_gpu=${GPU}"
echo "[flatquant-core] layers=${LAYERS} calib=${CALIB_SAMPLES} epochs=${EPOCHS} W4A8 g128=${WEIGHT_GROUP_SIZE}"
echo "[flatquant-core] transform_init=${TRANSFORM_INIT} output=${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
env CUDA_VISIBLE_DEVICES="${GPU}" "${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
