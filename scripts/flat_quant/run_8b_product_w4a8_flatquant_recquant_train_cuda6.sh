#!/usr/bin/env bash
set -euo pipefail

# Training only: FlatQuant -> RecQuant, on one GPU (default: physical GPU 6).
# Preserve the original 8B Product W4A8 main-experiment paths and protocol.
# Complete stages are reused; partial training stops rather than overwrites.
# DRY_RUN=1 prints commands without loading weights or creating outputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-6}"
DRY_RUN="${DRY_RUN:-0}"
if [[ ! "${GPU}" =~ ^[0-9]+$ || ! "${DRY_RUN}" =~ ^[01]$ ]]; then
    echo "GPU must be one numeric GPU ID; DRY_RUN must be 0 or 1." >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
MODEL_PATH="$(realpath -m "${MODEL_PATH:-${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}/8B}")"
DATA_DIR="$(realpath -m "${DATA_DIR:-${OOR_QUANT_BENCHMARK_DATA:-${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}/onerec_data/benchmark_data}}")"
W4A8_RUN_ROOT="$(realpath -m "${W4A8_RUN_ROOT:-${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}/results/main_experiments/8b_ad_product_w4a8_pc}")"
MODEL_NAME="$(basename "${MODEL_PATH}")"
TASK_ROOT="${W4A8_RUN_ROOT}/product"
FLAT_ROOT="${TASK_ROOT}/flatquant"
REC_ROOT="${TASK_ROOT}/recquant/abc_boundary0.3_joint"
FLAT_CHECKPOINT="${FLAT_ROOT}/final_train512_heldout512/${MODEL_NAME}/product/flatquant_calibration"

# Fail before training if a 1.7B model was accidentally supplied.
"${PYTHON_BIN}" - "${MODEL_PATH}" "${DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path
model, data = map(Path, sys.argv[1:])
config = json.loads((model / 'config.json').read_text())
if config.get('num_hidden_layers') != 36 or config.get('hidden_size') != 4096:
    raise SystemExit('Expected OneRec-8B: 36 blocks and hidden_size=4096.')
if not data.is_dir():
    raise SystemExit(f'Missing benchmark data directory: {data}')
PY

COMMON_ENV=(
    PYTHON_BIN="${PYTHON_BIN}" MODEL_PATH="${MODEL_PATH}" DATA_DIR="${DATA_DIR}"
    TASK=product GPUS="${GPU}" TRAIN_GPU="${GPU}" NUM_LAYERS=36
    QUANT_LABEL=w4a8 WEIGHT_QUANT_FORMAT=fp4_e2m1 ACTIVATION_QUANT_FORMAT=fp8_e4m3fn
    RUN_EVAL=0 EVAL_SAMPLE_SIZE=full OVERWRITE=0 EVAL_OVERWRITE=0 SMOKE=0 DRY_RUN="${DRY_RUN}"
    EPOCHS=15 TRANSFORM_LR=5e-3 LWC_LR=5e-2 LAC_LR=5e-2
    WEIGHT_DECAY=0.01 DIAG_ALPHA=0.5 INIT_LWC_LOGIT=4.0 INIT_LAC_LOGIT=4.0
)

echo "[protocol] OneRec-8B Product W4A8 per-channel; training GPU=${GPU}; full evaluation disabled."
echo "[protocol] original main-experiment directory=${TASK_ROOT}"
echo "[1/2] FlatQuant: prefix128, final512 + heldout512, 15 epochs."
env "${COMMON_ENV[@]}" RUN_ROOT="${FLAT_ROOT}" \
    PREFIX_CALIB_SAMPLES=128 FINAL_CALIB_SAMPLES=1024 \
    FINAL_TRAIN_SAMPLES=512 FINAL_HELDOUT_SAMPLES=512 \
    bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh

echo "[2/2] RecQuant: same FlatQuant checkpoint, final1024, 15 epochs, ABC CE + 0.3 boundary."
env "${COMMON_ENV[@]}" SOURCE_CHECKPOINT_DIR="${FLAT_CHECKPOINT}" \
    OUTPUT_DIR="${REC_ROOT}" CHECKPOINT_DIR="${REC_ROOT}/${MODEL_NAME}/product/flatquant_calibration" \
    EVAL_OUTPUT_DIR="${TASK_ROOT}/recquant_full_eval" CALIB_SAMPLES=1024 TRAIN_SAMPLES=1024 \
    LEARN_TRANSFORM=1 LEARN_LAC=1 LFQ_LOSS_WEIGHT=1.0 BOUNDARY_WEIGHT=0.3 \
    BOUNDARY_TOPK=32 BOUNDARY_NEGATIVES=32 BOUNDARY_TIE_THRESHOLD=0.01 BOUNDARY_GAP_SCALE=1.0 \
    bash scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh

echo "[done] Training stages handled; full-task evaluation remains pending."
echo "[checkpoint] FlatQuant: ${FLAT_CHECKPOINT}"
echo "[checkpoint] RecQuant: ${REC_ROOT}/${MODEL_NAME}/product/flatquant_calibration"
