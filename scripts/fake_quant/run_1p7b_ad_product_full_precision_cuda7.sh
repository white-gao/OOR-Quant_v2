#!/usr/bin/env bash
set -euo pipefail

# Serial OneRec-1.7B full-precision evaluation through the fake-QDQ runner.
# Runs AD first, then Product, on the new benchmark_data split.
#
# From the repository root:
#   bash scripts/fake_quant/run_1p7b_ad_product_full_precision_cuda7.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${ARTIFACTS_ROOT}/models/1.7B}"
DATA_DIR="${DATA_DIR:-${ARTIFACTS_ROOT}/data/onerec_data/benchmark_data}"
DEVICE="${DEVICE:-cuda:7}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OUTPUT_TAG="${OUTPUT_TAG:-1p7b_full_precision_new_benchmark_data_ad_product}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/${OUTPUT_TAG}}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

run_task() {
    local task="$1"
    local task_output_dir="${OUTPUT_ROOT}/${task}"
    mkdir -p "${task_output_dir}"

    echo "[fake-quant / 1.7B / full precision] ${task} -> ${task_output_dir}"
    "${PYTHON_BIN}" -m fake_quant.run_m1_onerec_ad \
        --task "${task}" \
        --mode full_precision \
        --weight_quant_format none \
        --activation_quant_format none \
        --model_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${task_output_dir}" \
        --device "${DEVICE}" \
        --eval_sample_size "${EVAL_SAMPLE_SIZE}" \
        --evaluate \
        "${OVERWRITE_ARGS[@]}" \
        2>&1 | tee "${task_output_dir}/run.log"
}

run_task ad
run_task product

echo "Completed AD and Product full-precision runs: ${OUTPUT_ROOT}"
