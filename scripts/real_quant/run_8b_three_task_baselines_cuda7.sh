#!/usr/bin/env bash
set -euo pipefail

# Serial full-precision and real naive-FP8 baselines for OneRec-8B.
# Run from the repository root:
#   bash scripts/real_quant/run_8b_three_task_baselines_cuda7.sh
#
# Optional overrides, for example:
#   DEVICE=cuda:7 OUTPUT_TAG=8b_baselines_rerun \
#     bash scripts/real_quant/run_8b_three_task_baselines_cuda7.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
REAL_RESULTS_ROOT="${ARTIFACTS_ROOT}/results/real_quant"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${ARTIFACTS_ROOT}/models/8B}"
DATA_DIR="${DATA_DIR:-${ARTIFACTS_ROOT}/data/onerec_data/benchmark_data}"
DEVICE="${DEVICE:-cuda:7}"
SAMPLE_SIZE="${SAMPLE_SIZE:-full}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_TAG="${OUTPUT_TAG:-8b_three_task_baselines_cuda7}"
OVERWRITE="${OVERWRITE:-0}"

FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-${REAL_RESULTS_ROOT}/recommender/bf16/${OUTPUT_TAG}}"
W8A8_OUTPUT_ROOT="${W8A8_OUTPUT_ROOT:-${REAL_RESULTS_ROOT}/recommender/rtn_w8a8/${OUTPUT_TAG}}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS+=(--overwrite)
fi

run_task() {
  local task="$1"
  local full_output_dir="${FULL_OUTPUT_ROOT}/${task}"
  local w8a8_output_dir="${W8A8_OUTPUT_ROOT}/${task}"

  mkdir -p "${full_output_dir}" "${w8a8_output_dir}"

  run_full_precision() {
    echo "[full precision] task=${task} device=${DEVICE}"
    "${PYTHON_BIN}" -m real_quant.full_precision.run_hf_baseline \
      --model_path "${MODEL_PATH}" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${full_output_dir}" \
      --task "${task}" \
      --sample_size "${SAMPLE_SIZE}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE}" \
      --evaluate \
      "${WRITE_ARGS[@]}" 2>&1 | tee "${full_output_dir}/run.log"
  }

  run_naive_w8a8() {
    echo "[naive W8A8] task=${task} device=${DEVICE}"
    "${PYTHON_BIN}" -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
      --model_path "${MODEL_PATH}" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${w8a8_output_dir}" \
      --task "${task}" \
      --sample_size "${SAMPLE_SIZE}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE}" \
      --weight_quant_mode minmax \
      --evaluate \
      "${WRITE_ARGS[@]}" 2>&1 | tee "${w8a8_output_dir}/run.log"
  }

  if [[ "${task}" == "video" ]]; then
    run_naive_w8a8
    run_full_precision
  else
    run_full_precision
    run_naive_w8a8
  fi
}

# One process at a time. Run pure W8A8 video first to estimate its wall time.
run_task video
run_task product
run_task ad

echo "All 8B baselines completed."
