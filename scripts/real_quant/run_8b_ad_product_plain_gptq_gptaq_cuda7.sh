#!/usr/bin/env bash
set -euo pipefail

# Serial pure-W8A8 GPTQ/GPTAQ experiments for OneRec-8B on ad and product.
# Run from the repository root:
#   bash scripts/real_quant/run_8b_ad_product_plain_gptq_gptaq_cuda7.sh
#
# Both methods use batch size 1, no decode-A16, and GPTAQ explicitly disables
# activation-aware statistics.  Override values when needed, for example:
#   GPTQ_CALIB_SAMPLE_SIZE=1024 OUTPUT_TAG=8b_gptq_gptaq_calib1024 \
#     bash scripts/real_quant/run_8b_ad_product_plain_gptq_gptaq_cuda7.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
REAL_RESULTS_ROOT="${ARTIFACTS_ROOT}/results/real_quant"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/8B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
DEVICE="${DEVICE:-cuda:7}"
SAMPLE_SIZE="${SAMPLE_SIZE:-full}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GPTQ_CALIB_SAMPLE_SIZE="${GPTQ_CALIB_SAMPLE_SIZE:-128}"
OUTPUT_TAG="${OUTPUT_TAG:-8b_ad_product_plain_gptq_gptaq_w8a8_calib${GPTQ_CALIB_SAMPLE_SIZE}}"
OVERWRITE="${OVERWRITE:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REAL_RESULTS_ROOT}/recommender/ptq/${OUTPUT_TAG}}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS+=(--overwrite)
fi

run_experiment() {
  local task="$1"
  local method="$2"
  local output_dir="${OUTPUT_ROOT}/${task}/${method}"
  local method_args=()

  case "${method}" in
    gptq)
      method_args=(--weight_quant_mode gptq)
      ;;
    gptaq_no_activation_aware)
      method_args=(--weight_quant_mode gptaq --no-gptaq_activation_aware)
      ;;
    *)
      echo "Unsupported method: ${method}" >&2
      return 2
      ;;
  esac

  mkdir -p "${output_dir}"
  echo "[${method}] task=${task} device=${DEVICE} calib=${GPTQ_CALIB_SAMPLE_SIZE}"
  "${PYTHON_BIN}" -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${output_dir}" \
    --task "${task}" \
    --sample_size "${SAMPLE_SIZE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --gptq_calib_sample_size "${GPTQ_CALIB_SAMPLE_SIZE}" \
    "${method_args[@]}" \
    --evaluate \
    "${WRITE_ARGS[@]}" 2>&1 | tee "${output_dir}/run.log"
}

# One process at a time: GPTQ then non-activation-aware GPTAQ for each domain.
for task in ad product; do
  run_experiment "${task}" gptq
  run_experiment "${task}" gptaq_no_activation_aware
done

echo "All 8B ad/product plain GPTQ and plain GPTAQ experiments completed."
