#!/usr/bin/env bash
set -euo pipefail

# Run OneRec fake-QDQ quantization on the AD benchmark.
# Defaults live in fake_quant/run_m1_onerec_ad.py. Calibration and evaluation
# use the shared benchmark_data directory, and recommendation generation uses
# deterministic 32-beam decoding.
# uses 32 beams.
#
# Examples:
#   bash fake_quant/run_learnable_quant_ad.sh
#   MODE=smoothquant_w8a8 DEVICE=cuda:1 bash fake_quant/run_learnable_quant_ad.sh
#   WEIGHT_QUANT_FORMAT=int4 ACTIVATION_QUANT_FORMAT=int8 DEVICE=cuda:7 bash scripts/fake_quant/run_learnable_quant_ad.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
FAKE_RESULTS_ROOT="${ARTIFACTS_ROOT}/results/fake_quant"

MODE="${MODE:-baseline_qdq}"  # baseline_qdq, baseline_w8a8, smoothquant_w8a8, or gptq_fp8_w8a8
WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-int8}"
ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT:-int8}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"

LAYERS="${LAYERS:-all}"
DEVICE="${DEVICE:-cuda}"
CALIB_SAMPLE_SIZE="${CALIB_SAMPLE_SIZE:-1024}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"

OVERWRITE="${OVERWRITE:-1}"
EVALUATE="${EVALUATE:-1}"
COMPUTE_SID_PPL="${COMPUTE_SID_PPL:-0}"
SID_PPL_MAX_ITEMS="${SID_PPL_MAX_ITEMS:-1}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="${MODE}_ad_calib${CALIB_SAMPLE_SIZE}_$(date +%Y%m%d_%H%M%S)"
  fi
  OUTPUT_DIR="${FAKE_RESULTS_ROOT}/recommender/${RUN_NAME}"
fi

args=(
  --mode "${MODE}"
  --weight_quant_format "${WEIGHT_QUANT_FORMAT}"
  --activation_quant_format "${ACTIVATION_QUANT_FORMAT}"
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --layers "${LAYERS}"
  --device "${DEVICE}"
  --calib_sample_size "${CALIB_SAMPLE_SIZE}"
  --eval_sample_size "${EVAL_SAMPLE_SIZE}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "${EVALUATE}" == "1" ]]; then
  args+=(--evaluate)
fi

if [[ "${COMPUTE_SID_PPL}" == "1" ]]; then
  args+=(--compute_sid_ppl --sid_ppl_max_items "${SID_PPL_MAX_ITEMS}")
fi

echo "Running fake-QDQ quantization:"
printf '  MODE=%s WEIGHT_QUANT_FORMAT=%s ACTIVATION_QUANT_FORMAT=%s\n' "${MODE}" "${WEIGHT_QUANT_FORMAT}" "${ACTIVATION_QUANT_FORMAT}"
printf '  DEVICE=%s LAYERS=%s\n' "${DEVICE}" "${LAYERS}"
printf '  MODEL_PATH=%s\n' "${MODEL_PATH}"
printf '  DATA_DIR=%s\n' "${DATA_DIR}"
printf '  OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
printf '  CALIB_SAMPLE_SIZE=%s EVAL_SAMPLE_SIZE=%s\n' "${CALIB_SAMPLE_SIZE}" "${EVAL_SAMPLE_SIZE}"
printf '  COMPUTE_SID_PPL=%s SID_PPL_MAX_ITEMS=%s\n' "${COMPUTE_SID_PPL}" "${SID_PPL_MAX_ITEMS}"

python3 -m fake_quant.run_m1_onerec_ad "${args[@]}" "$@"
