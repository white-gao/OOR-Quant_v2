#!/usr/bin/env bash
set -euo pipefail

# Paired OneRec-1.7B GSM8K evaluation for full precision and fake INT4-W/FP8-A.
# This matches the established non-thinking GSM8K protocol: fixed 5-shot CoT,
# greedy decoding, up to 512 new tokens, and numerical final-answer matching.
# Runs are resumable by default; set OVERWRITE=1 to restart either result.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DEVICE="${DEVICE:-cuda:7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/generic/gsm8k_1p7b_qdq_precision_sweep}"
SAMPLE_SIZE="${SAMPLE_SIZE:-full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-5}"
OVERWRITE="${OVERWRITE:-0}"

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS+=(--overwrite)
fi

run_gsm8k() {
  local label="$1"
  local quantization="$2"
  local output_dir="$3"
  shift 3

  echo "[gsm8k] ${label}"
  python -m real_quant.full_precision.run_math_benchmark \
    --model_path "${MODEL_PATH}" \
    --benchmark gsm8k \
    --output_dir "${output_dir}" \
    --device "${DEVICE}" \
    --sample_size "${SAMPLE_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --gsm8k_num_fewshot "${GSM8K_NUM_FEWSHOT}" \
    --quantization "${quantization}" \
    --no-enable_thinking \
    "${WRITE_ARGS[@]}" \
    "$@"
}

run_gsm8k "BF16 full precision" \
  full_precision "${OUTPUT_ROOT}/bf16"
run_gsm8k "INT4-W / FP8-A fake QDQ" \
  fake_qdq "${OUTPUT_ROOT}/int4w_fp8a" \
  --weight_quant_format int4 \
  --activation_quant_format fp8_e4m3fn
