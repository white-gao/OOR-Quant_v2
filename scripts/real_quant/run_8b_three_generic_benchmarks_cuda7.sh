#!/usr/bin/env bash
set -euo pipefail

# Serial BF16 vs. pure RTN W8A8 evaluation of OneRec-8B on the three generic
# controls used for OneRec-1.7B:
#   1. WikiText-2 raw test token PPL
#   2. GSM8K (non-thinking, greedy, fixed 5-shot CoT, flexible numerical match)
#   3. MATH-500 (non-thinking, greedy, boxed-answer matching)
#
# Each subprocess flushes records incrementally. Re-running this script resumes
# unfinished work in every result directory; set OVERWRITE=1 only to discard
# all six existing runs and restart the complete suite.
#
# Run:
#   bash scripts/real_quant/run_8b_three_generic_benchmarks_cuda7.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
REAL_RESULTS_ROOT="${ARTIFACTS_ROOT}/results/real_quant"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${ARTIFACTS_ROOT}/models/8B}"
DEVICE="${DEVICE:-cuda:7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REAL_RESULTS_ROOT}/generic/8b_generic_benchmark_suite}"
OVERWRITE="${OVERWRITE:-0}"

PPL_SEQUENCE_LENGTH="${PPL_SEQUENCE_LENGTH:-2048}"
PPL_MAX_SEQUENCES="${PPL_MAX_SEQUENCES:-full}"
MATH_SAMPLE_SIZE="${MATH_SAMPLE_SIZE:-full}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-1024}"
GSM8K_SAMPLE_SIZE="${GSM8K_SAMPLE_SIZE:-full}"
GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-512}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-5}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS+=(--overwrite)
fi

run_step() {
  local name="$1"
  shift
  printf '\n[8b-generic-suite] %s\n' "${name}"
  "$@"
}

# Run the short PPL controls first, then GSM8K. Leave the longest pair
# (MATH-500) for last so that the two quicker generic controls finish first.
run_step "WikiText-2 BF16 PPL" \
  "${PYTHON_BIN}" -m real_quant.full_precision.run_ppl_benchmark \
  --model_path "${MODEL_PATH}" \
  --dataset wikitext2 \
  --output_dir "${OUTPUT_ROOT}/wikitext2_bf16" \
  --device "${DEVICE}" \
  --sequence_length "${PPL_SEQUENCE_LENGTH}" \
  --max_sequences "${PPL_MAX_SEQUENCES}" \
  --quantization full_precision \
  "${WRITE_ARGS[@]}"

run_step "WikiText-2 RTN W8A8 PPL" \
  "${PYTHON_BIN}" -m real_quant.full_precision.run_ppl_benchmark \
  --model_path "${MODEL_PATH}" \
  --dataset wikitext2 \
  --output_dir "${OUTPUT_ROOT}/wikitext2_rtn_w8a8" \
  --device "${DEVICE}" \
  --sequence_length "${PPL_SEQUENCE_LENGTH}" \
  --max_sequences "${PPL_MAX_SEQUENCES}" \
  --quantization rtn_w8a8 \
  "${WRITE_ARGS[@]}"

run_step "GSM8K BF16 non-thinking 5-shot" \
  "${PYTHON_BIN}" -m real_quant.full_precision.run_math_benchmark \
  --model_path "${MODEL_PATH}" \
  --benchmark gsm8k \
  --output_dir "${OUTPUT_ROOT}/gsm8k_bf16_nonthinking_5shot" \
  --device "${DEVICE}" \
  --sample_size "${GSM8K_SAMPLE_SIZE}" \
  --max_new_tokens "${GSM8K_MAX_NEW_TOKENS}" \
  --gsm8k_num_fewshot "${GSM8K_NUM_FEWSHOT}" \
  --quantization full_precision \
  --no-enable_thinking \
  "${WRITE_ARGS[@]}"

run_step "GSM8K RTN W8A8 non-thinking 5-shot" \
  "${PYTHON_BIN}" -m real_quant.full_precision.run_math_benchmark \
  --model_path "${MODEL_PATH}" \
  --benchmark gsm8k \
  --output_dir "${OUTPUT_ROOT}/gsm8k_rtn_w8a8_nonthinking_5shot" \
  --device "${DEVICE}" \
  --sample_size "${GSM8K_SAMPLE_SIZE}" \
  --max_new_tokens "${GSM8K_MAX_NEW_TOKENS}" \
  --gsm8k_num_fewshot "${GSM8K_NUM_FEWSHOT}" \
  --quantization rtn_w8a8 \
  --no-enable_thinking \
  "${WRITE_ARGS[@]}"

run_step "MATH-500 BF16 non-thinking" \
  "${PYTHON_BIN}" -m real_quant.full_precision.run_math_benchmark \
  --model_path "${MODEL_PATH}" \
  --benchmark math500 \
  --output_dir "${OUTPUT_ROOT}/math500_bf16_nonthinking" \
  --device "${DEVICE}" \
  --sample_size "${MATH_SAMPLE_SIZE}" \
  --max_new_tokens "${MATH_MAX_NEW_TOKENS}" \
  --quantization full_precision \
  --no-enable_thinking \
  "${WRITE_ARGS[@]}"

run_step "MATH-500 RTN W8A8 non-thinking" \
  "${PYTHON_BIN}" -m real_quant.full_precision.run_math_benchmark \
  --model_path "${MODEL_PATH}" \
  --benchmark math500 \
  --output_dir "${OUTPUT_ROOT}/math500_rtn_w8a8_nonthinking" \
  --device "${DEVICE}" \
  --sample_size "${MATH_SAMPLE_SIZE}" \
  --max_new_tokens "${MATH_MAX_NEW_TOKENS}" \
  --quantization rtn_w8a8 \
  --no-enable_thinking \
  "${WRITE_ARGS[@]}"

printf '\n[8b-generic-suite] all six evaluations completed.\n'
