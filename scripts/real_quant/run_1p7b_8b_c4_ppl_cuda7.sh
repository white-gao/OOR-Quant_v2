#!/usr/bin/env bash
set -euo pipefail

# Serial C4 English-validation PPL comparison for OneRec-1.7B and OneRec-8B.
# The first step materializes one shared, deterministic 256 x 2048-token cache;
# the four following runs score that exact cache with BF16 and pure RTN W8A8.
#
# Run:
#   bash scripts/real_quant/run_1p7b_8b_c4_ppl_cuda7.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
REAL_RESULTS_ROOT="${ARTIFACTS_ROOT}/results/real_quant"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_1P7B="${MODEL_1P7B:-${MODEL_ROOT}/1.7B}"
MODEL_8B="${MODEL_8B:-${MODEL_ROOT}/8B}"
DEVICE="${DEVICE:-cuda:7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REAL_RESULTS_ROOT}/generic/c4_en_validation_256x2048}"
C4_NUM_SEQUENCES="${C4_NUM_SEQUENCES:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
MAX_SEQUENCES="${MAX_SEQUENCES:-full}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS+=(--overwrite)
fi

run_ppl() {
  local label="$1"
  local model_path="$2"
  local quantization="$3"
  local output_dir="$4"

  printf '\n[c4-ppl] %s\n' "${label}"
  "${PYTHON_BIN}" -m real_quant.full_precision.run_ppl_benchmark \
    --model_path "${model_path}" \
    --dataset c4 \
    --output_dir "${output_dir}" \
    --device "${DEVICE}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --c4_num_sequences "${C4_NUM_SEQUENCES}" \
    --max_sequences "${MAX_SEQUENCES}" \
    --quantization "${quantization}" \
    "${WRITE_ARGS[@]}"
}

printf '[c4-ppl] preparing shared C4 validation cache\n'
"${PYTHON_BIN}" -m real_quant.full_precision.run_ppl_benchmark \
  --model_path "${MODEL_1P7B}" \
  --dataset c4 \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --c4_num_sequences "${C4_NUM_SEQUENCES}" \
  --prepare_only

run_ppl "1.7B BF16" "${MODEL_1P7B}" full_precision "${OUTPUT_ROOT}/1p7b_bf16"
run_ppl "1.7B RTN W8A8" "${MODEL_1P7B}" rtn_w8a8 "${OUTPUT_ROOT}/1p7b_rtn_w8a8"
run_ppl "8B BF16" "${MODEL_8B}" full_precision "${OUTPUT_ROOT}/8b_bf16"
run_ppl "8B RTN W8A8" "${MODEL_8B}" rtn_w8a8 "${OUTPUT_ROOT}/8b_rtn_w8a8"

printf '\n[c4-ppl] all four paired evaluations completed.\n'
