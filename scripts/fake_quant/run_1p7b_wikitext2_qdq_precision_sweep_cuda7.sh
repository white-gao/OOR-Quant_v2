#!/usr/bin/env bash
set -euo pipefail

# Paired OneRec-1.7B WikiText-2 PPL sweep for fake-QDQ numeric formats.
# Every run uses the complete raw test stream and the same 2048-token windows.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DEVICE="${DEVICE:-cuda:7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/generic/wikitext2_1p7b_qdq_precision_sweep}"

run_ppl() {
  local label="$1"
  local quantization="$2"
  local weight_format="$3"
  local activation_format="$4"
  local output_dir="$5"

  echo "[wikitext2] ${label}"
  local args=(
    -m real_quant.full_precision.run_ppl_benchmark
    --model_path "${MODEL_PATH}"
    --dataset wikitext2
    --device "${DEVICE}"
    --output_dir "${output_dir}"
    --quantization "${quantization}"
    --overwrite
  )
  if [[ "${quantization}" == "fake_qdq" ]]; then
    args+=(
      --weight_quant_format "${weight_format}"
      --activation_quant_format "${activation_format}"
    )
  fi
  python "${args[@]}"
}

run_ppl "BF16 full precision" full_precision none none "${OUTPUT_ROOT}/bf16"
run_ppl "FP8-W / BF16-A" fake_qdq fp8_e4m3fn none "${OUTPUT_ROOT}/fp8w_bf16a"
run_ppl "FP8-W / FP8-A" fake_qdq fp8_e4m3fn fp8_e4m3fn "${OUTPUT_ROOT}/fp8w_fp8a"
run_ppl "INT8-W / BF16-A" fake_qdq int8 none "${OUTPUT_ROOT}/int8w_bf16a"
run_ppl "INT8-W / FP8-A" fake_qdq int8 fp8_e4m3fn "${OUTPUT_ROOT}/int8w_fp8a"
run_ppl "INT4-W / BF16-A" fake_qdq int4 none "${OUTPUT_ROOT}/int4w_bf16a"
run_ppl "INT4-W / FP8-A" fake_qdq int4 fp8_e4m3fn "${OUTPUT_ROOT}/int4w_fp8a"
