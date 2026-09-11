#!/usr/bin/env bash
set -euo pipefail

# Paired OneRec-1.7B AD evaluation under full precision and fake QDQ.
# All runs use the same first 1,000 examples of the AD test split, generation
# settings fixed by fake_quant.run_m1_onerec_ad, and BF16 F.linear after QDQ.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
DEVICE="${DEVICE:-cuda:7}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_ad${EVAL_SAMPLE_SIZE}_qdq_precision_sweep}"

run_ad() {
  local label="$1"
  local mode="$2"
  local weight_format="$3"
  local activation_format="$4"
  local output_dir="$5"

  echo "[ad] ${label}"
  python -m fake_quant.run_m1_onerec_ad \
    --task ad \
    --mode "${mode}" \
    --weight_quant_format "${weight_format}" \
    --activation_quant_format "${activation_format}" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${output_dir}" \
    --device "${DEVICE}" \
    --eval_sample_size "${EVAL_SAMPLE_SIZE}" \
    --overwrite \
    --evaluate
}

run_ad "BF16 full precision" full_precision none none "${OUTPUT_ROOT}/bf16"
run_ad "FP8-W / FP8-A" baseline_qdq fp8_e4m3fn fp8_e4m3fn "${OUTPUT_ROOT}/fp8w_fp8a"
run_ad "INT8-W / FP8-A" baseline_qdq int8 fp8_e4m3fn "${OUTPUT_ROOT}/int8w_fp8a"
run_ad "INT4-W / FP8-A" baseline_qdq int4 fp8_e4m3fn "${OUTPUT_ROOT}/int4w_fp8a"
