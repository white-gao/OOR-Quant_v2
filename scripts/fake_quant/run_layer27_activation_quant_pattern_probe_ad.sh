#!/usr/bin/env bash
set -euo pipefail

# Local activation-QDQ visualization for five reproducibly sampled AD prompts.
# The BF16 model is kept intact; only captured layer-27 q_proj/o_proj inputs
# are passed through the repository's dynamic per-token FP8 fake-QDQ function.

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/results/fake_quant/probes/activation_quant_patterns_layer27_ad_fp8}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m fake_quant.probe_activation_quant_patterns \
  --device cuda:0 \
  --layer 27 \
  --sample_pool_size 3000 \
  --num_samples 5 \
  --seed 42 \
  --target_tokens 200 \
  --channel_slice_size 512 \
  --activation_quant_format fp8_e4m3fn \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
