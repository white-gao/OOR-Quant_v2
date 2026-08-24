#!/usr/bin/env bash
set -euo pipefail

# Compare layer-27 q_proj/o_proj outputs from the BF16 trajectory against the
# full-model FP8-W/FP8-A RTN trajectory on the same five compressed prompts.

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PROBE_DIR="${INPUT_PROBE_DIR:-artifacts/results/fake_quant/probes/activation_quant_patterns_layer27_ad_fp8}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/results/fake_quant/probes/w8a8_projection_output_patterns_layer27_ad_fp8}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m fake_quant.probe_w8a8_projection_output_patterns \
  --device cuda:0 \
  --layer 27 \
  --channel_slice_size 512 \
  --weight_quant_format fp8_e4m3fn \
  --activation_quant_format fp8_e4m3fn \
  --input_probe_dir "${INPUT_PROBE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
