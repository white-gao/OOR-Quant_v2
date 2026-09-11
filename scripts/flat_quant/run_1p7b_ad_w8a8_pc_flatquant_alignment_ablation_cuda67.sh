#!/usr/bin/env bash
set -euo pipefail

# W8A8 entry point for the FlatQuant task-alignment ablation. The shared
# launcher keeps the W4A8 protocol and this W8A8 protocol exactly aligned.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

export QUANT_LABEL="${QUANT_LABEL:-w8a8}"
export WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-fp8_e4m3fn}"

exec bash "${SCRIPT_DIR}/run_1p7b_ad_w4a8_pc_flatquant_alignment_ablation_cuda67.sh" "$@"
