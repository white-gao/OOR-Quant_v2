#!/usr/bin/env bash
set -euo pipefail

# W8A8 entry point for the formal per-channel FlatQuant staged protocol.
# All training/evaluation controls (GPUS, RUN_EVAL, OVERWRITE, DRY_RUN, etc.)
# are inherited from the shared staged launcher.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

export WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT:-fp8_e4m3fn}"
export ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT:-fp8_e4m3fn}"
export QUANT_LABEL="${QUANT_LABEL:-w8a8}"

exec bash "${SCRIPT_DIR}/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh" "$@"
