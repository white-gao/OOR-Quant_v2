#!/usr/bin/env bash
# Usage: GPU=5 bash paper/motivation/scripts/run_wikitext2_reference.sh
set -euo pipefail
MOTIVATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/dataDisk/guowei/models/1.7B}"
GPU="${GPU:-5}"
RUN_TAG="${RUN_TAG:-v2_text_controls_seed42}"
FIGURE_TAG="${FIGURE_TAG:-v2_wikitext2_seed42}"
SID_RUN_TAG="${SID_RUN_TAG:-v1_seed42}"
DRY_RUN="${DRY_RUN:-0}"
ACTION="${1:-all}"
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU must be one physical GPU index." >&2
  exit 2
fi
for tag in "$RUN_TAG" "$FIGURE_TAG" "$SID_RUN_TAG"; do
  if [[ ! "$tag" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Run/figure tags may contain only letters, digits, _ and -." >&2
    exit 2
  fi
done
if [[ "$FIGURE_TAG" == "$SID_RUN_TAG" ]]; then
  echo "FIGURE_TAG must differ from SID_RUN_TAG to preserve existing figures." >&2
  exit 2
fi
SMOKE_ARGS=()
if [[ "$ACTION" == smoke ]]; then
  RUN_TAG="${RUN_TAG}_smoke"
  FIGURE_TAG="${FIGURE_TAG}_smoke"
  SMOKE_ARGS=(--max-articles 2 --positions-per-article 2 --length-positions 1)
fi
TEXT_ROOT="$MOTIVATION_DIR/results/$RUN_TAG/wikitext2/onerec_bf16"
SID_ROOT="$MOTIVATION_DIR/results/$SID_RUN_TAG"
FIGURE_ROOT="$MOTIVATION_DIR/figures/$FIGURE_TAG"
LOG_ROOT="$MOTIVATION_DIR/logs/$RUN_TAG"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HOME="$MOTIVATION_DIR/cache/huggingface"
export XDG_CACHE_HOME="$MOTIVATION_DIR/cache/xdg"
export MPLCONFIGDIR="$MOTIVATION_DIR/cache/matplotlib"
export TMPDIR="$MOTIVATION_DIR/cache/tmp"
if [[ "$DRY_RUN" != 1 ]]; then
  mkdir -p "$LOG_ROOT" "$TMPDIR"
fi
DATA_ARGS=()
if [[ -n "${WIKITEXT_FILE:-}" ]]; then
  DATA_ARGS=(--dataset-file "$WIKITEXT_FILE")
fi
COMMON_ARGS=(--model-path "$MODEL_PATH" --sid-run-root "$SID_ROOT" --output-dir "$TEXT_ROOT"
             "${DATA_ARGS[@]}" "${SMOKE_ARGS[@]}")

run() {
  local label="$1"
  shift
  printf '[command] '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@" 2>&1 | tee -a "$LOG_ROOT/$label.log"
  fi
}
preflight() {
  run preflight "$PYTHON_BIN" -u "$MOTIVATION_DIR/scripts/probe_wikitext2.py" preflight "${COMMON_ARGS[@]}"
}
collect() {
  run wikitext2 "$PYTHON_BIN" -u "$MOTIVATION_DIR/scripts/probe_wikitext2.py" run "${COMMON_ARGS[@]}"
}
plot() {
  run plot "$PYTHON_BIN" -u "$MOTIVATION_DIR/scripts/plot_motivation.py" \
    --run-root "$SID_ROOT" --task ad --text-root "$TEXT_ROOT" \
    --extra-text-root "$SID_ROOT/text/bf16" --output-dir "$FIGURE_ROOT"
}
echo "[protocol] OneRec-1.7B BF16 / WikiText-2 test; GPU=$GPU; no training or AD reevaluation"
echo "[protocol] primary + paired 256/512/1024-token controls; resume after each position"
echo "[reuse] $SID_ROOT (AD + CMRC, read only)"
echo "[results] $TEXT_ROOT"
echo "[figures] $FIGURE_ROOT"
case "$ACTION" in
  all|smoke) preflight; collect; plot ;;
  preflight) preflight ;;
  run) collect ;;
  plot) plot ;;
  *) echo "Usage: bash $0 [all|preflight|run|plot|smoke]" >&2; exit 2 ;;
esac
