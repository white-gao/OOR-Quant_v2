#!/usr/bin/env bash
# Usage: GPU=2 TASKS=ad bash paper/motivation/scripts/run_task.sh e2
set -euo pipefail
MOTIVATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
GPU="${GPU:-6}"
TASKS="${TASKS:-ad}"
RUN_TAG="${RUN_TAG:-v1_seed42}"
MODEL_PATH="${MODEL_PATH:-/root/dataDisk/guowei/models/1.7B}"
DATA_DIR="${DATA_DIR:-/root/dataDisk/guowei/data/onerec_data/benchmark_data}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
if [[ ! "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "RUN_TAG must contain only letters, digits, _ or -" >&2
  exit 2
fi
if [[ "$GPU" == *,* ]]; then
  echo "Each probe uses one GPU. Launch different jobs separately to use multiple GPUs." >&2
  exit 2
fi
RUN_ROOT="$MOTIVATION_DIR/results/$RUN_TAG"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HOME="$MOTIVATION_DIR/cache/huggingface"
export XDG_CACHE_HOME="$MOTIVATION_DIR/cache/xdg"
export MPLCONFIGDIR="$MOTIVATION_DIR/cache/matplotlib"
export TMPDIR="$MOTIVATION_DIR/cache/tmp"
if [[ "$DRY_RUN" != 1 ]]; then
  mkdir -p "$TMPDIR" "$MOTIVATION_DIR/logs/$RUN_TAG"
fi

run() {
  local label="$1"
  shift
  printf '[command] '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@" 2>&1 | tee -a "$MOTIVATION_DIR/logs/$RUN_TAG/$label.log"
  fi
}

probe() {
  local job="$1" task="$2" variant="$3" count="$4"
  local extra=()
  if [[ -n "${TEXT_JSON:-}" && "$job" == text ]]; then
    extra+=(--text-json "$TEXT_JSON")
  fi
  run "${job}_${task}_${variant}" "$PYTHON_BIN" -u "$MOTIVATION_DIR/scripts/run_probe.py" \
    "$job" --task "$task" --variant "$variant" --samples "$count" --seed "$SEED" \
    --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --run-root "$RUN_ROOT" "${extra[@]}"
}

dispatch() {
  local action="$1" task count variant
  case "$action" in
    e1)
      probe text ad bf16 "${TEXT_N:-512}"
      for task in $TASKS; do
        count=256; [[ "$task" != ad ]] || count=512
        probe conditional "$task" bf16 "${N:-$count}"
      done
      ;;
    e2)
      for task in $TASKS; do
        count=256; [[ "$task" != ad ]] || count=512
        for variant in bf16 w8a8 w4a8; do probe conditional "$task" "$variant" "${N:-$count}"; done
      done
      ;;
    e3)
      for task in $TASKS; do
        count=128; [[ "$task" != ad ]] || count=256
        for variant in bf16 w8a8 w4a8; do probe beam "$task" "$variant" "${BEAM_N:-$count}"; done
      done
      ;;
    preflight)
      for task in $TASKS; do
        for variant in bf16 w8a8 w4a8; do probe preflight "$task" "$variant" 1; done
      done
      ;;
    plot)
      for task in $TASKS; do
        run "plot_$task" "$PYTHON_BIN" -u "$MOTIVATION_DIR/scripts/plot_motivation.py" --run-root "$RUN_ROOT" --task "$task"
      done
      ;;
    all)
      dispatch preflight
      dispatch e1
      dispatch e2
      dispatch e3
      echo "All probes complete. Run this script with 'plot' to generate figures."
      ;;
    *) echo "Usage: bash $0 {preflight|e1|e2|e3|all|plot}" >&2; exit 2 ;;
  esac
}
dispatch "${1:-all}"
