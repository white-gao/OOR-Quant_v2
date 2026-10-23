#!/usr/bin/env bash
# BF16 only. All three tasks run serially on one configurable GPU.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
STAGE="${STAGE:-all}"
TASKS="${TASKS:-interactive label_pred item_understand}"
SAMPLE_SIZE="${SAMPLE_SIZE:-full}"
CHECK_ONLY="${CHECK_ONLY:-0}"
ALLOW_JUDGE="${ALLOW_JUDGE:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

common=(--stage "$STAGE" --sample_size "$SAMPLE_SIZE" --device cuda:0
        --chunk_size "${CHUNK_SIZE:-128}")
[[ -z "${MODEL_PATH:-}" ]] || common+=(--model_path "$MODEL_PATH")
[[ -z "${DATA_DIR:-}" ]] || common+=(--data_dir "$DATA_DIR")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  common+=(--output_dir "$OUTPUT_DIR")
elif [[ "$SAMPLE_SIZE" != full ]]; then
  # Smoke tests must not occupy the full-evaluation output directory.
  common+=(--output_dir "${OOR_QUANT_ARTIFACTS:-$REPO_ROOT/artifacts}/results/recif/bf16_smoke_$SAMPLE_SIZE")
fi
[[ "$CHECK_ONLY" != 1 ]] || common+=(--check_only)
if [[ "$ALLOW_JUDGE" == 1 ]]; then
  : "${JUDGE_CONFIG:?Set JUDGE_CONFIG to a Gemini or DeepSeek judge configuration}"
  common+=(--allow_judge --judge_config "$JUDGE_CONFIG")
fi
[[ -z "${BERTSCORE_MODEL:-}" ]] || common+=(--bertscore_model "$BERTSCORE_MODEL")
read -r -a tasks <<< "$TASKS"
if (( ${#tasks[@]} == 0 )); then
  echo "TASKS must not be empty" >&2
  exit 2
fi
# Fail early if a later task's data/config are missing, before loading any model.
for task in "${tasks[@]}"; do
  "$PYTHON_BIN" -m real_quant.full_precision.run_recif_baseline \
    "${common[@]}" --task "$task" --batch_size "${BATCH_SIZE:-1}" --check_only
done
if [[ "$CHECK_ONLY" == 1 ]]; then
  exit 0
fi
for task in "${tasks[@]}"; do
  echo "[run] task=$task GPU=$GPU stage=$STAGE samples=$SAMPLE_SIZE"
  "$PYTHON_BIN" -m real_quant.full_precision.run_recif_baseline \
    "${common[@]}" --task "$task" --batch_size "${BATCH_SIZE:-1}"
done
if [[ "$ALLOW_JUDGE" != 1 && " $TASKS " == *" item_understand "* && "$STAGE" != generate ]]; then
  echo "[pending] item_understand captions are saved; WIP judge scoring is NOT complete. See README."
fi
