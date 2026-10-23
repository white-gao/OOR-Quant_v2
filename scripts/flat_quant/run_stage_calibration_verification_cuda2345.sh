#!/usr/bin/env bash
# Four independent matched pilots; no full-generation benchmark or source overwrite.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/artifacts/results/flat_quant/stage_calibration_verification/$(date -u +%Y%m%d_%H%M%S)}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-2,3,4,5}"
TAGS=(ad_w4a8 ad_w8a8 video_w4a8 video_w8a8)
if [[ ${#GPU_LIST[@]} != 4 ]]; then
    echo 'Exactly four GPUs are required.' >&2
    exit 2
fi
declare -A seen_gpu=()
for gpu in "${GPU_LIST[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_gpu[$gpu]:-}" ]]; then
        echo "Invalid or duplicate GPU index: $gpu" >&2
        exit 2
    fi
    seen_gpu[$gpu]=1
    memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if (( memory_used > 1024 )); then
        echo "GPU $gpu is not idle ($memory_used MiB); refusing to start." >&2
        exit 2
    fi
done
EXTRA_ARGS=()
LOG_SUFFIX=""
if [[ "${DIAGNOSE_TRAIN:-0}" == 1 ]]; then
    [[ -d "$RUN_ROOT" ]] || { echo 'Set RUN_ROOT to a completed pilot.' >&2; exit 2; }
    EXTRA_ARGS+=(--diagnose_train)
    LOG_SUFFIX=".train_diagnostics"
elif [[ -e "$RUN_ROOT" ]]; then
    echo "Output already exists: $RUN_ROOT; choose a fresh RUN_ROOT." >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for i in 0 1 2 3; do
    tag="${TAGS[$i]}"
    task="${tag%_*}"
    precision="${tag#*_}"
    source_dir="$REPO_ROOT/artifacts/results/flat_quant/official_fake_quant/1p7b_${task}_${precision}_pc_prefix128_final512_heldout512_epoch15/final_train512_heldout512/1.7B/$task/flatquant_calibration"
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" "$PYTHON_BIN" -u -m flat_quant.verify_stage_calibration \
        --task "$task" --checkpoint_dir "$source_dir" --output_dir "$RUN_ROOT/$tag" \
        --layer "${LAYER:-26}" --samples "${SAMPLES:-128}" \
        --heldout_samples "${HELDOUT_SAMPLES:-128}" --heldout_offset "${HELDOUT_OFFSET:-768}" \
        --epochs "${EPOCHS:-5}" --seed 42 "${EXTRA_ARGS[@]}" \
        > "$RUN_ROOT/$tag$LOG_SUFFIX.log" 2>&1 &
    pids+=("$!")
    echo "[started] $tag GPU=${GPU_LIST[$i]} pid=$! log=$RUN_ROOT/$tag$LOG_SUFFIX.log"
done
status=0
for i in 0 1 2 3; do
    if wait "${pids[$i]}"; then
        echo "[complete] ${TAGS[$i]}"
    else
        echo "[failed] ${TAGS[$i]}; see $RUN_ROOT/${TAGS[$i]}$LOG_SUFFIX.log" >&2
        status=1
    fi
done
trap - EXIT
echo "[results] $RUN_ROOT"
exit "$status"
