#!/usr/bin/env bash
set -euo pipefail

# Wait until physical GPUs 0 and 1 are simultaneously idle, then run only the
# two unfinished Product experiments: SmoothQuant followed by OmniQuant.
#
# "Idle" means, for every monitored GPU:
#   no compute PID reported by nvidia-smi
#   memory.used <= IDLE_MEMORY_MIB (default 1024 MiB)
#   utilization.gpu <= IDLE_UTIL_PERCENT (default 10%)
# The condition must hold for REQUIRED_IDLE_POLLS consecutive polls.
#
# Usage:
#   bash scripts/fake_quant/wait_cuda01_then_run_fp8w8a8_six_baselines.sh
#
# Optional:
#   POLL_SECONDS=60 REQUIRED_IDLE_POLLS=3 #     bash scripts/fake_quant/wait_cuda01_then_run_fp8w8a8_six_baselines.sh
#   OVERWRITE=1 #     bash scripts/fake_quant/wait_cuda01_then_run_fp8w8a8_six_baselines.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-0,1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
REQUIRED_IDLE_POLLS="${REQUIRED_IDLE_POLLS:-2}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1024}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-10}"
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-${REPO_ROOT}/scripts/fake_quant/run_1p7b_ad_product_fp8w8a8_six_baselines_cuda01.sh}"

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} != 2 )); then
    echo "This monitor requires exactly two GPU IDs; got ${GPUS}." >&2
    exit 2
fi
if [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "GPU IDs must be distinct; got ${GPUS}." >&2
    exit 2
fi
for value_name in POLL_SECONDS REQUIRED_IDLE_POLLS IDLE_MEMORY_MIB IDLE_UTIL_PERCENT; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer; got ${value}." >&2
        exit 2
    fi
done
if (( POLL_SECONDS <= 0 || REQUIRED_IDLE_POLLS <= 0 )); then
    echo "POLL_SECONDS and REQUIRED_IDLE_POLLS must be positive." >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found." >&2
    exit 1
fi
if [[ ! -f "${EXPERIMENT_SCRIPT}" ]]; then
    echo "Experiment script does not exist: ${EXPERIMENT_SCRIPT}" >&2
    exit 1
fi

consecutive_idle=0
echo "[gpu wait] monitoring physical_gpus=${GPUS} interval=${POLL_SECONDS}s"
echo "[gpu wait] idle_thresholds memory<=${IDLE_MEMORY_MIB}MiB util<=${IDLE_UTIL_PERCENT}% no_compute_pids"
echo "[gpu wait] required_consecutive_polls=${REQUIRED_IDLE_POLLS}"
echo "[gpu wait] experiment=${EXPERIMENT_SCRIPT}"

while true; do
    all_idle=1
    status_parts=()

    for gpu in "${GPU_IDS[@]}"; do
        metrics=""
        if ! metrics="$(nvidia-smi -i "${gpu}"             --query-gpu=memory.used,utilization.gpu             --format=csv,noheader,nounits 2>/dev/null)"; then
            status_parts+=("gpu${gpu}[query_failed]")
            all_idle=0
            continue
        fi

        IFS=',' read -r memory_used utilization <<< "${metrics}"
        memory_used="${memory_used//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        if [[ ! "${memory_used}" =~ ^[0-9]+$ || ! "${utilization}" =~ ^[0-9]+$ ]]; then
            status_parts+=("gpu${gpu}[invalid_metrics=${metrics}]")
            all_idle=0
            continue
        fi

        process_count=0
        process_query_ok=1
        process_output=""
        if ! process_output="$(nvidia-smi -i "${gpu}"             --query-compute-apps=pid             --format=csv,noheader,nounits 2>/dev/null)"; then
            process_query_ok=0
        else
            while IFS= read -r pid; do
                pid="${pid//[[:space:]]/}"
                if [[ "${pid}" =~ ^[0-9]+$ ]]; then
                    process_count=$((process_count + 1))
                fi
            done <<< "${process_output}"
        fi

        state="idle"
        if (( process_query_ok == 0 )); then
            state="pid_query_failed"
            all_idle=0
        elif (( memory_used > IDLE_MEMORY_MIB || utilization > IDLE_UTIL_PERCENT || process_count > 0 )); then
            state="busy"
            all_idle=0
        fi
        status_parts+=(
            "gpu${gpu}[mem=${memory_used}MiB,util=${utilization}%,pids=${process_count},${state}]"
        )
    done

    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[gpu wait] ${timestamp} ${status_parts[*]}"

    if (( all_idle == 1 )); then
        consecutive_idle=$((consecutive_idle + 1))
        echo "[gpu wait] simultaneous_idle=${consecutive_idle}/${REQUIRED_IDLE_POLLS}"
        if (( consecutive_idle >= REQUIRED_IDLE_POLLS )); then
            echo "[gpu wait] GPUs are idle; starting the two remaining Product experiments."
            export GPUS
            export TASKS="${TASKS:-product}"
            export METHODS="${METHODS:-smoothquant omniquant}"
            echo "[gpu wait] selected TASKS='${TASKS}' METHODS='${METHODS}'"
            exec bash "${EXPERIMENT_SCRIPT}"
        fi
    else
        consecutive_idle=0
    fi

    sleep "${POLL_SECONDS}"
done
