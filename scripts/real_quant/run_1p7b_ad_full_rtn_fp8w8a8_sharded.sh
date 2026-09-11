#!/usr/bin/env bash
set -euo pipefail

# Real FP8 W8A8 RTN evaluation. Each GPU loads one model and generates one
# deterministic round-robin shard; the launcher then restores source order and
# computes benchmark metrics once on the merged full-AD result.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
BENCHMARK_DATA_DIR="${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${BENCHMARK_DATA_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/real_quant/recommender/rtn_w8a8/1p7b_ad_full}"
GPUS="${GPUS:-6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
NUM_SHARDS="${#GPU_IDS[@]}"
if (( NUM_SHARDS < 2 )); then
    echo "GPUS must contain at least two comma-separated GPU IDs." >&2
    exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
    if [[ -z "${gpu}" || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "GPUS contains an empty or duplicate GPU ID: ${GPUS}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
done

OUTPUT_ROOT="$(readlink -m "${OUTPUT_DIR}")"
SHARD_ROOT="${OUTPUT_ROOT}.shards"
mkdir -p "${OUTPUT_ROOT}" "${SHARD_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    WRITE_ARGS+=(--overwrite)
fi

declare -a SHARD_PIDS=()
cleanup() {
    trap - INT TERM
    local pid
    for pid in "${SHARD_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

for shard_id in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[${shard_id}]}"
    shard_label="$(printf 'shard_%03d_of_%03d' "${shard_id}" "${NUM_SHARDS}")"
    log_dir="${SHARD_ROOT}/${shard_label}"
    log_file="${log_dir}/run.log"
    mkdir -p "${log_dir}"
    command=(
        "${PYTHON_BIN}" -u -m real_quant.naive_w8a8.run_hf_naive_w8a8
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --output_dir "${OUTPUT_ROOT}"
        --task ad
        --sample_size full
        --device cuda:0
        --batch_size "${BATCH_SIZE}"
        --weight_quant_mode minmax
        --eval_num_shards "${NUM_SHARDS}"
        --eval_shard_id "${shard_id}"
        "${WRITE_ARGS[@]}"
    )
    echo "[real sharded eval] ${shard_label} physical_gpu=${gpu} log=${log_file}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" > >(tee "${log_file}") 2>&1 &
    SHARD_PIDS[${shard_id}]=$!
done

if [[ "${DRY_RUN}" != "1" ]]; then
    failed=0
    for shard_id in "${!SHARD_PIDS[@]}"; do
        if ! wait "${SHARD_PIDS[${shard_id}]}"; then
            echo "Evaluation shard ${shard_id}/${NUM_SHARDS} failed; inspect ${SHARD_ROOT}." >&2
            failed=1
        fi
    done
    if [[ "${failed}" != "0" ]]; then
        exit 1
    fi
fi

MERGE_COMMAND=(
    "${PYTHON_BIN}" -u -m real_quant.merge_eval_shards
    --output_dir "${OUTPUT_ROOT}"
    --data_dir "${DATA_DIR}"
    --task ad
    --num_shards "${NUM_SHARDS}"
    "${WRITE_ARGS[@]}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${MERGE_COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${MERGE_COMMAND[@]}" 2>&1 | tee "${OUTPUT_ROOT}/merge_and_evaluate.log"
echo "[real sharded eval] completed output=${OUTPUT_ROOT}"
echo "[real sharded eval] shard files retained at ${SHARD_ROOT}"
