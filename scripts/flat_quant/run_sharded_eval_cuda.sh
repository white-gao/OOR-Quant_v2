#!/usr/bin/env bash
set -euo pipefail

# Run one independent evaluation shard per GPU, merge the generated samples,
# then calculate benchmark metrics exactly once on the merged output.
#
# Example (the Python environment must already be activated):
#   GPUS=0,1,2,3 \
#   OUTPUT_DIR=artifacts/results/fake_quant/my_eval \
#   bash scripts/fake_quant/run_sharded_eval_cuda.sh \
#     --mode omniquant \
#     --omni_load_checkpoint_dir artifacts/results/.../omniquant_calibration
#
# Positional arguments are forwarded to flat_quant.run_m1_onerec_ad. Options
# owned by this launcher (task/model/data/output/device/shard/evaluate) must be
# provided through the environment variables below instead.

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
TASK="${TASK:-ad}"
GPUS="${GPUS:-0,1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR is required." >&2
    exit 2
fi

case "${TASK}" in
    ad|product|video|label_pred) ;;
    *)
        echo "TASK must be one of ad, product, video, or label_pred; got ${TASK}." >&2
        exit 2
        ;;
esac

for argument in "$@"; do
    case "${argument}" in
        --task|--task=*|--model_path|--model_path=*|--data_dir|--data_dir=*|\
        --output_dir|--output_dir=*|--device|--device=*|--eval_num_shards|\
        --eval_num_shards=*|--eval_shard_id|--eval_shard_id=*|--evaluate)
            echo "Argument ${argument} is owned by the sharded launcher; use its environment variable instead." >&2
            exit 2
            ;;
    esac
done

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
NUM_SHARDS="${#GPU_IDS[@]}"
if (( NUM_SHARDS < 2 )); then
    echo "GPUS must contain at least two comma-separated GPU IDs." >&2
    exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
    if [[ -z "${gpu}" ]]; then
        echo "GPUS contains an empty GPU ID: ${GPUS}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "GPUS contains duplicate ID ${gpu}." >&2
        exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
done

if [[ "${OUTPUT_DIR}" = /* ]]; then
    OUTPUT_ROOT="${OUTPUT_DIR}"
else
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_DIR}"
fi
SHARD_ROOT="${OUTPUT_ROOT}.shards"
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${OUTPUT_ROOT}" "${SHARD_ROOT}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUNNER_ARGS=("$@")
WRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
    WRITE_ARGS+=(--overwrite)
fi

print_command() {
    local gpu="$1"
    shift
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "$@"
    printf '\n'
}

run_shard() {
    local shard_id="$1"
    local gpu="$2"
    local shard_label
    shard_label="$(printf 'shard_%03d_of_%03d' "${shard_id}" "${NUM_SHARDS}")"
    local log_dir="${SHARD_ROOT}/${shard_label}"
    local log_file="${log_dir}/run.log"
    if [[ "${DRY_RUN}" != "1" ]]; then
        mkdir -p "${log_dir}"
    fi

    local command=(
        "${PYTHON_BIN}" -u -m flat_quant.run_m1_onerec_ad
        --task "${TASK}"
        --model_path "${MODEL_PATH}"
        --data_dir "${DATA_DIR}"
        --output_dir "${OUTPUT_ROOT}"
        --device cuda:0
        --eval_num_shards "${NUM_SHARDS}"
        --eval_shard_id "${shard_id}"
        "${WRITE_ARGS[@]}"
        "${RUNNER_ARGS[@]}"
    )

    echo "[sharded eval] ${shard_label} physical_gpu=${gpu} log=${log_file}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command "${gpu}" "${command[@]}"
        return
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" 2>&1 | tee "${log_file}"
}

declare -a SHARD_PIDS=()
for shard_id in "${!GPU_IDS[@]}"; do
    run_shard "${shard_id}" "${GPU_IDS[${shard_id}]}" &
    SHARD_PIDS[${shard_id}]=$!
done

failed=0
for shard_id in "${!SHARD_PIDS[@]}"; do
    if ! wait "${SHARD_PIDS[${shard_id}]}"; then
        echo "Evaluation shard ${shard_id}/${NUM_SHARDS} failed; see ${SHARD_ROOT} for logs." >&2
        failed=1
    fi
done
if [[ "${failed}" != "0" ]]; then
    exit 1
fi

MERGE_COMMAND=(
    "${PYTHON_BIN}" -u -m flat_quant.merge_eval_shards
    --output_dir "${OUTPUT_ROOT}"
    --data_dir "${DATA_DIR}"
    --task "${TASK}"
    --num_shards "${NUM_SHARDS}"
    "${WRITE_ARGS[@]}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${MERGE_COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${MERGE_COMMAND[@]}" 2>&1 | tee "${OUTPUT_ROOT}/merge_and_evaluate.log"
echo "[sharded eval] completed output=${OUTPUT_ROOT}"
echo "[sharded eval] shard files retained at ${SHARD_ROOT}"
