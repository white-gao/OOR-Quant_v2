#!/usr/bin/env bash
set -euo pipefail

# Serial Video-domain Qwen3-1.7B W4A8 per-output-channel baselines:
#   1. RTN
#   2. SmoothQuant (alpha=0.4, calibration=128)
#   3. OmniQuant-LWC MSE (prefix=128, final=1024, epochs=20)
#   4. FlatQuant MSE (prefix=128, final train/held-out=512/512, epochs=15)
#
# Calibration runs on the first GPU. Full Video evaluation is sharded across
# both GPUs. Once eval_results.json is safely written, large generated samples
# and shard copies are removed by default; set CLEAN_EVAL_ARTIFACTS=0 to retain
# them.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/1.7B}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/onerec_data/benchmark_data}"
PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

GPUS="${GPUS:-6,7}"
RUN_BASELINES="${RUN_BASELINES:-rtn smoothquant omniquant flatquant}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.4}"
OMNI_EPOCHS="${OMNI_EPOCHS:-20}"
FLAT_EPOCHS="${FLAT_EPOCHS:-15}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_EVAL_ARTIFACTS="${CLEAN_EVAL_ARTIFACTS:-1}"

for pair in "OVERWRITE:${OVERWRITE}" "DRY_RUN:${DRY_RUN}" "CLEAN_EVAL_ARTIFACTS:${CLEAN_EVAL_ARTIFACTS}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
done
if ! "${PYTHON_BIN}" -c 'import math,sys; x=float(sys.argv[1]); sys.exit(not (math.isfinite(x) and 0.0 <= x <= 1.0))' "${SMOOTHQUANT_ALPHA}"; then
    echo "SMOOTHQUANT_ALPHA must be finite and in [0, 1]; got ${SMOOTHQUANT_ALPHA}." >&2
    exit 2
fi

GPU_STRING="${GPUS//[[:space:]]/}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_STRING}"
if (( ${#GPU_IDS[@]} != 2 )) || [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "This launcher requires exactly two distinct GPUs; got GPUS=${GPUS}." >&2
    exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: ${gpu_id}." >&2
        exit 2
    fi
done

want_rtn=0
want_smoothquant=0
want_omniquant=0
want_flatquant=0
declare -A SEEN_BASELINES=()
for baseline in ${RUN_BASELINES}; do
    if [[ -n "${SEEN_BASELINES[${baseline}]:-}" ]]; then
        echo "Duplicate RUN_BASELINES entry: ${baseline}." >&2
        exit 2
    fi
    SEEN_BASELINES["${baseline}"]=1
    case "${baseline}" in
        rtn) want_rtn=1 ;;
        smoothquant) want_smoothquant=1 ;;
        omniquant) want_omniquant=1 ;;
        flatquant) want_flatquant=1 ;;
        *) echo "Unknown RUN_BASELINES entry: ${baseline}." >&2; exit 2 ;;
    esac
done
if (( want_rtn == 0 && want_smoothquant == 0 && want_omniquant == 0 && want_flatquant == 0 )); then
    echo "RUN_BASELINES must select at least one baseline." >&2
    exit 2
fi

RTN_OUTPUT_DIR="${RTN_OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_video_w4a8_pc_rtn_${EVAL_SAMPLE_SIZE}}"
SQ_OUTPUT_DIR="${SQ_OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_video_w4a8_pc_smoothquant_alpha0p4_calib128_${EVAL_SAMPLE_SIZE}}"
OMNI_ROOT="${OMNI_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_video_w4a8_pc_omniquant_lwc_mse_prefix128_final1024_epoch${OMNI_EPOCHS}}"
FLAT_ROOT="${FLAT_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_video_w4a8_pc_prefix128_final512_heldout512_epoch${FLAT_EPOCHS}}"
OMNI_EVAL_DIR="${OMNI_ROOT}/video_${EVAL_SAMPLE_SIZE}_eval"
FLAT_EVAL_DIR="${FLAT_ROOT}/video_${EVAL_SAMPLE_SIZE}_eval"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cleanup_eval_artifacts() {
    local output_dir="$1"
    local label="$2"
    if [[ "${CLEAN_EVAL_ARTIFACTS}" != "1" ]]; then
        return
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[${label}] dry-run: would remove generated samples and ${output_dir}.shards after successful merge."
        return
    fi
    if [[ ! -s "${output_dir}/eval_results.json" ]]; then
        echo "[${label}] refusing cleanup because eval_results.json is missing." >&2
        return 4
    fi
    find "${output_dir}" -type f \
        \( -name 'test_generated.json' -o -name 'test_generated.json.debug' \
           -o -name 'debug_pid.json' -o -name 'debug_sid.json' \) -delete
    local shard_dir="${output_dir}.shards"
    if [[ -d "${shard_dir}" ]]; then
        find "${shard_dir}" -depth -delete
    fi
    echo "[${label}] retained eval_results/config/logs; removed generated samples and shards."
}

run_simple_eval() {
    local label="$1"
    local method="$2"
    local output_dir="$3"
    local method_args=()

    if [[ -s "${output_dir}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[${label}] complete eval_results.json found; skipping."
        cleanup_eval_artifacts "${output_dir}" "${label}"
        return
    fi
    case "${method}" in
        rtn)
            method_args+=(--mode baseline_qdq)
            ;;
        smoothquant)
            method_args+=(
                --mode smoothquant_w8a8
                --smoothquant_alpha "${SMOOTHQUANT_ALPHA}"
                --calib_sample_size 128
            )
            ;;
        *)
            echo "Unsupported simple baseline: ${method}." >&2
            return 2
            ;;
    esac

    echo "[${label}] evaluation_gpus=${GPUS} output=${output_dir}"
    env \
        TASK=video \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        OUTPUT_DIR="${output_dir}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_sharded_eval_cuda.sh \
            "${method_args[@]}" \
            --weight_quant_format fp4_e2m1 \
            --activation_quant_format fp8_e4m3fn \
            --weight_quant_scheme symmetric \
            --weight_group_size 0 \
            --eval_sample_size "${EVAL_SAMPLE_SIZE}"
    cleanup_eval_artifacts "${output_dir}" "${label}"
}

echo "[protocol] Qwen3-1.7B Video W4A8 per-output-channel baselines"
echo "[protocol] order=${RUN_BASELINES}"
echo "[protocol] FP4-E2M1-W / dynamic per-token FP8-E4M3FN-A; symmetric weight QDQ"
echo "[protocol] SmoothQuant alpha=${SMOOTHQUANT_ALPHA} calib=128"
echo "[protocol] OmniQuant LWC-only prefix=128 final=1024 epochs=${OMNI_EPOCHS}"
echo "[protocol] FlatQuant prefix=128 final_train=512 heldout=512 epochs=${FLAT_EPOCHS}"
echo "[protocol] training_gpu=${GPU_IDS[0]} evaluation_gpus=${GPUS}"
echo "[protocol] cleanup_after_success=${CLEAN_EVAL_ARTIFACTS}"

if (( want_rtn == 1 )); then
    echo
    echo "======================= [1/4 RTN] ======================="
    run_simple_eval "RTN" rtn "${RTN_OUTPUT_DIR}"
fi

if (( want_smoothquant == 1 )); then
    echo
    echo "================== [2/4 SmoothQuant] ==================="
    run_simple_eval "SmoothQuant" smoothquant "${SQ_OUTPUT_DIR}"
fi

if (( want_omniquant == 1 )); then
    echo
    echo "=================== [3/4 OmniQuant] ===================="
    env \
        TASK=video \
        QUANT_LABEL=w4a8 \
        WEIGHT_QUANT_FORMAT=fp4_e2m1 \
        ACTIVATION_QUANT_FORMAT=fp8_e4m3fn \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        RESULTS_ROOT="${OMNI_ROOT}" \
        PREFIX_CALIB_SAMPLES=128 \
        FINAL_CALIB_SAMPLES=1024 \
        EPOCHS="${OMNI_EPOCHS}" \
        EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/fake_quant/run_1p7b_product_w4a8_pc_omniquant_mse_cuda67.sh
    cleanup_eval_artifacts "${OMNI_EVAL_DIR}" "OmniQuant"
fi

if (( want_flatquant == 1 )); then
    echo
    echo "==================== [4/4 FlatQuant] ==================="
    flat_run_eval=1
    if [[ -s "${FLAT_EVAL_DIR}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        flat_run_eval=0
        echo "[FlatQuant] complete evaluation found; checking training checkpoints only."
    fi
    env \
        TASK=video \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        RUN_ROOT="${FLAT_ROOT}" \
        WEIGHT_QUANT_FORMAT=fp4_e2m1 \
        ACTIVATION_QUANT_FORMAT=fp8_e4m3fn \
        QUANT_LABEL=w4a8 \
        PREFIX_CALIB_SAMPLES=128 \
        FINAL_CALIB_SAMPLES=1024 \
        FINAL_TRAIN_SAMPLES=512 \
        FINAL_HELDOUT_SAMPLES=512 \
        EPOCHS="${FLAT_EPOCHS}" \
        RUN_EVAL="${flat_run_eval}" \
        EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" \
        OVERWRITE="${OVERWRITE}" \
        DRY_RUN="${DRY_RUN}" \
        bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh
    cleanup_eval_artifacts "${FLAT_EVAL_DIR}" "FlatQuant"
fi

echo
echo "[done] RTN:         ${RTN_OUTPUT_DIR}/eval_results.json"
echo "[done] SmoothQuant: ${SQ_OUTPUT_DIR}/eval_results.json"
echo "[done] OmniQuant:   ${OMNI_EVAL_DIR}/eval_results.json"
echo "[done] FlatQuant:   ${FLAT_EVAL_DIR}/eval_results.json"
