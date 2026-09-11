#!/usr/bin/env bash
set -euo pipefail

# Serial Video-domain Qwen3-1.7B baselines:
# SmoothQuant, OmniQuant-LWC MSE, and FlatQuant MSE.
# W4A8 is the default; set QUANT_LABEL=w8a8 for the W8A8 counterpart.

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
QUANT_LABEL="${QUANT_LABEL:-w4a8}"
RUN_BASELINES="${RUN_BASELINES:-smoothquant omniquant flatquant}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.4}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FLAT_EPOCHS="${FLAT_EPOCHS:-15}"
OMNI_EPOCHS="${OMNI_EPOCHS:-20}"

case "${QUANT_LABEL}" in
    w4a8)
        WEIGHT_QUANT_FORMAT="fp4_e2m1"
        ACTIVATION_QUANT_FORMAT="fp8_e4m3fn"
        ;;
    w8a8)
        WEIGHT_QUANT_FORMAT="fp8_e4m3fn"
        ACTIVATION_QUANT_FORMAT="fp8_e4m3fn"
        ;;
    *)
        echo "QUANT_LABEL must be w4a8 or w8a8; got ${QUANT_LABEL}." >&2
        exit 2
        ;;
esac
for pair in "OVERWRITE:${OVERWRITE}" "DRY_RUN:${DRY_RUN}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1; got ${value}." >&2
        exit 2
    fi
done

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

want_smoothquant=0
want_omniquant=0
want_flatquant=0
for baseline in ${RUN_BASELINES}; do
    case "${baseline}" in
        smoothquant) want_smoothquant=1 ;;
        omniquant) want_omniquant=1 ;;
        flatquant) want_flatquant=1 ;;
        *) echo "Unknown RUN_BASELINES entry: ${baseline}." >&2; exit 2 ;;
    esac
done
if (( want_smoothquant == 0 && want_omniquant == 0 && want_flatquant == 0 )); then
    echo "RUN_BASELINES must select at least one baseline." >&2
    exit 2
fi

SQ_OUTPUT_DIR="${SQ_OUTPUT_DIR:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_video_${QUANT_LABEL}_pc_smoothquant_alpha0p4_calib128_${EVAL_SAMPLE_SIZE}}"
OMNI_ROOT="${OMNI_ROOT:-${ARTIFACTS_ROOT}/results/fake_quant/recommender/1p7b_video_${QUANT_LABEL}_pc_omniquant_lwc_mse_prefix128_final1024_epoch${OMNI_EPOCHS}}"
FLAT_ROOT="${FLAT_ROOT:-${ARTIFACTS_ROOT}/results/flat_quant/official_fake_quant/1p7b_video_${QUANT_LABEL}_pc_prefix128_final512_heldout512_epoch${FLAT_EPOCHS}}"

echo "[protocol] Qwen3-1.7B Video ${QUANT_LABEL} baselines"
echo "[protocol] order=${RUN_BASELINES}; weight=${WEIGHT_QUANT_FORMAT} activation=${ACTIVATION_QUANT_FORMAT}"
echo "[protocol] SmoothQuant alpha=${SMOOTHQUANT_ALPHA} calib=128"
echo "[protocol] OmniQuant LWC-only prefix=128 final=1024 epochs=${OMNI_EPOCHS}"
echo "[protocol] FlatQuant prefix=128 final_train=512 heldout=512 epochs=${FLAT_EPOCHS}"
echo "[protocol] training_gpu=${GPU_IDS[0]} evaluation_gpus=${GPUS}"

if (( want_smoothquant == 1 )); then
    echo
    echo "==================== [SmoothQuant] ===================="
    if [[ -s "${SQ_OUTPUT_DIR}/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        echo "[SmoothQuant] complete result found; skipping."
    else
        env \
            TASK=video \
            GPUS="${GPUS}" \
            MODEL_PATH="${MODEL_PATH}" \
            DATA_DIR="${DATA_DIR}" \
            OUTPUT_DIR="${SQ_OUTPUT_DIR}" \
            PYTHON_BIN="${PYTHON_BIN}" \
            OVERWRITE="${OVERWRITE}" \
            DRY_RUN="${DRY_RUN}" \
            bash scripts/fake_quant/run_sharded_eval_cuda.sh \
                --mode smoothquant_w8a8 \
                --weight_quant_format "${WEIGHT_QUANT_FORMAT}" \
                --activation_quant_format "${ACTIVATION_QUANT_FORMAT}" \
                --weight_quant_scheme symmetric \
                --weight_group_size 0 \
                --smoothquant_alpha "${SMOOTHQUANT_ALPHA}" \
                --calib_sample_size 128 \
                --eval_sample_size "${EVAL_SAMPLE_SIZE}"
    fi
fi

if (( want_omniquant == 1 )); then
    echo
    echo "==================== [OmniQuant] ====================="
    env \
        TASK=video \
        QUANT_LABEL="${QUANT_LABEL}" \
        WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT}" \
        ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT}" \
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
fi

if (( want_flatquant == 1 )); then
    echo
    echo "===================== [FlatQuant] ====================="
    flat_run_eval=1
    if [[ -s "${FLAT_ROOT}/video_${EVAL_SAMPLE_SIZE}_eval/eval_results.json" && "${OVERWRITE}" != "1" ]]; then
        flat_run_eval=0
        echo "[FlatQuant] complete evaluation found; training/checkpoint check only."
    fi
    env \
        TASK=video \
        GPUS="${GPUS}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        RUN_ROOT="${FLAT_ROOT}" \
        WEIGHT_QUANT_FORMAT="${WEIGHT_QUANT_FORMAT}" \
        ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT}" \
        QUANT_LABEL="${QUANT_LABEL}" \
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
fi

echo
echo "[done] SmoothQuant: ${SQ_OUTPUT_DIR}/eval_results.json"
echo "[done] OmniQuant: ${OMNI_ROOT}/video_${EVAL_SAMPLE_SIZE}_eval/eval_results.json"
echo "[done] FlatQuant: ${FLAT_ROOT}/video_${EVAL_SAMPLE_SIZE}_eval/eval_results.json"
