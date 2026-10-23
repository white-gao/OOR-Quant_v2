#!/usr/bin/env bash
set -euo pipefail

# OneRec-8B AD/Product: shared BF16 references + W8A8 and W4A8 main experiments.
# Historical filename retained so the same launch command now fills in W4A8.
# Default: validate/skip the completed W8A8 runs, then run 10 W4A8 experiments.
# Each precision/task runs RTN -> SQ -> OmniQuant-LWC -> FlatQuant -> RecQuant.
# RecQuant = FlatQuant + final-block ABC CE + 0.3 boundary, joint transforms.
# Protocol matches the historical 1.7B runs, NOT equal budgets across methods.
# Training uses GPUS[0]; evaluation uses one independent replica per GPU.
#
# bash scripts/fake_quant/run_8b_ad_product_w8a8_main_cuda67.sh
# DRY_RUN=1 bash scripts/fake_quant/run_8b_ad_product_w8a8_main_cuda67.sh
# PRECISIONS=w4a8 bash scripts/fake_quant/run_8b_ad_product_w8a8_main_cuda67.sh
# TASKS=product METHODS='flatquant recquant' GPUS=2,3 bash ...
# RUN_ROOT retains its old meaning: W8A8 results and shared BF16 references.
# W4A8_RUN_ROOT overrides the separate W4A8 output directory.
# Use PRECISIONS=w4a8 with overwrite flags to leave W8A8 quantized runs alone.
# EVAL_OVERWRITE=1 reruns evaluation without replacing completed training.
# OVERWRITE=1 explicitly replaces training AND evaluation. Partial training
# is NOT resumed inside a layer/epoch: move that stage aside or overwrite it.
# No historical 8B result is silently imported from another output directory.

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/guowei/miniconda3/envs/benchmark/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN=python
fi
MODEL_ROOT="${OOR_QUANT_MODEL_ROOT:-/root/dataDisk/guowei/models}"
DATA_ROOT="${OOR_QUANT_DATA_ROOT:-/root/dataDisk/guowei/data}"
ARTIFACTS_ROOT="${OOR_QUANT_ARTIFACTS:-${REPO_ROOT}/artifacts}"
MODEL_PATH="$(realpath -m "${MODEL_PATH:-${MODEL_ROOT}/8B}")"
DATA_DIR="$(realpath -m "${DATA_DIR:-${OOR_QUANT_BENCHMARK_DATA:-${DATA_ROOT}/onerec_data/benchmark_data}}")"
RUN_ROOT="$(realpath -m "${RUN_ROOT:-${ARTIFACTS_ROOT}/results/main_experiments/8b_ad_product_w8a8_pc}")"
W4A8_RUN_ROOT="$(realpath -m "${W4A8_RUN_ROOT:-${ARTIFACTS_ROOT}/results/main_experiments/8b_ad_product_w4a8_pc}")"
MODEL_NAME="$(basename "${MODEL_PATH}")"
GPUS="${GPUS:-6,7}"
TASKS="${TASKS:-ad product}"
METHODS="${METHODS:-bf16 rtn smoothquant omniquant flatquant recquant}"
PRECISIONS="${PRECISIONS:-w8a8 w4a8}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-full}"
OVERWRITE="${OVERWRITE:-0}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-${OVERWRITE}}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_EVAL_ARTIFACTS="${CLEAN_EVAL_ARTIFACTS:-1}"

for pair in "OVERWRITE:${OVERWRITE}" "EVAL_OVERWRITE:${EVAL_OVERWRITE}" \
    "DRY_RUN:${DRY_RUN}" "CLEAN_EVAL_ARTIFACTS:${CLEAN_EVAL_ARTIFACTS}"; do
    if [[ "${pair#*:}" != 0 && "${pair#*:}" != 1 ]]; then
        echo "${pair%%:*} must be 0 or 1." >&2
        exit 2
    fi
done
if [[ "${OVERWRITE}" == 1 && "${EVAL_OVERWRITE}" != 1 ]]; then
    echo "Replacing training requires EVAL_OVERWRITE=1 to avoid stale metrics." >&2
    exit 2
fi
if [[ "${EVAL_SAMPLE_SIZE}" != full && ! "${EVAL_SAMPLE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_SAMPLE_SIZE must be full or a positive integer." >&2
    exit 2
fi

GPUS="${GPUS//[[:space:]]/}"
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
    echo "GPUS must list at least two comma-separated numeric GPU IDs." >&2
    exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
declare -A SEEN_GPUS=() SELECTED_TASKS=() SELECTED_METHODS=() SELECTED_PRECISIONS=()
for gpu in "${GPU_IDS[@]}"; do
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "Duplicate GPU ID: ${gpu}." >&2
        exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
done
read -r -a TASK_LIST <<< "${TASKS}"
read -r -a METHOD_LIST <<< "${METHODS}"
read -r -a PRECISION_LIST <<< "${PRECISIONS}"
if (( ${#TASK_LIST[@]} == 0 || ${#METHOD_LIST[@]} == 0 || ${#PRECISION_LIST[@]} == 0 )); then
    echo "TASKS, METHODS and PRECISIONS must not be empty." >&2
    exit 2
fi
for task in "${TASK_LIST[@]}"; do
    case "${task}" in
        ad|product) ;;
        *) echo "Unsupported task: ${task}." >&2; exit 2 ;;
    esac
    if [[ -n "${SELECTED_TASKS[${task}]:-}" ]]; then
        echo "Duplicate task: ${task}." >&2
        exit 2
    fi
    SELECTED_TASKS["${task}"]=1
done
for method in "${METHOD_LIST[@]}"; do
    case "${method}" in
        bf16|rtn|smoothquant|omniquant|flatquant|recquant) ;;
        *) echo "Unsupported method: ${method}." >&2; exit 2 ;;
    esac
    if [[ -n "${SELECTED_METHODS[${method}]:-}" ]]; then
        echo "Duplicate method: ${method}." >&2
        exit 2
    fi
    SELECTED_METHODS["${method}"]=1
done
for precision in "${PRECISION_LIST[@]}"; do
    case "${precision}" in
        w8a8|w4a8) ;;
        *) echo "Unsupported precision: ${precision}." >&2; exit 2 ;;
    esac
    if [[ -n "${SELECTED_PRECISIONS[${precision}]:-}" ]]; then
        echo "Duplicate precision: ${precision}." >&2
        exit 2
    fi
    SELECTED_PRECISIONS["${precision}"]=1
done
if [[ "${RUN_ROOT}" == "${W4A8_RUN_ROOT}" ||
      "${RUN_ROOT}" == "${W4A8_RUN_ROOT}/"* ||
      "${W4A8_RUN_ROOT}" == "${RUN_ROOT}/"* ]]; then
    echo "RUN_ROOT and W4A8_RUN_ROOT must be separate, non-nested directories." >&2
    exit 2
fi

# Read local config, without loading model weights or touching CUDA.
NUM_LAYERS="$("${PYTHON_BIN}" - "${MODEL_PATH}" "${DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path
model, data = map(Path, sys.argv[1:])
config = json.loads((model / 'config.json').read_text())
if config.get('num_hidden_layers') != 36 or config.get('hidden_size') != 4096:
    raise SystemExit('Expected OneRec-8B config: 36 layers and hidden_size=4096.')
if not data.is_dir():
    raise SystemExit(f'Missing benchmark data directory: {data}')
print(config['num_hidden_layers'])
PY
)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
COMMON_ENV=(
    PYTHON_BIN="${PYTHON_BIN}" MODEL_PATH="${MODEL_PATH}" DATA_DIR="${DATA_DIR}"
    GPUS="${GPUS}" NUM_LAYERS="${NUM_LAYERS}" TRAIN_GPU="${GPU_IDS[0]}"
    ACTIVATION_QUANT_FORMAT=fp8_e4m3fn
    PREFIX_CALIB_SAMPLES=128 FINAL_CALIB_SAMPLES=1024
    EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE}" RUN_EVAL=1 SMOKE=0
    OVERWRITE="${OVERWRITE}" EVAL_OVERWRITE="${EVAL_OVERWRITE}" DRY_RUN="${DRY_RUN}"
)
FLAT_ENV=(
    EPOCHS=15 TRANSFORM_LR=5e-3 LWC_LR=5e-2 LAC_LR=5e-2
    WEIGHT_DECAY=0.01 DIAG_ALPHA=0.5 INIT_LWC_LOGIT=4.0 INIT_LAC_LOGIT=4.0
)

validate_result() {
    "${PYTHON_BIN}" - "$1" "${MODEL_NAME}" "$2" "$3" "$4" \
        "${MODEL_PATH}" "${DATA_DIR}" "${EVAL_SAMPLE_SIZE}" <<'PY'
import json
import math
import sys
from pathlib import Path
directory, model, task, method, precision, model_path, data_dir, sample_size = sys.argv[1:]
directory = Path(directory)
path = directory / 'eval_results.json'
metrics = json.loads(path.read_text())[model][task]['test']
expected_samples = {'ad': 27677, 'product': 27910}[task]
if sample_size != 'full':
    expected_samples = min(expected_samples, int(sample_size))
if metrics.get('total_samples') != expected_samples:
    raise SystemExit(f'Expected {expected_samples} test samples: {path}')
for key in ('recall@16', 'recall@32', 'pass@16', 'pass@32'):
    value = float(metrics[key])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise SystemExit(f'Invalid {key}: {path}')
mode = {'bf16': 'full_precision', 'rtn': 'baseline_qdq', 'smoothquant': 'smoothquant_w8a8',
        'omniquant': 'omniquant', 'flatquant': 'flatquant_core', 'recquant': 'flatquant_core'}[method]
config_path = directory / model / task / f'{mode}_config.json'
cfg = json.loads(config_path.read_text())
expected = {'method': mode, 'task': task, 'layers': list(range(36)), 'dtype': 'bfloat16',
            'eval_sample_size': sample_size, 'eval_offset': 0, 'eval_merged': True,
            'split': 'test', 'num_beams': 32, 'num_return_sequences': 32,
            'max_new_tokens': 3, 'seed': 42, 'calibration_only': False}
if method != 'bf16':
    expected.update(weight_quant_format={'w8a8': 'fp8_e4m3fn', 'w4a8': 'fp4_e2m1'}[precision],
                    activation_quant_format='fp8_e4m3fn', weight_group_size=0,
                    weight_quant_scheme='symmetric', act_quant='per_token', act_quant_mode='shared_input')
for key, value in expected.items():
    if cfg.get(key) != value:
        raise SystemExit(f'Config mismatch: {config_path}: {key}={cfg.get(key)!r}, expected {value!r}')
expected_paths = {'eval_data_dir': data_dir}
# Historical runner metadata identifies the model by its 8B directory/key,
# but does not store model_path. Check it when present; do not reject old runs.
if 'model_path' in cfg:
    expected_paths['model_path'] = model_path
for key, expected_path in expected_paths.items():
    if not cfg.get(key) or Path(cfg[key]).resolve() != Path(expected_path).resolve():
        raise SystemExit(f'Config mismatch: {config_path}: {key}')
print(f"[result] {task}/{method}/{precision}: n={metrics['total_samples']} "
      f"Recall@32={100 * metrics['recall@32']:.4f}% path={path}")
PY
}

cleanup_eval() {
    local output_dir="$1"
    [[ "${CLEAN_EVAL_ARTIFACTS}" == 1 ]] || return 0
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "[cleanup] after successful evaluation: generated/debug JSON only; retain metrics/config/logs/checkpoints."
        return
    fi
    # Called only AFTER parsing and validating eval_results.json successfully.
    # Delete only these known large files inside this experiment's eval paths.
    local target
    for target in "${output_dir}" "${output_dir}.shards"; do
        if [[ -d "${target}" ]]; then
            find "${target}" -type f \
                \( -name test_generated.json -o -name test_generated.json.debug \
                   -o -name debug_pid.json -o -name debug_sid.json \) -delete
        fi
    done
    echo "[cleanup] removed generated/debug JSON; kept metrics, configs, logs and checkpoints."
}

run_method() {
    local task="$1" method="$2" precision="$3"
    local precision_root="${RUN_ROOT}" weight_format=fp8_e4m3fn
    if [[ "${precision}" == w4a8 && "${method}" != bf16 ]]; then
        precision_root="${W4A8_RUN_ROOT}"
        weight_format=fp4_e2m1
    fi
    # BF16 is precision-independent and always uses its original W8A8-root path.
    local task_root="${precision_root}/${task}"
    local precision_env=(QUANT_LABEL="${precision}" WEIGHT_QUANT_FORMAT="${weight_format}")
    local output_dir="${task_root}/${method}_${EVAL_SAMPLE_SIZE}_eval"
    local omni_root="${task_root}/omniquant"
    local flat_root="${task_root}/flatquant"
    local rec_root="${task_root}/recquant/abc_boundary0.3_joint"
    local flat_checkpoint="${flat_root}/final_train512_heldout512/${MODEL_NAME}/${task}/flatquant_calibration"
    # Staged FlatQuant's evaluation path is owned by its existing launcher.
    if [[ "${method}" == flatquant ]]; then
        output_dir="${flat_root}/${task}_${EVAL_SAMPLE_SIZE}_eval"
    fi

    if [[ -s "${output_dir}/eval_results.json" && "${EVAL_OVERWRITE}" != 1 ]]; then
        validate_result "${output_dir}" "${task}" "${method}" "${precision}"
        echo "[skip] ${task}/${method}/${precision}: validated completed result; left unchanged."
        return
    fi
    if [[ "${DRY_RUN}" != 1 && "${EVAL_OVERWRITE}" != 1 && \
          ( -e "${output_dir}" || -e "${output_dir}.shards" ) ]]; then
        echo "Partial evaluation: ${output_dir}. Move it aside or use EVAL_OVERWRITE=1 (training is retained)." >&2
        return 3
    fi

    case "${method}" in
        bf16|rtn|smoothquant)
            local args=()
            case "${method}" in
                bf16) args+=(--mode full_precision) ;;
                rtn) args+=(--mode baseline_qdq) ;;
                smoothquant) args+=(--mode smoothquant_w8a8 --smoothquant_alpha 0.4 --calib_sample_size 128) ;;
            esac
            env "${COMMON_ENV[@]}" "${precision_env[@]}" TASK="${task}" OUTPUT_DIR="${output_dir}" \
                OVERWRITE="${EVAL_OVERWRITE}" \
                bash scripts/fake_quant/run_sharded_eval_cuda.sh \
                "${args[@]}" --layers all --weight_quant_format "${weight_format}" \
                --activation_quant_format fp8_e4m3fn --weight_quant_scheme symmetric \
                --weight_group_size 0 --eval_sample_size "${EVAL_SAMPLE_SIZE}"
            ;;
        omniquant)
            env "${COMMON_ENV[@]}" "${precision_env[@]}" TASK="${task}" RESULTS_ROOT="${omni_root}" \
                PREFIX_OUTPUT_DIR="${omni_root}/prefix_calib128" \
                PREFIX_CHECKPOINT_DIR="${omni_root}/prefix_calib128/${MODEL_NAME}/${task}/omniquant_calibration" \
                MSE_CALIB_DIR="${omni_root}/mse_lwc_final1024_calibration" \
                MSE_EVAL_DIR="${output_dir}" EPOCHS=20 LWC_LR=1e-2 INIT_LWC_LOGIT=4.0 \
                bash scripts/fake_quant/run_1p7b_product_w4a8_pc_omniquant_mse_cuda67.sh
            ;;
        flatquant)
            env "${COMMON_ENV[@]}" "${precision_env[@]}" "${FLAT_ENV[@]}" TASK="${task}" RUN_ROOT="${flat_root}" \
                FINAL_TRAIN_SAMPLES=512 FINAL_HELDOUT_SAMPLES=512 \
                bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh
            ;;
        recquant)
            # The shared arm verifies all layer_00..35.pt before training.
            # Selecting RecQuant alone requires the preceding FlatQuant run.
            env "${COMMON_ENV[@]}" "${precision_env[@]}" "${FLAT_ENV[@]}" TASK="${task}" \
                SOURCE_CHECKPOINT_DIR="${flat_checkpoint}" OUTPUT_DIR="${rec_root}" \
                CHECKPOINT_DIR="${rec_root}/${MODEL_NAME}/${task}/flatquant_calibration" \
                EVAL_OUTPUT_DIR="${output_dir}" CALIB_SAMPLES=1024 TRAIN_SAMPLES=1024 \
                LEARN_TRANSFORM=1 LEARN_LAC=1 LFQ_LOSS_WEIGHT=1.0 BOUNDARY_WEIGHT=0.3 \
                BOUNDARY_TOPK=32 BOUNDARY_NEGATIVES=32 BOUNDARY_TIE_THRESHOLD=0.01 BOUNDARY_GAP_SCALE=1.0 \
                bash scripts/flat_quant/run_1p7b_w4a8_pc_flatquant_final_alignment_arm.sh
            ;;
    esac
    if [[ "${DRY_RUN}" != 1 ]]; then
        validate_result "${output_dir}" "${task}" "${method}" "${precision}"
    fi
    cleanup_eval "${output_dir}"
    echo "[output] ${task}/${method}/${precision}: ${output_dir}/eval_results.json"
}

main() {
    echo "[protocol] OneRec-8B; tasks=${TASKS}; methods=${METHODS}; precisions=${PRECISIONS}"
    echo "[protocol] 36 blocks: prefix=0-34, final=35; training_gpu=${GPU_IDS[0]}, evaluation_gpus=${GPUS}"
    echo "[protocol] W8A8=FP8-E4M3FN W/A; W4A8=FP4-E2M1 W + FP8-E4M3FN A"
    echo "[protocol] per-output-channel weights; dynamic per-token shared-input activations; shared BF16 references"
    echo "[protocol] BF16 operators; FP32 QDQ/loss; seed=42; beam=32; return=32; new_tokens=3; eval=${EVAL_SAMPLE_SIZE}"
    echo "[protocol] SQ alpha=0.4/calib128; Omni LWC-only prefix128/final1024/epoch20, LET=off"
    echo "[protocol] FlatQuant prefix128/final512+heldout512/epoch15, best_epoch=off"
    echo "[protocol] RecQuant from SAME FlatQuant MSE checkpoint; final1024/epoch15, ABC CE + 0.3 boundary"
    echo "[protocol] RecQuant jointly learns matrix/diagonal/LWC/LAC; prefix frozen; no independent held-out after LFQ"
    echo "[protocol] successful eval cleanup=${CLEAN_EVAL_ARTIFACTS}; no automatic import of older results"
    echo "[protocol] W8A8/shared BF16 root=${RUN_ROOT}"
    echo "[protocol] W4A8 root=${W4A8_RUN_ROOT}"
    local task method precision index=0 bf16_count=0
    [[ -z "${SELECTED_METHODS[bf16]:-}" ]] || bf16_count=1
    local total=$(( ${#TASK_LIST[@]} * (bf16_count + (${#METHOD_LIST[@]} - bf16_count) * ${#PRECISION_LIST[@]}) ))
    local -A bf16_done=()
    for precision in w8a8 w4a8; do
        [[ -n "${SELECTED_PRECISIONS[${precision}]:-}" ]] || continue
        for task in "${TASK_LIST[@]}"; do
            # Always preserve dependencies, even if METHODS is specified out of order.
            for method in bf16 rtn smoothquant omniquant flatquant recquant; do
                [[ -n "${SELECTED_METHODS[${method}]:-}" ]] || continue
                if [[ "${method}" == bf16 ]]; then
                    [[ -z "${bf16_done[${task}]:-}" ]] || continue
                    bf16_done["${task}"]=1
                fi
                index=$((index + 1))
                printf '\n[experiment %d/%d] task=%s method=%s precision=%s\n' "${index}" "${total}" "${task}" "${method}" "${precision}"
                run_method "${task}" "${method}" "${precision}"
            done
        done
    done
    echo "[done] ${total} selected experiments handled; W8A8/BF16=${RUN_ROOT}; W4A8=${W4A8_RUN_ROOT}"
}

if [[ "${DRY_RUN}" == 1 ]]; then
    main
else
    LOG_ROOT="${RUN_ROOT}"
    if [[ -n "${SELECTED_PRECISIONS[w4a8]:-}" ]]; then
        LOG_ROOT="${W4A8_RUN_ROOT}"
    fi
    mkdir -p "${LOG_ROOT}"
    main 2>&1 | tee -a "${LOG_ROOT}/run.log"
fi
