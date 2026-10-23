# OOR-Quant

OOR-Quant studies post-training quantization for OpenOneRec. Real quantization
is retained as an executable FP8 deployment baseline; the main low-bit quality
research path is composable fake QDQ. Fake QDQ supports independent weight and
activation formats (`none`, FP8 E4M3, FP4 E2M1, INT8, and INT4), but is not a latency or
packed-memory measurement.

## Layout

```text
real_quant/naive_w8a8/  executable real FP8 W8A8 baseline and FP8 PTQ variants
fake_quant/             composable FP8/INT8/INT4 QDQ quality experiments
flat_quant/             FlatQuant parameterization and RecQuant task alignment
benchmarks/             recommendation and general-capability evaluators
scripts/                serial experiment launchers
paper/motivation/       motivation probe/plot sources (manuscripts are local)
paper/efficiency/       stage-timing collection and plotting sources
artifacts/              Git-ignored generated results
```

New outputs belong under `artifacts/results/`; see
[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) for the result policy.

## Environment and data

Run commands from the repository root. Real FP8 requires CUDA PyTorch and a GPU
that supports FP8 `torch._scaled_mm`.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(hasattr(torch, '_scaled_mm'))"
```

The server defaults are `/root/dataDisk/guowei/models/1.7B` and
`/root/dataDisk/guowei/data/onerec_data/benchmark_data`. Calibration and test
splits must remain separate. Override the defaults with
`OOR_QUANT_MODEL_ROOT`, `OOR_QUANT_DATA_ROOT`, or the direct
`OOR_QUANT_BENCHMARK_DATA` path. Generated outputs remain controlled by
`OOR_QUANT_ARTIFACTS` (default: `<repo>/artifacts`).

### Moving to another server

Git contains source code, tests, launchers, and public configuration examples,
not model weights, datasets, calibration checkpoints, full evaluation outputs,
private API settings, or manuscript/report drafts. Copy those required assets
separately over your trusted transfer channel. Preserve the relative directory
layout under `artifacts/results/`: many resume/evaluation workflows use it to
locate source checkpoints and reference metrics. Keep each calibration directory
with all `layer_*.pt` files **and** its parent `flatquant_core_config.json`.
RecIF generation resumption additionally needs `run_config.json` and `parts/`;
`eval_results.json` alone is not enough to resume unfinished generation.

Use Python 3.10. The source environment uses PyTorch 2.9.0+cu128 and
Transformers 4.57.6; `requirements-migration.txt` records the core non-Torch
packages. The old `requirements.txt` is a historical broad environment export,
not the current migration lockfile. Install a PyTorch CUDA build compatible
with the new GPU/driver first, then:

```bash
python -m pip install -r requirements-migration.txt
export PYTHON_BIN="$(command -v python)"
export OOR_QUANT_MODEL_ROOT=/path/to/models
export OOR_QUANT_BENCHMARK_DATA=/path/to/benchmark_data
export OOR_QUANT_ARTIFACTS=/path/to/artifacts
```

The model root should contain directories named `1.7B` and/or `8B`, with their
original tokenizer/config/weight files. The benchmark directory contains task
subdirectories and their test/calibration parquet files. Set `GPU` or `GPUS`
explicitly on the new server; historical launcher filenames do not determine
the actual GPU selection. Existing manifests can contain absolute paths and
file timestamps. If moving them causes a provenance mismatch, do not edit the
manifest to bypass validation: retain the reference run and use a new output
directory, or configure the documented source/reference paths explicitly.

CPU checks and read-only previews (no large-model experiments):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests paper/motivation/tests paper/efficiency/tests
TASKS='interactive label_pred' CHECK_ONLY=1 \
  bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh
DRY_RUN=1 GPUS=0,1 PRECISIONS=w8a8 \
  bash scripts/fake_quant/run_8b_ad_product_w8a8_main_cuda67.sh
```

The 8B main launcher supports both W8A8 and W4A8 (both by default), including
full evaluation. The separate `run_8b_product_w4a8_flatquant_recquant_train_cuda6.sh`
is deliberately **training-only**. Some launchers clean large generation files
after successful scoring; set `CLEAN_EVAL_ARTIFACTS=0` when you need to retain them.

RecIF BF16 supports interactive recommendation, behavior prediction, and item
understanding; the latter requires an explicitly enabled external judge for
scoring. See [the task README](real_quant/full_precision/README.md). Copy a
public `recif_judge.*.example.json` into a Git-ignored `*.local.json` on the new
server and configure credentials there, never in committed examples. No judge
call is made by the CPU checks above. The unfinished auxiliary-task quantized
adapter is not part of this migration release; its three-task comparison is
not yet a validated runnable workflow.

The native FP8 path also needs compatible FP8 hardware and its custom input
quantization kernels; core fake-QDQ tests do not validate that backend. The
efficiency collector reports measured BF16/RTN times and **operator-estimated**
RecQuant times, not deployed end-to-end RecQuant latency. Files named `mockup`
contain illustrative values only. A clean installation and native GPU kernels
must still be checked on the destination machine.

## Real FP8 runner

```bash
python -m real_quant.naive_w8a8.run_hf_naive_w8a8 --help
```

`--weight_quant_mode` supports:

| Mode | Description |
| --- | --- |
| `minmax` | Naive per-output-channel FP8 W8A8 baseline. |
| `gptq` | Plain GPTQ using calibration-input Hessians. |
| `gptaq` | Plain GPTAQ; `--gptaq_alpha` and `--[no-]gptaq_activation_aware` control its calibration target. |

Example: plain GPTQ W8A8 on the AD domain.

```bash
CUDA_VISIBLE_DEVICES=0 python -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  --model_path /root/dataDisk/guowei/models/1.7B \
  --task ad \
  --weight_quant_mode gptq \
  --gptq_calib_sample_size 128 \
  --sample_size 1000 \
  --output_dir artifacts/results/real_quant/recommender/ptq/ad_1p7b_plain_gptq_w8a8_calib128_test1000 \
  --evaluate
```

For a deployment-oriented comparison, `--decode_a16_single_token` keeps the
FP8 weights but bypasses activation FP8-QDQ for single-token decode calls.
Algorithm comparisons should use the same decode setting for every method.

## Fake QDQ runner

```bash
python -m fake_quant.run_m1_onerec_ad \
  --mode baseline_qdq \
  --model_path /root/dataDisk/guowei/models/1.7B \
  --task ad \
  --weight_quant_format int4 \
  --activation_quant_format int8 \
  --device cuda:7 \
  --output_dir artifacts/results/fake_quant/recommender/int4w_int8a_ad_1p7b \
  --evaluate
```

Current main experiments use FP8 E4M3FN weights/activations for W8A8 and FP4
E2M1 weights with FP8 activations for W4A8. The generic runner still defaults
to `baseline_qdq` with asymmetric INT8-W and dynamic per-token symmetric
INT8-A; main-experiment launchers explicitly set their floating-point formats.
Integer and weight-only combinations remain available for ablations. `baseline_w8a8`,
`smoothquant_w8a8`, and `gptq_fp8_w8a8` are retained FP8 compatibility
modes and should specify FP8 formats explicitly.

Integer weights use per-output-channel asymmetric affine QDQ with a zero point
by default. This default is shared by RTN and OmniQuant. Use
`--weight_quant_scheme symmetric` only for an explicitly zero-centered
integer control. FP8 E4M3FN remains symmetric because it has no affine integer
zero point. The legacy `--omni_weight_quant_scheme` option is retained as an
alias for `--weight_quant_scheme`.

All fake-quant baselines use one deployment-matched numerical contract:
master weights, quantization parameters, QDQ arithmetic, and calibration
losses stay FP32; QDQ outputs are cast back to the model dtype before model
operators execute. OmniQuant checkpoints produced by the former FP32-surrogate
calibration path are intentionally rejected and must be recalibrated.

## Result conventions

The real runner records the selected quantization method, calibration settings,
runtime settings, and generated outputs in its output directory. Calibration
time is offline preprocessing and is not part of generation latency. Use
`calib=128, test=1000` only for quick iteration; final reports should use the
full held-out test set and the agreed calibration protocol.
