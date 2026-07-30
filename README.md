# OOR-Quant

OOR-Quant studies post-training quantization for OpenOneRec. Real quantization
is retained as an executable FP8 deployment baseline; the main low-bit quality
research path is composable fake QDQ. Fake QDQ supports independent weight and
activation formats (`none`, FP8 E4M3, INT8, and INT4), but is not a latency or
packed-memory measurement.

## Layout

```text
real_quant/naive_w8a8/  executable real FP8 W8A8 baseline and FP8 PTQ variants
fake_quant/             composable FP8/INT8/INT4 QDQ quality experiments
benchmarks/             recommendation and general-capability evaluators
scripts/                serial experiment launchers
artifacts/              git-ignored models, data, and results
```

New outputs belong under `artifacts/results/`; see
[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) for the result policy.

## Environment and data

Run commands from the repository root. Real FP8 requires CUDA PyTorch and a GPU
that supports FP8 `torch._scaled_mm`.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(hasattr(torch, '_scaled_mm'))"
```

The default model and data locations are `artifacts/models/1.7B` and
`artifacts/data/onerec_data/benchmark_data`. Calibration and test
splits must remain separate.

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
  --model_path artifacts/models/1.7B \
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
  --model_path artifacts/models/1.7B \
  --task ad \
  --weight_quant_format int4 \
  --activation_quant_format fp8_e4m3fn \
  --device cuda:7 \
  --output_dir artifacts/results/fake_quant/recommender/int4w_fp8a_ad_1p7b \
  --evaluate
```

Use `--mode baseline_qdq` for mixed-format studies. For example,
`--weight_quant_format int4 --activation_quant_format none` is INT4-W / BF16-A
and `--weight_quant_format int4 --activation_quant_format fp8_e4m3fn` is
INT4-W / FP8-A. `baseline_w8a8`, `smoothquant_w8a8`, and `gptq_fp8_w8a8` are
retained FP8-only compatibility modes.

## Result conventions

The real runner records the selected quantization method, calibration settings,
runtime settings, and generated outputs in its output directory. Calibration
time is offline preprocessing and is not part of generation latency. Use
`calib=128, test=1000` only for quick iteration; final reports should use the
full held-out test set and the agreed calibration protocol.
