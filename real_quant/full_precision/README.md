# HF Full-Precision Baseline

This directory contains the standalone HuggingFace full-precision baseline for
real FP8 runtime comparisons. It is intentionally separated from
`fake_quant` so BF16 latency, real naive W8A8, real GPTQ, and real GPTAQ can
share one runner without mixing in fake-quant code.

## OpenOneRec Alignment

The implementation follows the public OpenOneRec benchmark structure:

- data loading uses `benchmarks/benchmark/tasks/v1_0/registry.py::get_loader`;
- metric evaluation uses `benchmark.Benchmark.evaluate_dev`;
- outputs are saved as `<output>/<model>/<task>/<split>_generated.json`;
- recommendation prompts append `<|sid_begin|>` before SID generation;
- default recommendation generation uses `num_beams=32`,
  `num_return_sequences=32`, and `max_new_tokens=3`, matching the retained
  recommendation task configuration.

The intentional difference is the backend: this runner uses
`transformers.AutoModelForCausalLM.generate` instead of vLLM/Ray. That keeps the
baseline comparable to upcoming real FP8 PyTorch module replacements.

## Usage

AD full precision, one GPU:

```bash
PYTHONPATH=. python3 -m real_quant.full_precision.run_hf_baseline \
  --model_path /root/dataDisk/guowei/models/1.7B \
  --data_dir /root/dataDisk/guowei/data/onerec_data/benchmark_data \
  --output_dir artifacts/results/real_quant/recommender/bf16/ad_1p7b_bf16 \
  --task ad \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --max_new_tokens 3 \
  --overwrite \
  --evaluate
```

`--batch_size` defaults to `1`. This is the safest setting for accuracy
alignment because it avoids any batch-coupled effects from padding, batched
kernel choices, or future activation quantization scale implementations.

For throughput experiments, pass `--batch_size auto` explicitly. The runner then
checks the selected CUDA device total memory, infers the model size from the
model path, and chooses a conservative throughput-oriented batch size. On the
current 140GB-class cards, the auto values are:

- OneRec-1.7B: `ad/product/label_cond=8`, `video/interactive=4`;
- OneRec-8B: `ad/product/label_cond=4`, `video/interactive=2`.

Override it explicitly, for example `--batch_size 2`, if the card is already
heavily occupied or if a task OOMs. With beam search, the effective active
sequences are roughly `batch_size * num_beams`, so video and 8B runs need lower
batches than ad/product.

## Latency Fields

The generated JSON keeps the OpenOneRec-compatible fields and adds:

- top-level `latency`: aggregate tokenize/generate/decode/end-to-end stats;
- per-sample `latency`: token counts and timing;
- per-sample `input_tokens`, `output_tokens`, and `times`, compatible with the
  benchmark MFU-style schema.

For `batch_size > 1`, batch wall time is distributed across samples, so the
reported per-sample values are batch-amortized latency. With `batch_size=1`,
they are single-request latency.

For BF16 vs real FP8 comparisons, use `generate_time_*` as the primary compute
latency and keep `end_to_end_time_*` as the user-visible total.

## Generic perplexity controls

`run_ppl_benchmark.py` evaluates WikiText-2 or a cached C4 English-validation
subset with full precision, real FP8 RTN W8A8, or fake QDQ. C4 is fixed to 256
non-overlapping 2048-token windows by default. Build the shared cache once,
then launch the paired real-FP8 1.7B/8B experiment suite with:

```bash
bash scripts/real_quant/run_1p7b_8b_c4_ppl_cuda7.sh
```

For an INT4-W / FP8-A quality experiment, use the same cache and add the fake
QDQ format pair explicitly:

```bash
python -m real_quant.full_precision.run_ppl_benchmark \
  --model_path /root/dataDisk/guowei/models/1.7B \
  --dataset c4 \
  --device cuda:7 \
  --quantization fake_qdq \
  --weight_quant_format int4 \
  --activation_quant_format fp8_e4m3fn \
  --output_dir artifacts/results/fake_quant/generic/c4_1p7b_int4w_fp8a
```
