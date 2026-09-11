# FlatQuant fake-quant reproduction

This directory is a source-level copy of `fake_quant`. FlatQuant changes remain
isolated here and do not alter the active OmniQuant and ABC-LFQ implementation.

The formal `flatquant_core` path now includes:

- official Qwen3 transform placement: attention input, attention head,
  attention head dimension, MLP input, and down input;
- `U diag(s) V^T` factors with Cayley-parameterized orthogonal `U` and `V`, raw
  trainable singular vectors initialized to one, and random-orthogonal init;
- decomposed Kronecker transforms for attention input, MLP input, and down
  input, plus single transforms for head and head dimension;
- trainable diagonals at attention input, MLP input, and down input;
- per-output-channel two-sided LWC and four shared-site two-sided LAC modules;
- joint AdamW optimization with normalized block-output MSE and cosine decay;
- deployment-matched FP4-E2M1 weight and FP8-E4M3 activation fake-QDQ;
- cumulative FP-teacher versus quantized-prefix block alignment;
- finalization that folds diagonals, materializes transforms and non-STE QDQ,
  then saves/restores one checkpoint per decoder block;
- prefix checkpoint reuse for the 128-sample prefix plus 512/512 final-block
  train and held-out protocol.

KV-cache quantization and fused real-quant kernels are intentionally outside
this fake-quant baseline.

## Recommended staged run

First run the one-block environment and gradient smoke test:

```bash
SMOKE=1 bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh
```

Then train the complete formal baseline:

```bash
bash scripts/flat_quant/run_1p7b_ad_w4a8_pc_official_staged_cuda67.sh
```

The complete run trains layers 0-26 on the first 128 calibration records. It
then restores that prefix, trains layer 27 on records 0-511, and reports held-out
MSE on records 512-1023. Both stages retain the final epoch, so
`best_epoch=None` is expected. Set `RUN_EVAL=1` to evaluate the final checkpoint
on the full AD set with cards 6 and 7.

The formal defaults are per-output-channel weights, 15 epochs, matrix and
diagonal LR `5e-3`, LWC/LAC LR `5e-2`, AdamW weight decay `0.01`, diagonal
alpha `0.5`, and seed 42.

## Preliminary launchers

The older `w4a8_g128` launchers are retained only to locate results from the
initial matrix-core probes. They use a nonformal group-wise setting and should
not be used as the official FlatQuant comparison.
