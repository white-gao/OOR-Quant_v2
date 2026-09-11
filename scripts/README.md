# Experiment launchers

Canonical experiment launchers are separated from Python packages:

- `scripts/real_quant/`: real FP8/W8A8 runtime, recommender, and generic jobs.
- `scripts/fake_quant/`: fake-QDQ and learnable-quant jobs.

The paired 1.7B AD-1000 fake-QDQ sweep is launched with
`bash scripts/fake_quant/run_1p7b_ad1000_qdq_precision_sweep_cuda7.sh`. It
serially evaluates BF16, FP8-W/FP8-A, INT8-W/FP8-A, and INT4-W/FP8-A on the
same first 1,000 AD test samples. Results are written to
`artifacts/results/fake_quant/recommender/1p7b_ad1000_qdq_precision_sweep/`
by default.

The paired full-precision versus INT4-W/FP8-A GSM8K run is launched with
`bash scripts/fake_quant/run_1p7b_gsm8k_int4w_fp8a_cuda7.sh`. It uses the
established non-thinking, fixed-5-shot protocol and writes resumable results
under `artifacts/results/fake_quant/generic/gsm8k_1p7b_qdq_precision_sweep/`.

Package-local launcher duplicates have been removed; use the paths in this
directory.

All launchers use `/root/dataDisk/guowei/models` and
`/root/dataDisk/guowei/data/onerec_data/benchmark_data` by default, while writing
generated files below `OOR_QUANT_ARTIFACTS` (default: `./artifacts`). Override
inputs with `OOR_QUANT_MODEL_ROOT`, `OOR_QUANT_DATA_ROOT`,
`OOR_QUANT_BENCHMARK_DATA`, `MODEL_PATH`, or `DATA_DIR`.

## Controlled FP4-W/FP8-A ABC boundary comparison

Use the dependency-aware multi-GPU launcher below for the 1.7B AD experiment:

```bash
GPUS=6,7 bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_lwc_abc_boundary_calib1024.sh
```

It trains one deployment-matched symmetric LWC-only MSE prefix on the first
128 calibration records, then reuses its exact layers 0--26 for three
final-layer arms. Every final-layer arm trains on the first 512 records:
MSE-LWC, ABC-LFQ, and ABC-LFQ plus boundary loss. The LFQ
arms load all 1,024 records but backpropagate only through the first 512; the
last 512 are a fixed validation tail and are excluded from checkpoint
selection. The MSE arm uses the same first 512 records. Finally, a standalone
diagnostic evaluates all three checkpoints on the exact same held-out tail and
writes CE/KL, top-32 retention, intruder, boundary-violation, gap-error, and
paired-bootstrap results to JSON.

The launcher defaults to 20 epochs, seed 42, boundary weight 0.1, and supports
resume-by-stage. The prefix runs on the first listed GPU; after it completes,
the three independent final-layer arms are assigned round-robin to `GPUS`, and
the held-out diagnostic returns to the first GPU. A single GPU remains supported.
Use `DRY_RUN=1` to print every command, `OVERWRITE=1` to rerun completed stages,
or `RUN_DIAGNOSTICS=0` to stop after checkpoint training.

For the first standard group-wise OmniQuant-LWC probe, run only the composite
ABC plus boundary arm with group size 128:

```bash
GPUS=7 bash scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_groupwise_g128_lwc_abc_boundary.sh
```

This retrains the required group-wise MSE prefix, trains only the final
ABC-CE plus 0.3-boundary branch, prints the final layer's fixed-state
initial/final block-output MSE, and then evaluates the untouched calibration
tail `[512, 1024)`. The terminal summary and JSON report include per-slot
A/B/C CE, KL, top-1 agreement, top-5/top-10/top-32 retention, boundary
violation, and near/far intruder rates. The diagnostic runs on the first GPU
in `GPUS`; override it with `DIAGNOSTIC_GPU`, skip it with
`RUN_HELDOUT_DIAGNOSTICS=0`, or replace an existing report with
`DIAGNOSTIC_OVERWRITE=1`.

Override `WEIGHT_GROUP_SIZE` to test another group size. The underlying
controlled launcher also accepts `FINAL_ARMS=mse,abc,boundary` subsets while
preserving its original per-channel three-arm defaults.

## W4A8/W8A8 g128 six-run AD-full matrix

Run the compact two-precision, three-method matrix on GPUs 6 and 7:

```bash
bash scripts/fake_quant/run_1p7b_ad_full_w4w8a8_g128_core6_cuda67.sh
```

For each of FP4-W/FP8-A and FP8-W/FP8-A, the launcher evaluates RTN-g128,
OmniQuant-LWC-MSE-g128, and OmniQuant-LWC with ABC-CE plus 0.3 boundary.
Learned branches use a 128-record Layer 0--26 prefix and independently train
Layer 27 on all 1,024 calibration records. The completed W4 g128 prefix is
reused by default; the W8 prefix is trained once. MSE and ABC-boundary
final-layer calibration run concurrently on the first two GPUs, while each
AD-full evaluation is sharded over every GPU in `GPUS`. Completed stages are
skipped, and the final six-method table is saved as `core6_eval_summary.json`.


## Full FP8-W/FP8-A ABC boundary comparison

Run the two final methods serially, with every AD-full evaluation sharded over
physical GPUs 6 and 7:

```bash
bash scripts/fake_quant/run_1p7b_ad_full_fp8w8a8_lwc_abc_boundary_cuda67.sh
```

The launcher trains one symmetric OmniQuant-LWC prefix for layers 0--26 on
128 calibration records. ABC-LFQ and ABC-LFQ plus boundary then independently
train layer 27 on all 1,024 calibration records from the same prefix and seed.
The boundary arm defaults to weight 0.3 and does not use B/C prefix expansion.
After each calibration, the full AD test set is round-robin sharded over both
GPUs and merged in source order. Once both methods finish, the launcher prints
their aggregate recommendation metrics and boundary-minus-ABC differences.
