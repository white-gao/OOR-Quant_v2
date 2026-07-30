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

All launchers resolve their repository root through their canonical path and
write generated files below `OOR_QUANT_ARTIFACTS` (default: `./artifacts`).
