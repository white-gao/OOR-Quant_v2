# Repository layout and artifact policy

## Quantization tracks

The project has two deliberately separate research tracks:

- `real_quant/`: actual FP8 W8A8 runtime.  It patches model Linear modules and
  executes real FP8 matrix multiplication.  Only this track may be used for
  end-to-end latency or deployment claims.
- `fake_quant/`: fake-QDQ, SmoothQuant, plain GPTQ, and diagnostic research.
  It is useful for numerical analysis and method development, but is not a
  substitute for the real runtime.

`benchmarks/benchmark/` is the local OpenOneRec evaluator dependency.  Keep it
available on `PYTHONPATH` until it is packaged under `third_party/` in a later,
tested migration.

Launchers live only in `scripts/real_quant/` and `scripts/fake_quant/`.

## Artifact root

All generated or large files belong below `artifacts/` (Git ignored):

```text
artifacts/
├── models/                 # local 1.7B and 8B checkpoints
├── data/                   # local OneRec data
└── results/
    ├── real_quant/
    │   ├── recommender/    # retained BF16 / RTN / plain GPTQ/GPTAQ runs
    │   ├── generic/        # retained WikiText-2, GSM8K, and MATH-500 runs
    │   └── profiling/     # created only when profiling is explicitly retained
    └── fake_quant/         # fake-QDQ / OmniQuant recommender and generic runs
```

Set `OOR_QUANT_ARTIFACTS=/absolute/path` before a run to place this entire tree
on another filesystem.  `shared.paths` is the single Python source of truth;
new code should not introduce another hard-coded `*/results` default.

## Retained result set — 2026-07-24

The repository retains only the real-runtime evidence needed for the current
research line:

- 8B recommender BF16 and pure RTN W8A8 baselines for AD/Product/Video;
- 8B plain GPTQ/GPTAQ recommender experiments;
- completed 1.7B and 8B WikiText-2, GSM8K, and MATH-500 control experiments;

Deleted as non-main or superseded: imported OpenOneRec/QDQ benchmark artifacts,
all historical fake-QDQ results, incomplete 1.7B thinking GSM8K output, and
early 1.7B recommendation sweeps, decode-A16 variants, alpha sweeps,
profiling/fusion experiments, and non-full (`test1000`) Stage-A
activation/weight/channel probes. Incomplete logs without evaluation artifacts
were also removed.

## Cleanup policy

1. Review and classify before deleting.
2. Keep real and fake results separate even when methods have similar names.
3. Remove only reproducible caches (`__pycache__`, `.pytest_cache`) without a
   separate review.
