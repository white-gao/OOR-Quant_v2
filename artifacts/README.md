# Local artifacts

This directory is intentionally ignored by Git. It contains generated project
files, while reusable models and datasets live on the server data disk.

```text
artifacts/
└── results/
    ├── real_quant/         # Retained actual FP8/W8A8 runtime experiments
    └── fake_quant/         # Created only when a future fake-QDQ run is needed
```

Models default to `/root/dataDisk/guowei/models`, benchmark data defaults to
`/root/dataDisk/guowei/data/onerec_data/benchmark_data`, and generated
results use this directory. Set `OOR_QUANT_MODEL_ROOT`, `OOR_QUANT_DATA_ROOT`,
`OOR_QUANT_BENCHMARK_DATA`, or `OOR_QUANT_ARTIFACTS` to override each location
without changing experiment commands.

Results from the real and fake quantization tracks must remain in their
separate subtrees; they are not interchangeable measurements.
