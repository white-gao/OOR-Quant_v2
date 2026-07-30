# Local artifacts

This directory is intentionally ignored by Git.  It contains all large or
generated project files, while the source tree contains only code, scripts,
tests, and documentation.

```text
artifacts/
├── models/                 # Local OneRec checkpoints
├── data/                   # Local OneRec data
└── results/
    ├── real_quant/         # Retained actual FP8/W8A8 runtime experiments
    └── fake_quant/         # Created only when a future fake-QDQ run is needed
```

All code resolves this root through `shared.paths`.  Set
`OOR_QUANT_ARTIFACTS=/path/to/storage` to keep artifacts outside the repository
without changing experiment commands.

Results from the real and fake quantization tracks must remain in their
separate subtrees; they are not interchangeable measurements.
