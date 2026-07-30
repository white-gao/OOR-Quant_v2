# Real FP8 W8A8 inference

This package replaces selected `nn.Linear` modules with `RealFP8Linear` and
uses `torch._scaled_mm` for FP8 W8A8 matrix multiplication.

- weight: FP8 E4M3FN, per-output-channel absmax scale;
- activation: FP8 E4M3FN, dynamic per-token scale by default;
- output: BF16 by default;
- `lm_head`: kept in BF16;
- QKV and gate/up: share one input activation quantization within each group.

The recommendation runner is:

```bash
PYTHONPATH=. python -m real_quant.naive_w8a8.run_hf_naive_w8a8 --help
```

`--weight_quant_mode` accepts `minmax`, `gptq`, and `gptaq`. The default
`batch_size=1` aligns real FP8 runs with the full-precision recommendation
baseline. `--decode_a16_single_token` is an optional deployment configuration:
it retains FP8 weights while bypassing activation FP8-QDQ for single-token
decode calls.

For reproducible 8B runs, use the retained launchers under
`scripts/real_quant/`.
