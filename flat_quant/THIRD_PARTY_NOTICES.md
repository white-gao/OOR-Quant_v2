# Third-party notice

The Kronecker/SVD/Cayley design in `flat_quant/flatquant/transforms.py` is
derived from concepts and implementation structure in the official FlatQuant
repository:

- Project: <https://github.com/ruikangliu/FlatQuant>
- Paper: *FlatQuant: Flatness Matters for LLM Quantization*
- Copyright (c) 2024 ruikangliu
- License: MIT, <https://github.com/ruikangliu/FlatQuant/blob/main/LICENSE>

The implementation in this directory is adapted for this repository's Qwen3,
FP4/FP8 fake-QDQ, per-output-channel LWC, and deployment-matched execution contract.
