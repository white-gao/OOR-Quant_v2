# fake_quant/support

This package contains internal helpers used by the active fake-quant and
OmniQuant runners.

- `smoothquant_core.py`: standalone SmoothQuant scale and weight-folding helpers
  migrated from the legacy `fake_quant` package.
- `smoothquant_runtime.py`: SmoothQuant scale collection, exact folds, and fixed
  SQ W8A8 wrapper construction.
- `runtime_utils.py`: tensor-tree detach/device helpers used by the runner and
  calibration pipeline.
