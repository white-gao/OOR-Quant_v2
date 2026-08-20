# FP8 W8A8 OmniQuant 最后一层 MSE 汇总

更新时间：2026-08-06

本文汇总 1.7B、AD、FP8-W/FP8-A OmniQuant 实验中第 27 层（最后一个 Transformer block）的校准 MSE，便于后续实验快速对比。推荐指标仍应以全量评测为准；这里的 MSE 主要作为优化过程和候选配置筛选的 proxy。

## 主表：calib=128

下表按优化后 Layer 27 MSE 从低到高排列。`全层 final MSE 和`是 Layer 0–27 各层 final MSE 的直接求和，只作为辅助统计。

| 排名 | 配置 | LET 初始化 | LET scale 参数化 | Layer 27 初始 MSE | Layer 27 最终 MSE | 绝对下降 | 相对下降 | 全层 final MSE 和 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | LWC-only | — | 无 LET | 21.492066 | **21.244718** | 0.247349 | 1.151% | **57.536160** |
| 2 | LWC + learned LET，SQ α=1 | SmoothQuant | unbounded log-scale | 22.539917 | **21.939486** | 0.600431 | 2.664% | 59.705350 |
| 3 | LWC + learned LET，SQ α=0.5 | SmoothQuant | unbounded log-scale | 22.365641 | **22.000721** | 0.364919 | 1.632% | 59.531320 |
| 4 | LWC + learned LET，SQ α=0.75 | SmoothQuant | unbounded log-scale | 22.504660 | **22.008982** | 0.495677 | 2.203% | 59.709999 |
| 5 | LWC + learned LET，1-init | 全 1 | bounded `[0.05, 20]` | 22.291260 | **22.040111** | 0.251149 | 1.127% | 59.679860 |
| 6 | LWC + learned LET，SQ α=0.25 | SmoothQuant | unbounded log-scale | 22.305933 | **22.070297** | 0.235636 | 1.056% | 59.715977 |
| 7 | LWC + learned LET，SQ α=0 | SmoothQuant | unbounded log-scale | 22.764140 | **22.186707** | 0.577433 | 2.537% | 60.289518 |

## 当前观察

- α sweep 内最好的 α=1 与最差的 α=0，最终 MSE 相差 `0.247221`，约占 α=0 最终 MSE 的 `1.11%`；固定 α 的总体影响不大。
- α=1 相比 1-init 的最终 MSE 低 `0.100625`，约 `0.46%`。但两者的 LET scale 参数化不同，因此不是严格单变量对照。
- LWC-only 的最终 MSE 比最佳 LET 配置 α=1 低 `0.694769`，约 `3.27%`；当前结果仍然更支持 LWC-only。
- α=0.5 的全层 final MSE 和在 LET 实验中最低，而 α=1 的最后一层 MSE 最低，说明不同深度偏好的 α 并不完全一致。
- α=1 学到的 scale 更激进。此前统计显示其 Layer 27 约 `60.37%` 的 scale 大于 20，因此即使校准 MSE 略优，也需要通过全量推荐指标检查泛化和数值稳定性。

## 历史补充结果

这些结果可以用于观察趋势，但不应直接混入上面的严格排序。

| 配置 | calib | LET scale 参数化 | Layer 27 初始 MSE | Layer 27 最终 MSE | 相对下降 | 全层 final MSE 和 | 说明 |
|---|---:|---|---:|---:|---:|---:|---|
| LWC + learned LET，SQ α=0.5 | 128 | bounded `[0.05, 20]` | 22.369795 | 22.010746 | 1.605% | 59.570581 | 旧边界版本；与 unbounded α=0.5 的 `22.000721` 几乎一致 |
| LWC + learned LET，1-init | 1024 | bounded `[0.05, 20]` | 22.174510 | 21.826898 | 1.568% | 58.906396 | calib 不同，不应据此判断 1024 一定优于 128 |

## 数据来源

- LWC-only：`artifacts/results/fake_quant/omniquant_sym_lwc_nolet_fp8w_fp8a_calib128_ad/1.7B/ad/omniquant_config.json`
- 1-init、calib=128：`artifacts/results/fake_quant/omniquant_sym_lwc_learned_let_ones_fp8w_fp8a_ad3000/1.7B/ad/omniquant_config.json`
- SmoothQuant α sweep：`artifacts/results/fake_quant/omniquant_sym_lwc_learned_let_smoothquant_alpha*_unbounded_fp8w_fp8a_calib128_ad/1.7B/ad/omniquant_config.json`
- SmoothQuant α=0.5、旧边界版本：`artifacts/results/fake_quant/omniquant_sym_lwc_learned_let_smoothquant_fp8w_fp8a_calib128_ad/1.7B/ad/omniquant_config.json`
- 1-init、calib=1024：`artifacts/results/fake_quant/omniquant_sym_lwc_learned_let_ones_fp8w_fp8a_calib1024_ad_calibration/1.7B/ad/omniquant_config.json`

## 比较注意事项

1. 只在量化格式、模型、任务、校准集和代码参数化一致时直接比较 MSE。
2. 1-init、旧 SQ α=0.5 实验运行于 LET scale 有界版本；最新 α sweep 运行于 unbounded log-scale 版本。
3. `eval_sample_size` 不参与逐层校准 MSE 的计算；这里主要需要保持 `calib_sample_size` 和校准数据一致。
4. Layer 27 MSE 是校准集上的 block-output MSE，不等价于推荐指标，也不能替代全量 beam-search 评测。
