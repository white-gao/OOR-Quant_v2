# SID_a LFQ 512-sample diagnostics

生成时间：2026-08-03T07:00:24+00:00  
设备：`NVIDIA GeForce RTX 5090`（逻辑 `cuda:0`，`CUDA_VISIBLE_DEVICES=0`）  
软件：Python `3.10.19`，PyTorch `2.9.0+cu128`  

## 结论

LFQ 在这 512 条 test 样本上显著降低了 SID_a KL（配对 bootstrap 95% CI 全部小于 0），而且 512/512 个样本的 KL 都低于 MSE-LWC，说明 SID_a soft-distribution 对齐能够泛化到测试集。

LFQ 还把 FP top-32 overlap 从 82.18% 提高到 88.92%。真实 SID_a 指标呈轻微混合波动：MRR 和 hit@1 上升，hit@32 从 67.97% 变为 67.77%，只相差 1/512 个样本。现有证据不支持“SID_a 泛化失败”这一解释；结合完整 benchmark 中 pass@1 上升而较大 beam k 下降，下一嫌疑应是 SID_b/SID_c 的条件分布或 beam-search 路径/排序交互。需要做后续 token 的 teacher-forcing 条件 KL 才能确认。

这项诊断是第一生成步 `<s_a_*>` 上的 teacher-distribution proxy，不等价于完整 beam-search 三 token SID 的推荐指标。

## 实验控制

- 数据：AD `test` 的前 `512` 条，offset `0`；首/末 sample id：`0` / `511`。
- 词表：只在全部 `8192` 个 `<s_a_*>` token 上重新归一化 softmax；比较位置为 prompt 的最后一个 token，即 SID_a 的下一 token 分布。
- FP：原始 `artifacts/models/1.7B`，BF16。
- MSE-LWC：`artifacts/results/fake_quant/omniquant_asym_lwc_w4a16_ad3000/1.7B/ad/omniquant_calibration`，W4A16 asymmetric LWC、无 LET。
- LFQ-LWC：`artifacts/results/fake_quant/omniquant_asym_lwc_lfq_strict_prefix_w4a16_ad3000/1.7B/ad/omniquant_calibration`；0–26 层来自同一 MSE prefix，只有第 27 层使用 SID_a LFQ-CE 优化。
- 三个模型使用完全相同的 prompt；batch size `4`；随机种子 `42`。

## FP 分布对齐指标

所有距离均越低越好；KL 为 `KL(FP || Quant)`。

| 模型 | mean KL | median KL | p90 KL | mean JSD | mean TV | FP top-1 一致率 | FP top-1 mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSE-LWC | 0.068988 | 0.065939 | 0.088747 | 0.016748 | 0.146024 | 67.38% | 1.63 |
| LFQ-LWC | 0.030278 | 0.028266 | 0.040178 | 0.007507 | 0.096053 | 76.56% | 1.34 |

### 配对 KL 差值

定义 `ΔKL = KL_LFQ - KL_MSE`。mean `-0.038710`，median `-0.038354`，paired bootstrap 95% CI `[-0.039844, -0.037576]`；LFQ 单样本胜率 `100.00%`，完全相同比例 `0.00%`。

## FP top-k 候选集合重合率

`overlap@k = |TopK_FP ∩ TopK_Quant| / k`；它比全分布 KL 更贴近 beam-search 候选保留。

| k | MSE mean overlap | LFQ mean overlap | LFQ−MSE | MSE exact-set | LFQ exact-set |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.38% | 76.56% | +9.18 pp | 67.38% | 76.56% |
| 4 | 71.78% | 80.57% | +8.79 pp | 19.14% | 35.16% |
| 8 | 75.49% | 83.33% | +7.84 pp | 3.71% | 13.87% |
| 16 | 78.54% | 86.10% | +7.56 pp | 0.20% | 4.10% |
| 32 | 82.18% | 88.92% | +6.74 pp | 0.00% | 0.78% |

## Ground-truth SID_a 排名

可解析 ground-truth 的样本数：`512/512`。若一条样本有多个真值，取最佳 SID_a rank。

| 模型 | MRR | mean rank | median rank | hit@1 | hit@4 | hit@8 | hit@16 | hit@32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 0.258096 | 97.93 | 11 | 14.84% | 32.81% | 46.09% | 56.84% | 69.14% |
| MSE-LWC | 0.258392 | 105.30 | 13 | 15.62% | 31.84% | 44.92% | 55.08% | 67.97% |
| LFQ-LWC | 0.263447 | 99.51 | 11 | 16.21% | 31.84% | 44.34% | 56.45% | 67.77% |

## 已有 3000-sample beam-search benchmark

| 指标 | MSE-LWC | LFQ-LWC | LFQ−MSE |
|---|---:|---:|---:|
| pass@1 | 1.27% | 1.40% | +0.13 pp |
| pass@4 | 3.53% | 3.17% | -0.37 pp |
| pass@8 | 4.67% | 4.30% | -0.37 pp |
| pass@16 | 6.77% | 6.53% | -0.23 pp |
| pass@32 | 9.23% | 8.97% | -0.27 pp |
| recall@32 | 2.68% | 2.59% | -0.09 pp |

## 运行信息与完整性检查

| 模型 | logits 收集耗时 | 输出形状 | 每行 logsumexp 最大绝对误差 |
|---|---:|---:|---:|
| FP | 14.3 s | `(512, 8192)` | `4.768e-07` |
| MSE-LWC | 21.7 s | `(512, 8192)` | `2.384e-07` |
| LFQ-LWC | 21.5 s | `(512, 8192)` | `4.768e-07` |

## 后续判断规则

1. 若 LFQ test KL 未下降：补跑原 128 calibration prompts 的 MSE/LFQ A/B；calib 降而 test 不降才可严格称为泛化问题。
2. 若 KL 下降但 top-k overlap 或真实 SID_a hit 下降：调整最后层目标（top-k-aware、hard-label CE 或混合 MSE/KL），而不是单纯扩大校准集。
3. 若 SID_a 的 KL、top-k 和真值 hit 都改善，但完整推荐仍掉点：下一步测 teacher-forcing 的 SID_b/SID_c 条件 KL，定位后续 token 或 beam 路径偏移。

本报告只保留聚合统计；一次性诊断脚本与中间 logits 在运行后删除。
