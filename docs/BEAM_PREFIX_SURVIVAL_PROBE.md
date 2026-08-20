# AD 全量 final-beam SID 前缀存活诊断

## 1. 目的与范围

本 probe 用已有的全量 AD `test_generated.json` 比较三组模型：

- BF16：`artifacts/results/fake_quant/recommender/1p7b_ad_product_full_bf16_w8a8/bf16/ad`
- naive FP8 W8A8：`artifacts/results/fake_quant/recommender/1p7b_ad_product_full_bf16_w8a8/w8a8/ad`
- symmetric LWC、无 LET 的 FP8 W8A8：`artifacts/results/fake_quant/omniquant_sym_lwc_nolet_fp8w_fp8a_calib128_ad_full`

三组结果包含完全对齐的 27,677 条测试样本，每条样本最多保存 32 条完整 SID 路径。统计定义如下：

- `A hit@k`：前 k 条完整生成路径中，至少一条路径的 SID-A 与某条 ground truth 的 SID-A 相同。
- `AB hit@k`：至少一条路径的 `(SID-A, SID-B)` 与同一条 ground truth 路径的 AB 前缀相同；不会把不同 ground truth 的 A、B 交叉组合。
- `ABC hit@k`：至少一条完整路径匹配 ground truth，等价于当前 SID `pass@k`。
- prefix overlap：忽略重复前缀后，比较量化模型与 BF16 最终 top-k 路径的前缀集合。

重要限制：当前文件只保存最终返回的 top-32 完整路径，没有保存每一步的候选分数、父节点和被剪枝路径。因此本文分析的是 **final-beam prefix survival**，可以定位结果在哪个 SID 深度开始分歧，但不能直接证明某条路径是在真实 beam-search 的哪一步被剪掉。

## 2. 数据校验

| Model | Samples | 无法解析的 generation | 不足 32 条的样本 |
|---|---:|---:|---:|
| BF16 | 27,677 | 0 | 0 |
| Naive FP8 | 27,677 | 1 | 1 |
| LWC FP8 | 27,677 | 2 | 2 |

异常生成最多只影响 2/27,677 条。probe 得到的 `ABC hit@32` 分别为 6,028、5,683、5,711 条，与三份 `eval_results.json` 中的 `pass@32` 完全一致，说明 SID 解析和样本对齐口径正确。

## 3. 核心结论

### 3.1 naive FP8 的主要新增损失不在 SID-A，而在 SID-B 之后

在 top-32 下：

- `A hit`：BF16 52.777%，naive FP8 52.903%，反而高 0.126 pp；配对检验不显著（p=0.435）。
- `AB hit`：52.777% 的 A 覆盖最终只转化为 26.112% 的 AB 覆盖，相比 BF16 下降 0.842 pp（p≈1.0e-7）。
- `ABC hit`：naive FP8 相比 BF16 下降 1.247 pp（p≈5.0e-15）。

因此，量化模型并不是普遍无法找到正确 SID-A。正确 A 已经进入最终 beam 时，后续 B 分支和 C 分支的保留能力下降才是更直接的问题。

在 BF16 能命中、naive FP8 却丢失的 1,144 条样本中：

- 251 条（21.9%）已经在 A 阶段丢失；
- 620 条（54.2%）保留了正确 A，但丢失了正确 AB；
- 273 条（23.9%）保留了正确 AB，但丢失了正确 ABC。

新增失败有一半以上落在 B 阶段，这说明后续如果做 beam-aware 诊断，应优先观察 `p(SID-B | prefix, SID-A)` 以及正确 AB 路径相对 beam 阈值的位置，而不是继续只优化 SID-A。

### 3.2 LWC 让 beam 更像 BF16，但没有稳定转化成 GT 命中提升

LWC 相对 naive FP8 的 `ABC hit@32` 从 5,683 增加到 5,711，净增加 28 条，即 +0.101 pp。但配对内部发生了大规模互换：

- LWC 打破了 naive 原本命中的 1,174 条；
- 同时修复了 naive 原本失败的 1,202 条；
- 95% 配对区间为 [-0.244, +0.446] pp，p=0.566。

所以当前 +28 条不能视为稳定收益，而是大量 repair/break 抵消后的微小净值。这也解释了为什么不同子集上的结果容易波动。

另一方面，LWC 与 BF16 的最终 beam 重合度在所有深度的 top-32 上都明显高于 naive：

- A-prefix Jaccard：68.005% → 70.258%；
- AB-prefix Jaccard：57.490% → 59.894%；
- ABC-path Jaccard：52.419% → 54.480%。

这说明逐层 MSE/LWC 的“处处尽量像原模型”确实反映到了生成路径上；问题不是完全没有对齐，而是这种整体路径对齐尚未集中到能够增加 ground-truth 命中的方向。

### 3.3 LWC 存在 beam 收窄现象

LWC 相对 naive FP8 的平均 unique A 数从 15.277 降到 14.909，平均 unique AB 数从 27.146 降到 26.702。对应地：

- `A hit@32` 下降 0.495 pp，共净减少 137 条，p=0.00544；
- `AB hit@32` 只增加 0.036 pp；
- `ABC hit@32` 只增加 0.101 pp。

因此 LWC 的表现可以概括为：路径整体更接近 BF16、已有 A 下的 B/C 条件转化率略有恢复，但探索到的 A 前缀更少。beam breadth 的下降抵消了部分条件预测改善。

## 4. Ground-truth 前缀存活率

| Depth | k | BF16 | Naive FP8 | LWC FP8 | Naive−BF16 | LWC−Naive |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 11.024% | 11.363% | 11.432% | +0.340 pp | +0.069 pp |
| A | 4 | 23.120% | 23.951% | 23.648% | +0.831 pp | -0.304 pp |
| A | 8 | 32.059% | 32.478% | 32.587% | +0.419 pp | +0.108 pp |
| A | 16 | 42.884% | 42.880% | 43.021% | -0.004 pp | +0.141 pp |
| A | 32 | 52.777% | 52.903% | 52.408% | +0.126 pp | -0.495 pp |
| AB | 1 | 2.992% | 3.111% | 3.183% | +0.119 pp | +0.072 pp |
| AB | 4 | 7.855% | 7.909% | 8.144% | +0.054 pp | +0.235 pp |
| AB | 8 | 12.259% | 12.183% | 12.281% | -0.076 pp | +0.098 pp |
| AB | 16 | 18.228% | 18.181% | 18.279% | -0.047 pp | +0.098 pp |
| AB | 32 | 26.954% | 26.112% | 26.148% | -0.842 pp | +0.036 pp |
| ABC | 1 | 1.955% | 1.915% | 1.958% | -0.040 pp | +0.043 pp |
| ABC | 4 | 5.770% | 5.499% | 5.582% | -0.271 pp | +0.083 pp |
| ABC | 8 | 9.582% | 9.083% | 9.141% | -0.499 pp | +0.058 pp |
| ABC | 16 | 14.727% | 14.232% | 14.218% | -0.495 pp | -0.014 pp |
| ABC | 32 | 21.780% | 20.533% | 20.634% | -1.247 pp | +0.101 pp |

这里还有一个值得注意的现象：naive FP8 在很小的 k 上 A/AB 命中并不差，但随着 k 扩大到 32，AB 和 ABC 相对 BF16 的缺口才明显出现。这更像是候选前缀覆盖和路径排序结构改变，而不是所有正确 token 的概率都发生了同方向下降。

## 5. Top-32 配对 repair/break

`Breaks` 表示基准命中而候选模型失败，`Repairs` 表示基准失败而候选模型命中。置信区间是样本级配对差值的正态近似 95% CI；p 值是大样本 McNemar/sign 正态近似。

| Comparison | Depth | Breaks | Repairs | Net Δ | 95% paired CI | p |
|---|---|---:|---:|---:|---:|---:|
| BF16 → Naive FP8 | A | 986 | 1,021 | +0.126 pp | [-0.191, +0.444] pp | 0.435 |
| BF16 → Naive FP8 | AB | 1,076 | 843 | -0.842 pp | [-1.152, -0.532] pp | 1.04e-7 |
| BF16 → Naive FP8 | ABC | 1,144 | 799 | -1.247 pp | [-1.558, -0.935] pp | 5.01e-15 |
| Naive FP8 → LWC FP8 | A | 1,283 | 1,146 | -0.495 pp | [-0.844, -0.146] pp | 0.00544 |
| Naive FP8 → LWC FP8 | AB | 1,172 | 1,182 | +0.036 pp | [-0.307, +0.380] pp | 0.837 |
| Naive FP8 → LWC FP8 | ABC | 1,174 | 1,202 | +0.101 pp | [-0.244, +0.446] pp | 0.566 |
| BF16 → LWC FP8 | A | 972 | 870 | -0.369 pp | [-0.672, -0.065] pp | 0.0175 |
| BF16 → LWC FP8 | AB | 994 | 771 | -0.806 pp | [-1.103, -0.508] pp | 1.11e-7 |
| BF16 → LWC FP8 | ABC | 1,056 | 739 | -1.145 pp | [-1.445, -0.846] pp | 7.31e-14 |

## 6. Top-32 failure funnel

| Model | Lost at A | A survives, lost at B | AB survives, lost at C | ABC hit | AB/A | ABC/AB |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 13,070 (47.223%) | 7,147 (25.823%) | 1,432 (5.174%) | 6,028 (21.780%) | 51.071% | 80.804% |
| Naive FP8 | 13,035 (47.097%) | 7,415 (26.791%) | 1,544 (5.579%) | 5,683 (20.533%) | 49.358% | 78.636% |
| LWC FP8 | 13,172 (47.592%) | 7,268 (26.260%) | 1,526 (5.514%) | 5,711 (20.634%) | 49.893% | 78.914% |

相对 naive FP8，LWC 的 `AB/A` 从 49.358% 回升到 49.893%，`ABC/AB` 从 78.636% 回升到 78.914%，但两者仍低于 BF16。与此同时 A 的绝对覆盖下降，导致最终净收益很小。

## 7. BF16 成功样本的丢失位置

| Candidate | BF16 hit → candidate miss | Lost at A | Lost at B | Lost at C | BF16 miss → candidate hit | Net hits vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| Naive FP8 | 1,144 | 251 | 620 | 273 | 799 | -345 |
| LWC FP8 | 1,056 | 214 | 578 | 264 | 739 | -317 |

LWC 把相对 BF16 的 break 从 1,144 降到了 1,056，但也把“BF16 未中而量化模型命中”的 repair 从 799 降到了 739，最终只回收 28/345≈8.1% 的 naive FP8 gap。

## 8. 与 BF16 的 final-beam overlap

下表只展示 top-32。`BF16-prefix recall` 表示 BF16 前缀有多少仍出现在量化模型中；precision 的分母是量化模型的 unique 前缀数。

| Candidate | Depth | BF16-prefix recall | Candidate-prefix precision | Jaccard |
|---|---|---:|---:|---:|
| Naive FP8 | A | 80.813% | 80.810% | 68.005% |
| LWC FP8 | A | 81.466% | 83.337% | 70.258% |
| Naive FP8 | AB | 72.286% | 72.947% | 57.490% |
| LWC FP8 | AB | 73.652% | 75.526% | 59.894% |
| Naive FP8 | ABC | 68.379% | 68.379% | 52.419% |
| LWC FP8 | ABC | 70.172% | 70.172% | 54.480% |

从 A 到 AB 再到 ABC，BF16 路径的保留率持续下降，体现了自回归误差随 SID 深度累积。LWC 在三个深度都改善 overlap，说明它不是完全无效；但 overlap 改善并不等价于 ground-truth 排名改善。

## 9. Beam 前缀多样性

| Model | Mean unique A | Mean unique AB | Mean unique ABC | Mean B branches per A |
|---|---:|---:|---:|---:|
| BF16 | 15.255 | 27.373 | 32.000 | 1.881 |
| Naive FP8 | 15.277 | 27.146 | 32.000 | 1.867 |
| LWC FP8 | 14.909 | 26.702 | 32.000 | 1.878 |

三组模型的完整路径都基本保持 32 条唯一输出，差异来自这些路径如何共享 A/AB 前缀。LWC 的 B-per-A 更接近 BF16，但 unique A 和 unique AB 更少，表现为更多路径集中到较少前缀上。

## 10. 对后续实验的含义

1. **先不要把主要精力放在 SID-A-only 对齐上。** 全量结果显示 A 覆盖不是 naive FP8 相对 BF16 的主要缺口；B 条件分支才是第一优先级。
2. **“接近 BF16”与“命中 GT”需要同时报告。** LWC 显著提高路径 overlap，却只产生不显著的 +28 条净命中。后续 beam-aware 目标至少要同时检查 prefix overlap、GT prefix survival、repair/break 和 beam breadth。
3. **小校准集上的纯 beam KL 容易学成路径重排。** 当前 LWC 已经出现上千条 repair 与 break 互换，说明只看平均分布 loss 或净指标会掩盖很强的样本级不稳定性。
4. **下一步最有信息量的是保存真实 beam trace，而不是立刻扩大训练目标。** 可以先在 512～2,000 条固定测试样本上保存每一步的候选 log-prob、父前缀、beam threshold 和最终 parent chain，专门分析 BF16 命中但 FP8 在 B 阶段丢失的样本。这样可以区分：
   - 正确 B 从一开始概率就偏低；
   - 正确 B 曾进入候选但跌出 top-32；
   - 正确 AB 存活，但 C 的条件概率或累计路径分数导致最终失败。

总体上，这个 probe 已经把问题从笼统的“量化让 beam-search 变差”收缩为：**FP8 的主要可定位损失发生在正确 A 之后的 B/C 条件路径；LWC 改善了整体路径对齐和条件转化，但同时收窄 A 前缀覆盖，且样本级 repair/break 很大，因此最终收益不稳定。**
