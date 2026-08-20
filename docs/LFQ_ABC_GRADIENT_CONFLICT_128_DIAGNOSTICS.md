# LFQ ABC gradient-conflict diagnostics (128 calibration samples)

生成时间：2026-08-03T12:22:16+00:00  
设备：`NVIDIA GeForce RTX 5090`（物理 GPU 1，进程内逻辑 `cuda:0`）  
样本：AD calibration `128` 条，随机种子 `42`

## 结论

实验确认了两个不同阶段的问题：

1. **初始化时没有明显的全局反向梯度，但 SID_c 的梯度尺度占优。**纯 LWC 初始点的 128-sample mean gradient 两两 cosine 为 `a-b=+0.0775`、`a-c=+0.0173`、`b-c=+0.0275`，都不是负数，但接近正交；SID_c mean-gradient norm 是 SID_a 的 `2.82×`，联合梯度与 c 的 cosine 高达 `0.884`，与 a 只有 `0.349`。所以等权 ABC 从一开始就更接近“主要沿 c 方向优化”，而不是三个 slot 均匀前进。
2. **ABC 训练后出现了明确的 Pareto/梯度冲突。**训练后 checkpoint 的 mean gradient 两两 cosine 全部变负：`a-b=-0.1915`、`a-c=-0.0984`、`b-c=-0.0301`。在逐样本梯度上，ABC 总梯度会局部增大 SID_a loss 的样本比例达到 `36.72%`。这说明优化进入了三个 slot 的共享 LWC 参数难以继续共同改进的区域。

LET=1 提供了新的、总体上更一致的梯度方向：LET 参数组初始 mean-gradient cosine 为 `a-b=+0.0839`、`a-c=+0.0707`、`b-c=+0.1354`，明显好于 LWC 的 `+0.0775/+0.0173/+0.0275`。因此 **LET 确实有机会扩大共同可行空间**。但 LET 方向仍由 SID_c 主导：c 的 mean-gradient norm 是 a 的 `6.62×`，逐样本联合 LET 梯度仍会在 `22.66%` 的样本上增大 a loss。LET 不是自动解决方案，需要配合 slot 平衡、验证集或梯度冲突处理。

另一个很强的信号是跨样本梯度一致性：纯 LWC 初始化时，`||mean(g)|| / mean(||g||)` 在 a/b/c 上分别为 `70.5%/12.0%/10.4%`。SID_b/c 梯度在不同条件前缀间高度异质、相互抵消，而 SID_a 梯度更稳定。这与此前“SID_c calibration CE 大幅下降，但测试 KL 不改善”的泛化现象一致。

## 实验设计

- 第 0–26 层恢复自 `omniquant_asym_lwc_w4a16_ad3000` MSE-LWC checkpoint。
- FP teacher 使用原始 BF16 final block、final norm 和 LM head。
- 输入为 `prompt + a_gt + b_gt`，最后三个位置分别预测 SID_a/b/c；每个位置在自己的 8192-token SID 子词表内计算 soft-label CE。
- 对每条样本分别计算 `g_a=∇L_a`、`g_b=∇L_b`、`g_c=∇L_c`，不执行 optimizer step。
- 等权联合梯度定义为 `g_total=(g_a+g_b+g_c)/3`。
- 同时计算逐样本梯度统计，以及先对 128 条样本求平均后的 full-calibration objective gradient。

比较三个状态：

| 状态 | LWC 参数 | LET 参数 | 说明 |
|---|---:|---:|---|
| `lwc_init` | 40,960 | 0 | asymmetric LWC 默认初始化 |
| `abc_trained` | 40,960 | 0 | 已训练的 ABC-LFQ layer-27 checkpoint |
| `lwc_let_ones_init` | 40,960 | 5,120 | 与 `lwc_init` 相同前向起点，新增 qkv/mlp/vo LET log-scale |

## 完整性检查

LET ones 与纯 LWC 初始点的三个 CE 完全相同，证明 LET 对照没有改变初始前向模型；ABC checkpoint CE 与原训练日志完全一致。

| 状态 | SID_a CE | SID_b CE | SID_c CE |
|---|---:|---:|---:|
| LWC init | 6.283825 | 3.891525 | 3.505408 |
| LWC+LET ones init | 6.283825 | 3.891525 | 3.505408 |
| ABC trained | 6.249006 | 3.785313 | 2.853334 |

三个状态完成 128 条、每条三次 slot backward 的耗时分别为 `3.16 s`、`3.06 s`、`3.48 s`，没有缺失梯度或非有限值。

## Full-calibration mean gradient

先在 128 条样本上分别平均每个 slot 的梯度，再计算 cosine。这最接近完整 calibration objective 在当前参数点的局部几何。

| 状态/参数组 | `||g_a||` | `||g_b||` | `||g_c||` | cos(a,b) | cos(a,c) | cos(b,c) |
|---|---:|---:|---:|---:|---:|---:|
| LWC init / LWC | 1.868e-4 | 1.987e-4 | 5.262e-4 | +0.0775 | +0.0173 | +0.0275 |
| ABC trained / LWC | 8.217e-4 | 1.022e-3 | 2.841e-3 | **-0.1915** | **-0.0984** | **-0.0301** |
| LET ones init / LET | 5.580e-3 | 1.555e-2 | 3.694e-2 | +0.0839 | +0.0707 | +0.1354 |
| LET ones init / LWC+LET | 5.583e-3 | 1.555e-2 | 3.694e-2 | +0.0838 | +0.0707 | +0.1354 |

LWC 与 LET 参数的数值参数化和学习率不同，不能用两组 raw norm 的绝对大小直接预测 Adam 的实际 step 大小；但同一参数组内部的 slot norm 比例和 cosine 可以比较。

### 与等权总梯度的兼容性

正 cosine 表示沿 `-g_total` 做足够小的更新时，该 slot 的 calibration mean loss 会下降。

| 状态/参数组 | cos(a,total) | cos(b,total) | cos(c,total) | c/a norm ratio |
|---|---:|---:|---:|---:|
| LWC init / LWC | 0.349 | 0.376 | **0.884** | 2.82× |
| ABC trained / LWC | 0.116 | 0.262 | **0.918** | 3.46× |
| LET ones init / LET | 0.222 | 0.491 | **0.920** | 6.62× |

训练后虽然 pairwise cosine 已经全部为负，但三个 `g_slot·g_total` 在 128-sample mean 上仍为正，因为 c 的梯度足够大，联合方向尚能小幅降低三项平均 loss。不过 a 与 total 的 cosine 只剩 `0.116`，表示联合更新对 a 的有效分量很小，绝大部分优化预算继续服务 c。

## 逐样本梯度冲突

下表的 pair negative rate 是单条样本上两个 slot 梯度点积小于 0 的比例；`slot↔total conflict` 是沿该样本联合梯度更新会局部增大对应 slot loss 的比例。

| 状态/参数组 | a-b negative | a-c negative | b-c negative | a↔total conflict | b↔total conflict | c↔total conflict |
|---|---:|---:|---:|---:|---:|---:|
| LWC init / LWC | 38.28% | 46.88% | 46.09% | 6.25% | 0.78% | 0.00% |
| ABC trained / LWC | **58.59%** | **56.25%** | **51.56%** | **36.72%** | 3.91% | 0.00% |
| LET ones init / LET | 39.84% | 50.00% | 40.63% | 22.66% | 0.00% | 0.00% |

初始化时 pairwise negative rate 接近随机正交方向，但等权 total 几乎总能降低 b/c，原因仍是 b/c、尤其 c 的梯度尺度更大。训练后，a 与 b/c 的负冲突率和 a↔total conflict 同时上升，说明这不是只有尺度不平衡，而是优化过程中逐渐到达了真实的多目标折中边界。

## 跨样本梯度一致性

定义 coherence：

`coherence = ||mean_i(g_i)|| / mean_i(||g_i||)`。

值越低，表示不同 calibration 样本要求的更新方向越分散、平均时抵消越严重。

| 状态/参数组 | SID_a | SID_b | SID_c |
|---|---:|---:|---:|
| LWC init / LWC | **70.5%** | 12.0% | 10.4% |
| ABC trained / LWC | **58.3%** | 11.8% | 12.0% |
| LET ones init / LET | **70.1%** | 14.0% | 10.7% |

SID_a 的梯度在不同 prompt 间高度一致；SID_b/c 因为条件依赖 `a_gt` 和 `a_gt,b_gt`，不同 SID 前缀产生的方向高度分散。使用 128 条、batch size 1、重复 10 epoch 时，优化器容易跟随 b/c 的大幅单样本梯度拟合 calibration 条件，却很难学到能泛化到新前缀的统一方向。

## 对 LET 的判断

梯度证据支持尝试 `ABC + asymmetric LWC + learned LET (ones init)`：

- LET 增加的是 input-channel/weight-column 重参数化方向，与仅调整每个 output row 上下截断界的 LWC 互补。
- LET mean gradients 的三个 pairwise cosine 均为正，尤其 a-c 和 b-c 明显高于 LWC，说明新增空间中存在更兼容的共同下降方向。
- LET 只增加 5,120 个参数（相对 LWC `+12.5%`），但改变的是量化误差在 channel 间的分配，表达价值高于参数数量本身。

但直接保持等权 ABC 并不能保证三者共同泛化：

- LET 内部 c/a mean-gradient norm ratio 由 LWC 的 `2.82×` 上升到 `6.62×`；c 主导可能更强。
- LET 的 a↔total 单样本冲突率为 `22.66%`，高于纯 LWC 初始点的 `6.25%`。
- 新增参数也增加对 128 条 calibration 前缀过拟合的空间。

因此更稳妥的后续顺序是：先以相同 128 条数据跑一个受控的 LET-ones ABC 对照，同时保留独立 validation 监控三个 slot KL；如果 c 继续快速下降而 a 停滞，应采用 slot 梯度归一化、PCGrad，或提高 a/降低 c 的权重，而不是单纯增加 epoch。当前默认 `let_lr=1e-3` 已比 `lwc_lr=1e-2` 小 10 倍，适合作为第一版起点，但 raw gradient 结果仍建议密切观察每个 slot 的单独 loss。

## 解释边界

- 本实验测量的是训练所用 STE 的 raw Euclidean gradient；Adam 的 per-parameter preconditioning 会改变实际更新几何。
- LET 只在 scale=1 初始点测量，没有实际训练，因此只能证明存在新的局部方向，不能直接保证最终测试 KL 提升。
- Pairwise cosine 只描述局部一阶关系；非凸量化优化中，后续 clipping/scale 变化可能改变冲突结构。
- 结论针对当前 1.7B、W4A16 asymmetric LWC、MSE prefix 0–26 和这 128 条 AD calibration 样本。

本报告只保留聚合统计；一次性诊断代码、冒烟输出和完整中间 JSON 已在实验后删除。
