# FP8-W8A8 全局低秩残差补偿 Probe

## 1. 结论

这个 probe 得到了一个“局部成立、末端直接补偿不成立”的结果。

- **低秩补偿确实能恢复本层量化误差，而且能泛化到留出样本。** 对同一个 FP 输入，rank-16 的 activation-weighted SVD 在 32 条留出样本上恢复了 **34.39%** 的线性层输出残差；rank-64 恢复 **39.06%**。训练集和留出集的结果几乎一致，说明这部分不是校准集记忆。
- **只对权重差做普通 SVD 明显不够。** rank-16 只能恢复 **9.49%** 的局部残差。激活协方差决定了哪些权重误差真正影响当前数据分布，这正是 MASQuant CMC/QERA 类 activation-aware 分解有价值的原因。
- **单独在某个后层拟合已经累积的 FP/量化输出差异不泛化。** `actual_output_rrr` 在拟合集上 rank-16 能恢复 **18.07%**，但在留出集上变成 **-65.71%**；rank 越大越差。这说明用 128 条校准样本直接学习“所有前层误差传播后的补偿映射”非常容易过拟合。
- **后层的主要误差已经不是本层权重 QDQ 本身，而是输入轨迹漂移。** 在所选模块总体上，本层局部残差能量只占实际传播残差能量约 **5.68%**；第 27 层约 **4.99%**。因此一个只放在最后几层的 LoRA 补偿分支，即使完整修复本层权重误差，也不可能消除前面已经积累的偏差。

因此，probe 支持继续研究低秩补偿，但支持的是下面这种形式：

> 逐 block、按前向顺序拟合 activation-aware 的低秩局部残差，并在拟合下一层之前应用/冻结前层补偿，持续阻止误差轨迹漂移。

它不支持“只在最后一层用少量校准样本拟合最终累计误差”的方案。现阶段也没有必要先引入 token 范围、SID slot 或 token-channel 路由；先验证顺序式全局补偿能否把 hidden/logit MSE 和推荐指标一起拉回更重要。

## 2. Probe 设置

| 项目 | 设置 |
|---|---|
| 模型 | OneRec Qwen3-1.7B，BF16 teacher |
| 量化模型 | naive FP8-E4M3 weight / FP8-E4M3 activation fake QDQ |
| 权重量化 | per-output-channel |
| 激活量化 | dynamic per-token，`shared_input`，与当前正式评测实现一致 |
| 数据 | AD calibration 128 |
| 数据划分 | 前 96 条拟合，后 32 条独立验证 |
| token | 每条 prompt 在完整序列上均匀取最多 64 个位置；不按 prompt/SID、slot 或通道分类 |
| block | 0、2、7、14、21、27 |
| Linear | Q/K/V/O、gate/up/down，共 42 个模块 |
| rank | 1、2、4、8、16、32、64 |
| 低秩支路输入 | 量化 Linear 之前的 BF16 raw activation；主分支仍做 FP8 activation QDQ |

这里同时定义两种残差：

1. **局部残差**

   \[
   R_{local}=X_{fp}W_{fp}^{T}-Q_a(X_{fp})W_q^{T}
   \]

   它只测当前 Linear 的 W8A8 量化误差，不混入前层传播误差。

2. **实际传播残差**

   \[
   R_{actual}=X_{fp}W_{fp}^{T}-Q_a(X_q)W_q^{T}
   \]

   其中 `X_q` 来自整个 naive W8A8 模型的真实前向轨迹，所以包含前层误差造成的输入漂移。

恢复率统一定义为：

\[
\text{recovery}=1-\frac{\lVert R-\Delta Y_r\rVert_F^2}{\lVert R\rVert_F^2}
\]

恢复率为 0 表示没有改善，1 表示完全恢复；负数表示补偿后反而更差。汇总值按各模块残差能量加权，而不是简单平均百分比。

## 3. 比较方法

| 方法 | 含义 |
|---|---|
| Weight SVD | 直接对 `W_fp - W_q` 做截断 SVD，不看激活分布 |
| Activation-weighted SVD | 用拟合集激活协方差对白化后的权重残差做截断 SVD，再映射回原空间 |
| Local output RRR | 直接在拟合集回归 `R_local` 的低秩映射 |
| Propagated output RRR | 直接在拟合集用 `X_q` 回归 `R_actual`，作为“拟合累计误差”的上界/过拟合检查 |

`RRR` 是 reduced-rank regression。所有低秩映射都可写成标准 LoRA 形式 `X B A`，这里只离线求解最优/近似最优因子，没有更新原模型参数。

### 3.1 Activation-aware SVD 的两步公式

首先定义权重量化误差 `ΔW = W_fp - W_q`，用校准激活 `X` 构造带 ridge 的激活协方差，并对激活加权后的权重误差做 rank-`r` 截断 SVD：

\[
C_X=\frac{1}{N}X^\top X+\lambda I=LL^\top,
\qquad
\Delta W L\approx U_r\Sigma_rV_r^\top.
\]

然后将分解结果从激活加权空间映射回原权重空间，并拆成 LoRA 形式的两个低秩因子：

\[
\Delta W_r
=U_r\Sigma_rV_r^\top L^{-1}
=\underbrace{U_r\Sigma_r^{1/2}}_{B}
 \underbrace{\Sigma_r^{1/2}V_r^\top L^{-1}}_{A}
=BA.
\]

其中：

- `X` 是 calibration token 的输入激活，`C_X` 表示这些激活在各输入通道上的重要程度；
- `L` 是 `C_X` 的 Cholesky 因子，`λI` 用于提高数值稳定性；
- `U_r、Σ_r、V_r` 是只保留前 `r` 个奇异方向的 SVD；
- `BA` 是最终加入量化 Linear 的低秩补偿权重，推理时旁路输出为 `X A^T B^T`。

因此，该分解近似优化的是

\[
\min_{\operatorname{rank}(BA)\le r}
\left\|X(\Delta W-BA)^\top\right\|_F^2,
\]

而不是普通 Weight SVD 所优化的 `||ΔW-BA||²_F`。直观上，它优先保留“在真实 calibration 激活下会产生较大输出误差”的权重误差方向。

## 4. 核心数值

### 4.1 留出集局部残差恢复率

| 方法 | rank 1 | rank 4 | rank 16 | rank 64 |
|---|---:|---:|---:|---:|
| Weight SVD | 6.77% | 8.10% | 9.49% | 13.00% |
| Activation-weighted SVD | **20.12%** | **29.08%** | **34.39%** | **39.06%** |
| Local output RRR | 17.52% | -23.51% | -144.56% | -205.95% |

最重要的是 activation-weighted SVD：rank-16 低秩参数量平均约为原 Linear 权重的 **1.56%**，但能在留出样本上恢复约三分之一的局部输出残差；rank-64 参数比例约 **6.25%**，收益只再增加约 4.7 个百分点，说明 rank-16/32 更可能是后续实验的合理起点。

直接输出回归在 rank 增大后迅速过拟合，说明“只要把训练 MSE 降下来”并不足以证明补偿有效。相反，activation-weighted 权重分解利用了确定的权重误差结构，训练/留出结果更稳定。

### 4.2 实际传播残差恢复率

| 方法 | rank 1 | rank 4 | rank 16 | rank 64 |
|---|---:|---:|---:|---:|
| Weight SVD | 0.33% | 0.43% | 0.51% | 0.71% |
| Activation-weighted SVD | **0.98%** | **1.61%** | **1.89%** | **2.14%** |
| Local output RRR | 0.86% | -1.68% | -8.80% | -12.28% |
| Propagated output RRR（拟合集） | 7.47% | 11.83% | 18.07% | 30.66% |
| Propagated output RRR（留出集） | -38.95% | -51.90% | -65.71% | -84.81% |

这里的低数字不能解读为“局部低秩补偿完全没用”。当前计算是在前面所有层仍保持 naive W8A8 的条件下，**单独**评估某一个 Linear 的补偿。到了后层，`X_q` 已经偏离 `X_fp`，修复当前权重误差只能覆盖总偏差的一小部分。

### 4.3 误差传播随深度增长

| block | 局部残差 / 实际传播残差（能量） | rank-16 activation-weighted SVD：局部恢复 | 同一补偿：实际传播恢复 |
|---:|---:|---:|---:|
| 0 | 30.50% | 28.15% | 8.45% |
| 2 | 39.82% | 92.17% | 40.87% |
| 7 | 11.41% | 29.04% | 3.37% |
| 14 | 7.04% | 32.12% | 2.32% |
| 21 | 10.86% | 28.81% | 3.11% |
| 27 | 4.99% | 33.37% | 1.57% |

第 0 层 Q/K/V 输入还没有前层量化漂移，其 rank-16 实际恢复率分别为 36.6%、39.1%、48.0%。同一方法到了第 27 层大多只恢复约 0.7%–1.3%，不是因为第 27 层局部权重误差突然不再低秩，而是因为总偏差已经被输入漂移支配。

## 5. 图

![不同方法在留出集上的实际传播残差恢复率](../artifacts/results/fake_quant/probes/fp8_w8a8_low_rank_residual_ad128/actual_val_recovery_by_rank.png)

![拟合集与留出集对比](../artifacts/results/fake_quant/probes/fp8_w8a8_low_rank_residual_ad128/actual_train_vs_val_recovery.png)

![rank-16 propagated-output RRR 逐层热力图](../artifacts/results/fake_quant/probes/fp8_w8a8_low_rank_residual_ad128/actual_rrr_rank16_heatmap.png)

## 6. 对方案设计的含义

1. **优先采用 activation-aware 的确定性低秩初始化。** 普通 Weight SVD 丢掉了输入分布信息；activation-weighted SVD 明显更强且不容易过拟合。
2. **必须阻止误差逐层累积。** 后层直接拟合最终残差需要学习 `X_q -> X_fp W_fp` 之间高度样本相关的映射，128 条样本不足以支持这种自由拟合。
3. **下一版仍然可以保持“全局残差”而不做 token 特化。** 建议先在每个 block 的 7 个 Linear 上加入 rank-16（可补 rank-32）的 BF16 低秩支路；从 layer 0 到 27 顺序求解/短暂优化，前层完成后冻结并用补偿后的量化轨迹收集下一层输入。
4. **下一次验证必须包含端到端量。** 除逐层 hidden MSE 外，至少比较最终 hidden-state MSE、全词表 logit KL，以及 AD 子集/全集指标。只有端到端推荐指标改善，才能把“局部残差可恢复”升级为“精度补偿有效”。
5. **暂不加入 routed expert 或 token/channel 选择。** 这些机制解决的是容量分配问题；当前首要问题是先证明顺序式单专家低秩补偿能阻断传播误差。

## 7. 文件与复现

- Probe 实现：已删除；实验结果与本文档保留
- 完整逐模块记录：`artifacts/results/fake_quant/probes/fp8_w8a8_low_rank_residual_ad128/records.csv`
- 聚合记录：`artifacts/results/fake_quant/probes/fp8_w8a8_low_rank_residual_ad128/aggregates.csv`
- 配置与模块统计：`artifacts/results/fake_quant/probes/fp8_w8a8_low_rank_residual_ad128/summary.json`

低秩 probe 与补偿分支的实现代码已在 2026-08-18 的仓库清理中删除；本节保留结果文件位置与结论，作为历史负向证据。

## 8. 限制

- 这是结构性 probe，不是已经训练并插入模型的 LoRA 补偿实现，也没有直接报告推荐 metric。
- 只选了 6 个代表 block，而不是 28 个 block 全覆盖。
- 每条 prompt 均匀采样最多 64 个 token，覆盖完整序列范围，但不是保存每个位置；这是为了把 42 个大矩阵的协方差分解控制在可接受范围内。
- 当前低秩支路假设以 BF16 raw activation 计算。如果最终要求补偿支路也使用 FP8，需要重新测一次可恢复率。
- `Propagated output RRR` 的负泛化结果同时受到小校准集、高维输入和前层轨迹漂移影响；它是对“直接拟合累计误差”的否定证据，不代表所有经过正则化或逐层训练的低秩补偿都会失败。
