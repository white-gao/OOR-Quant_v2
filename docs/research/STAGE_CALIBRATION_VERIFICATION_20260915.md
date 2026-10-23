# 固定 SID 阶段：校准覆盖与重建权重验证

日期：2026-09-15。状态：四组 smoke test、12 个训练臂、四组训练集/留出集诊断均已完成，GPU 2–5 已释放。新增产物合计约 40 MB。

**结论：本轮不支持把“补齐两步输入 + 简单阶段等权 MSE”直接升级为全前缀方案。**
单纯补齐输入影响很小；阶段等权确实能拟合训练集的 B/C 误差，但没有迁移到留出集。
A 的改善更容易泛化，不能据此把 B/C 的瓶颈简单归结为“参数完全学不动”。

## 问题与范围

接续 [上一轮探针](FIXED_SID_STAGE_PROBES_20260915.md)，验证两个不同因素：

1. MSE 校准是否需要显式覆盖本次请求的 B/C 决策状态？
2. 补齐输入后，历史长度是否仍让这些状态在重建目标中被稀释？

本轮是单 block continuation pilot，不是全模型重训，也不是完整推荐 benchmark。
选择 **block 26（倒数第二层）**，因为此前已发现它的 B/C 累计误差明显，而且它尚未经过最后层 LFQ 优化。
block 0–25 和 block 27 保持原始 FlatQuant checkpoint 不变。冻结的前缀可能限制恢复空间；冻结的最后层也可能掩盖局部改善。

## 配对协议

| 项目 | 设置 |
|---|---|
| 模型与任务 | Qwen3-1.7B，AD / Video |
| 精度 | W4A8 = FP4 E2M1 / FP8 E4M3；W8A8 = FP8 E4M3 / FP8 E4M3 |
| 粒度、计算 | per-output-channel，BF16 deployment-matched fake-QDQ |
| 初始化 | 对应纯 FlatQuant MSE baseline 的同一 block 26 |
| 训练 | calib `[0,128)`，5 epochs，seed 42，固定样本顺序 |
| 留出 | calib `[768,896)`，128 条，不用于训练或选 epoch |
| 更新参数 | 同一组矩阵、对角 scale、LWC、LAC；每组独立重置 AdamW |
| 学习率等 | 继承源 checkpoint；cosine schedule；MSE 梯度归一化沿用原实现 |
| checkpoint 选择 | 最后 epoch，不用 best epoch |
| 物理 GPU | 2=AD W4A8，3=AD W8A8，4=Video W4A8，5=Video W8A8 |

留出数据未用于这些 baseline 的训练，但属于以前的 validation tail，上一轮也已观察过；**不是盲测集**。不能拿它评估使用完整 calib1024 训练的 RecQuant。

| 实验 | 输入 | 重建 loss |
|---|---|---|
| baseline | 不增训 | 仅提供参照 |
| prefill_mse | prompt，以 `<sid_begin>` 结尾 | 全 token MSE |
| complete_mse | 同一 prompt + GT A + GT B | 全 token MSE |
| balanced_mse | 与 complete_mse 完全相同 | history / A / B / C 各自平均 MSE，再等权平均 |

阶段以**当前输入位置预测什么**命名：完整输入最后三个位置预测 A/B/C，之前均为 history。
不做参考能量归一化，不增加新参数，也不改变推理中的量化位置、精度、矩阵或权重数量。
四个权重都是 0.25，不表示各阶段实际 loss 或梯度贡献必然各占 25%。

FP teacher target 来自完整 BF16 前缀；student block 输入来自冻结的量化前缀，保持原有累计误差对齐方式。
两条 stream 分开运行和缓存；不互换 FlatQuant 变换坐标系中的 KV。

## 指标和判定

- 相同 GT 前缀、相同输入形状下的 A/B/C CE、KL、top-32 保留率和边界诊断。
- block 25、26、27 的 history/A/B/C 累计残差流 MSE，同时记录绝对值和 pooled 相对误差。
- `complete_mse - prefill_mse` 隔离覆盖效应；`balanced_mse - complete_mse` 隔离本轮固定权重的效应。
- KL 和保留率提供按样本配对的 bootstrap 95% CI，2,000 次、seed 42；探索性结果未做多重比较校正。

训练日志中的 `*_mse` 对 balanced_mse 指的是加权重建值，不能跨组直接比较这些训练标量。
应比较相同留出指标。降低某个阶段的 hidden MSE，不自动等于提高 SID 分布对齐，更不等于提高全量 recall@32。

## 结果

### 1. 留出集 A/B/C 条件 top-32 保留率

单位 %。这不是完整商品的 recall@32；各阶段使用同一个 GT 前缀。

| 模型 | 实验 | A | B | C |
|---|---|---:|---:|---:|
| AD W4A8 | baseline | 86.890 | 69.897 | 42.017 |
| | prefill_mse | 87.183 | 69.482 | 41.553 |
| | complete_mse | 87.183 | 69.800 | 41.968 |
| | balanced_mse | 89.478 | 69.751 | 41.162 |
| AD W8A8 | baseline | 93.115 | 88.037 | 70.435 |
| | prefill_mse | 93.652 | 87.573 | 70.825 |
| | complete_mse | 93.701 | 88.403 | 70.483 |
| | balanced_mse | 93.896 | 87.671 | 70.508 |
| Video W4A8 | baseline | 85.474 | 80.493 | 54.297 |
| | prefill_mse | 85.034 | 80.127 | 54.077 |
| | complete_mse | 84.839 | 80.396 | 53.882 |
| | balanced_mse | 85.889 | 79.541 | 54.150 |
| Video W8A8 | baseline | 92.944 | 91.943 | 78.687 |
| | prefill_mse | 93.530 | 91.748 | 78.516 |
| | complete_mse | 93.457 | 91.724 | 78.491 |
| | balanced_mse | 93.164 | 91.431 | 77.954 |

覆盖效应 `complete - prefill` 的 C 保留率变化依次为 +0.415 / -0.342 / -0.195 / -0.024 pp，四个配对 CI 均跨 0。
阶段等权相对 complete 的 A 保留率在 AD W4 增长 2.295 pp（CI [1.685, 2.905]），Video W4 增长 1.050 pp（CI [0.391, 1.733]）。
但 AD W4 的 C 下降 0.806 pp（CI [-1.587, -0.049]），AD W8 / Video W4 的 B 分别下降 0.732 / 0.854 pp，区间也不跨 0。
没有一组给出稳定的 B/C 同步提升。以上仍是未经多重校正的探索性区间。

KL 也没有给出一致的 B/C 改善：

| 模型 | baseline 的 A / B / C KL | balanced 的 A / B / C KL |
|---|---|---|
| AD W4A8 | 0.030168 / 0.443829 / 1.570615 | 0.022171 / 0.475758 / 1.614675 |
| AD W8A8 | 0.007712 / 0.050370 / 0.174366 | 0.007523 / 0.050929 / 0.179747 |
| Video W4A8 | 0.027198 / 0.118082 / 0.615730 | 0.022740 / 0.119034 / 0.620127 |
| Video W8A8 | 0.004652 / 0.015495 / 0.110390 | 0.004792 / 0.015977 / 0.112240 |

### 2. 不是 B/C 完全无法拟合，而是训练改善未迁移

训练结束后重新加载各臂 checkpoint，在相同的 128 条训练数据上做固化模型前向，无优化器、无增训。
下表为 balanced 相对 baseline 的 **block 26 绝对 MSE 均值变化百分比**；负数表示误差下降。

| 模型 | 训练 A | 留出 A | 训练 B | 留出 B | 训练 C | 留出 C |
|---|---:|---:|---:|---:|---:|---:|
| AD W4A8 | -42.34% | -40.19% | -15.96% | +0.07% | -15.68% | +1.12% |
| AD W8A8 | -8.03% | -6.85% | -7.02% | +2.61% | -8.11% | +0.69% |
| Video W4A8 | -18.96% | -15.80% | -6.32% | +6.23% | -15.40% | +2.06% |
| Video W8A8 | -4.60% | -1.91% | -2.32% | +4.42% | -8.16% | +1.48% |

四阶段等权 MSE 在训练集分别下降 13.58% / 6.29% / 10.38% / 5.24%，留出集反而增加 0.29% / 1.16% / 1.96% / 1.81%。
例如 AD W4，训练集 C 保留率从 42.480% 升至 45.166%，留出却从 42.017% 降至 41.162%；B 的训练保留率也有提升，而留出没有。

这与“对训练状态做了可拟合、但 B/C 泛化较弱的修正”一致，并非只有 STE 日志下降、固化后完全消失。
它不能单独证明增加样本就能解决，也不能排除参数耦合、前缀条件多样性、正则化和优化超参的影响。

### 3. 排除一个误解：A 不是绝对 loss 最大的一项

baseline 在留出集的阶段等权重建中，实际误差份额为：

| 模型 | history | A | B | C |
|---|---:|---:|---:|---:|
| AD W4A8 | 12.65% | 4.07% | 23.68% | 59.60% |
| AD W8A8 | 12.00% | 5.04% | 18.15% | 64.80% |
| Video W4A8 | 17.74% | 7.10% | 13.84% | 61.31% |
| Video W8A8 | 17.96% | 6.23% | 11.52% | 64.29% |

因此不能用“A 的绝对 loss 太大、把 B/C 淹没”解释这个等权实验。
这里记录的是误差份额，不是梯度份额；更符合当前证据的描述是：**A 的修正更具跨样本一致性，B/C 的训练收益更容易局限于见过的状态。**
后一句是基于训练/留出差异的解释，不是已完成敏感子空间或因果分解的证明。

### 4. 有效性检查与边界

- 四组 baseline 的 A/B/C CE、KL、top-32 与上轮 128 条结果逐项完全一致（最大差 0）。
- 四臂 block 25 的逐样本/阶段误差记录完全一致（最大差 0），符合冻结前缀的设计。
- 本轮验证是在 block 26 就出现 B/C 泛化失败，不仅是最后 block 27 抵消了局部收益；但冻结上游仍限制了可恢复空间。
- 无 NaN/Inf；部分 balanced 训练出现大梯度尖峰，例如 AD W8 最大范数约 4758。没有临时更改学习率或 clipping 来追求正结果，尚不能排除优化稳定性的影响。
- 本轮没有重训整个前缀、没有加入最后层 RecQuant、没有做新完整 beam/全量推荐测评。
- 本轮不能替代最后 block 的严格 MSE-1024 decode-complete 对照，因此也不能彻底解决历史 LFQ 实验的输入覆盖归因问题。

**下一步建议：暂停扩大这套简单覆盖/等权方案。** 若继续固定阶段方向，应先验证哪些 B/C 修正跨样本、跨条件前缀稳定，再考虑共享变换的结构约束；目前不支持直接再加阶段参数、调权重或宣称全模型收益。

## 代码、结果与清理

入口：`flat_quant/verify_stage_calibration.py`。
启动：`bash scripts/flat_quant/run_stage_calibration_verification_cuda2345.sh`。
启动脚本拒绝占用忙碌 GPU 或覆盖已有结果目录。日志包含捕获/诊断进度条及逐 epoch 训练值。
正式训练入口的默认行为不变；仅给内部 MSE 训练器增加一个可选重建 reduction 回调，复用原优化器、调度器和固化。

本轮结果根目录：

```text
artifacts/results/flat_quant/stage_calibration_verification/
  smoke_20260915/                                  # 2 train / 2 heldout / 1 epoch
  block26_train128_heldout128_epoch5_20260915/
    ad_w4a8/ ad_w8a8/ video_w4a8/ video_w8a8/
```

每组保留 `protocol.json`、`summary.json`、逐样本诊断、各臂训练曲线及 `complete.json`。
每臂仅保存改变的 `layer_26.pt`；其余层仍从原始 `checkpoint_dir` 加载，不能把这个单层目录当成完整模型目录。

`train_summary.json`、`train_*_samples.json`、`train_complete.json` 是新增的只读训练集诊断，不能当作 held-out 结果。
复查训练集时，对完成的目录使用相同参数及 `DIAGNOSE_TRAIN=1 RUN_ROOT=... bash scripts/flat_quant/run_stage_calibration_verification_cuda2345.sh`；已有诊断会拒绝覆盖。

上一轮的 `probe_fixed_sid_stages.py`、对应 launcher 和 test 已移除，结果及报告保留。
源码可从 `artifacts/results/flat_quant/fixed_sid_stage_probes/legacy_probe_source_20260915.tar.gz` 恢复。

验证记录：新测试 7 项 + FlatQuant 核心/集成 16 项通过；分片测试 1 项通过；四组真实模型 smoke test 完成。
其他部分旧测试仍导入工作区中已删除的 `fake_quant` 包，收集时报错；本轮没有恢复这些无关的删除。
