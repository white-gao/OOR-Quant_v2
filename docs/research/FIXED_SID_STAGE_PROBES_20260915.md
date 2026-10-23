# 固定 SID 三阶段量化：探针与下一步方向

日期：2026-09-15。模型为 OneRec/Qwen3-1.7B，28 blocks（0–27）。

状态：四组 128 条固定前缀复核、四组 64 条最终 beam/逐层画像、AD test32 的四组方法检查均已完成。GPU 2–5 已释放。所有探针及调试产物合计约 567 MB，没有写入新的模型 checkpoint。

## 核心判断

目前最值得优先验证的不是增加三套 LAC，而是 **decode-complete、stage-balanced 的共享量化校准**：现有 MSE 标定没有显式覆盖当前推荐请求的 B/C 决策状态；即使补齐两个输入 token，普通全 token MSE 仍主要由历史位置贡献。可以保留一套 FlatQuant 矩阵、对角缩放、LWC/LAC 和量化权重，仅改变这些参数接触的校准状态及重建误差的统计方式。

这是有代码与误差画像支持的研究假设，**尚未通过重新训练证明收益，不是已经完成的新算法**。本轮没有修改原训练入口、已有 checkpoint、正式评测设置或权重量化粒度。

后续更新：已完成 [block 26 的覆盖/阶段等权训练验证](STAGE_CALIBRATION_VERIFICATION_20260915.md)。简单方案没有带来稳定的 B/C 留出收益；能降低训练误差，但出现明显泛化差距。因此下文的优先方向应理解为当时待验证的假设，不是已确认有效的方法。

## 协议与结果入口

- 使用物理 GPU 2/3/4/5，对应 AD W4A8、AD W8A8、Video W4A8、Video W8A8；不操作 0/1/6/7 上的任务。
- 四组都加载纯 FlatQuant MSE baseline：per-output-channel；prefix 0–26 使用 128 条，block 27 使用 512 条，15 epochs。
- W4A8：FP4 E2M1 权重 + FP8 E4M3FN 激活。W8A8：两者均为 FP8 E4M3FN。计算路径为 BF16 fake-QDQ，不代表真实低比特 kernel 的时延或显存。
- 首轮筛选：calib `[512,544)`，32 条，14 种激活干预。
- 固定前缀复核：calib `[768,896)`，128 条；输入为 prompt + 首个 GT SID 的 A/B。
- 逐层画像和 beam 复核：calib `[768,832)`，64 条；beam=32，生成三步，不额外限制 SID 词表（词表约束仅作为单独诊断）。
- 数据按 loader 的确定性 `head(n)` 及 offset 切分。这些样本未用于对应 baseline 的训练，但属于此前的 validation tail，**不是从未观察过的独立 test**。也不能作为 train1024 RecQuant checkpoint 的 held-out 数据。
- 本轮主体测的是纯 FlatQuant baseline，不能直接推广为 FlatQuant+RecQuant 的残余瓶颈。

结果统一位于 `artifacts/results/flat_quant/fixed_sid_stage_probes/`：

| 内容 | 子目录 | 文件 |
|---|---|---|
| 32 条、14 种激活干预 | `discovery_20260915/<task_precision>/` | `screen_summary.json`、`screen_samples.json` |
| 128 条固定前缀复核，本文激活表来源 | `confirmation_hf_20260915/<task_precision>/` | `screen_summary.json`、`screen_samples.json` |
| 64 条逐层画像与最终 HF beam | `final_20260915/<task_precision>/` | `layer_profile_summary.json`、`beam_summary.json`、逐样本 JSON、`complete.json` |

`<task_precision>` 为 `ad_w4a8`、`ad_w8a8`、`video_w4a8`、`video_w8a8`。

中间调试目录 `confirmation_20260915`、`beam_hf_cached_20260915` 的 beam 没有完成；首轮简化 beam 也不作为最终证据。`confirmation_hf_20260915` 的 128 条 screen 已完成，但其整轮 beam 后来停止以优化耗时。**不要混用这些目录的未完成 beam 均值。**

## 1. 校准状态覆盖：代码中一个重要的控制变量

当前入口 `flat_quant/run_m1_onerec_ad.py`：

1. 加载 calibration 时使用 `drop_final_assistant=True`；
2. 只有训练 `omni_final_objective=lfq_ce` 时，调用 `build_lfq_sid_slot_batches`，显式追加 GT 的 A/B；
3. 普通 MSE 路径调用 `build_model_batches`，输入在 `<|sid_begin|>` 结束；
4. 现有 RecQuant continuation 恢复此前 MSE prefix，只增训最后 block。因此前 27 个 block 的量化参数没有针对本次请求的两个 decode 决策状态重新优化。

已运行 baseline 的 `flatquant_core_config.json` 也记录 `omni_final_objective: mse`，与该路径一致。

| 输入最后位置 | 该位置的预测任务 | 普通 MSE calibration | LFQ calibration |
|---|---|---|---|
| `<sid_begin>` | 预测 A | 有 | 有 |
| 本次输出的 A | 预测 B | 无显式覆盖 | 有 |
| 本次输出的 B | 预测 C | 无显式覆盖 | 有 |

“无显式覆盖”不等于没有见过 SID-A/B 这种 token：历史中当然有 SID。缺少的是**当前请求末端、给定生成前缀的条件决策状态**。这也不是说原 baseline 实现错误；prefill-only 是一种合法校准策略，但尚未确认它是否适合这个三步推荐任务。

重要含义：此前 MSE-1024 与 LFQ-1024 并非只改变了 loss，输入也由 prompt 变成了 prompt+A+B。不能因此否定已有推荐增益，但若要严格归因于新目标，应该再补 **同样 prompt+A+B、仍然只优化 MSE** 的对照。

## 2. 逐层误差：B/C 在最后 block 之前已经变差

比较同一输入下 BF16 与量化模型的 block 输出残差流。这里是全模型累计误差，不是独立单 block 的重建误差，也不是训练 loss 日志。

相对 MSE 定义为：该位置的平方误差总和 / BF16 hidden state 平方能量总和。下表为 64 条样本的 pooled 比值，单位 %。

| 模型 | block | 历史位置 | A 决策位置 | B 决策位置 | C 决策位置 |
|---|---:|---:|---:|---:|---:|
| AD W4A8 | 26 | 2.79 | 0.58 | 11.02 | 32.29 |
| AD W4A8 | 27 | 6.42 | 1.06 | 19.56 | 45.94 |
| AD W8A8 | 26 | 0.53 | 0.15 | 1.72 | 7.14 |
| AD W8A8 | 27 | 1.29 | 0.27 | 3.18 | 11.19 |
| Video W4A8 | 26 | 2.23 | 0.46 | 2.43 | 13.15 |
| Video W4A8 | 27 | 5.87 | 0.72 | 4.71 | 21.28 |
| Video W8A8 | 26 | 0.39 | 0.07 | 0.36 | 2.33 |
| Video W8A8 | 27 | 1.08 | 0.11 | 0.70 | 3.71 |

B/C 相比 A 更差，但不同域的幅度不同。不能仅凭这张表断定缺少 decode calibration 是唯一原因；条件任务难度、输入能量和量化误差传播都可能参与。

即便在探针中补齐两个 decode 输入，若仍对整个序列按 token 平均、再对样本等权平均，block 27 的 ABC 三个位置合计仅贡献：

| 模型 | ABC 占全部平方误差的比例 |
|---|---:|
| AD W4A8 | 1.041% |
| AD W8A8 | 1.121% |
| Video W4A8 | 0.191% |
| Video W8A8 | 0.175% |

因此“补两个 token”与“让优化真正重视这两个状态”是两个不同的问题。该占比是对补齐输入后的诊断统计，**不是声称原 MSE 训练已经包含 B/C**。

另外，block 27 的 C hidden-error 均值能量只占跨样本误差能量约 2%；A 则明显更高。这是误差向量的统计，不是梯度噪声测量；它不支持把 C 的问题简单当成一个固定偏置来修正。

## 3. 激活干预：W4 与 W8 的空间很不一样

干预保留同一份量化权重、学习到的矩阵和对角缩放，只在指定位置、变换之后跳过 activation clipping + QDQ。A16 在此实际表示 BF16 激活路径。它不是训练后的方法，也不是严格的性能上界，因为已有 W/T/LAC 已经联合适应了原量化误差。

下表是 **GT 固定前缀条件下、单个 C 分布的 top-32 保留率**，不是最终商品 recall@32。

| 模型 | 原始 A8 | 仅 B/C 位置 A16 | 全部位置 A16 | B/C 干预增量及 95% 配对 CI |
|---|---:|---:|---:|---|
| AD W4A8 | 42.017% | 43.506% | 43.872% | +1.489 pp [0.732, 2.295] |
| AD W8A8 | 70.435% | 77.466% | 77.466% | +7.031 pp [5.933, 8.130] |
| Video W4A8 | 54.297% | 54.932% | 54.858% | +0.635 pp [-0.073, 1.343] |
| Video W8A8 | 78.687% | 83.911% | 84.692% | +5.225 pp [4.272, 6.201] |

配对 bootstrap 以样本为单位，2,000 次，seed=42。探索了多个干预，这些区间没有作多重比较校正。

对应 C 的 KL：

| 模型 | 原始 A8 | 仅 B/C 位置 A16 | 全部位置 A16 |
|---|---:|---:|---:|
| AD W4A8 | 1.57061 | 1.42758 | 1.43119 |
| AD W8A8 | 0.17437 | 0.08516 | 0.08246 |
| Video W4A8 | 0.61573 | 0.60127 | 0.59500 |
| Video W8A8 | 0.11039 | 0.04850 | 0.04469 |

W8A8 的 B/C 局部激活误差确实值得关注；W4A8 中单独消除激活误差的改善有限，不支持把 activation/LAC 当成当前主要恢复手段。

只恢复整个 prefill 的激活，C 保留率在 AD W8A8 仅 70.532%，Video W8A8 为 79.565%，明显不及直接保护 B/C。它不支持“只要把历史 KV 做准，decode 就自然恢复”的简单解释。缓存没有额外做 KV 低比特压缩，这里讨论的是此前计算带来的缓存误差。

保留 LAC clipping、只跳过舍入时，AD/Video W8A8 的 C 保留率分别为 76.465%/83.740%；与全部 A16 比较接近。说明收益并不主要来自把 LAC 截断完全取消，但这不等于证明任何重新学习的 LAC 都无效。

## 4. Beam 诊断定义与实现校验

每一步直接使用正式 `model.generate()`，通过 logits observer 记录真实运行中的 A、AB、ABC frontier，避免重新实现 `topk` 时的 BF16 并列分数差异。量化和教师各自维护缓存，绝不把不同 FlatQuant 变换坐标系的原始 KV 直接互换。

- `*_beam_retention`：同一深度，量化 beam 与 BF16 beam 的集合交集比例。
- `*_teacher_final_reachable`：BF16 最终 32 个 item 中，其对应前缀仍在量化模型该步 beam 内的比例。按完整 item 计数，不能把不同 item 的共同前缀只算一次。
- `oracle_a/b/c`：只把指定阶段的预测分数替换成 BF16 在**相同当前前缀**下的分数；量化模型内部状态不替换。
- `oracle_bc`：同时替换 B/C，检查联合修复空间。
- `oracle_b/c_offset`：只校正每个父前缀的一项共同分数偏移，保留该前缀内部的全部排序与分数差；偏移取教师 top-32 上的平均 log-prob 误差。它只是一个特定的 oracle 偏移诊断，负面结果不能否定所有跨前缀误差。

这些 oracle 使用了不可部署的教师信息，不应写成实际方法结果，更不能直接换算成下游 recall 提升。64 条的 GT 命中只作 sanity check，不用于方法优劣判断。

缓存复用与未复用的正式 HF 版本已核对 1,584 个中间候选集合，全部一致。复用只发生在同一个 prompt、相同 prefill 数值路径之间，每次使用独立缓存副本；切换 prefill 精度会使用不同缓存键。新探针的五项单元测试通过。

### 64 条最终 beam 结果

先看 BF16 最终 32 个 item 的前缀在量化 beam 中还剩多少：

| 模型 | A 步后可达 | AB 步后可达 | ABC 最终保留 |
|---|---:|---:|---:|
| AD W4A8 | 92.24% | 54.25% | 28.81% |
| AD W8A8 | 97.07% | 81.49% | 70.85% |
| Video W4A8 | 93.36% | 72.07% | 56.49% |
| Video W8A8 | 96.73% | 88.53% | 81.45% |

AD W4A8 中，BF16 最终候选在 A/AB/ABC 三个步骤首次丢失的比例分别为 7.76/37.99/25.44 pp。A 的候选差异不等于有同等比例的最终 item 丢失；最大的绝对丢失量发生在 B 展开。

下表均为最终 ABC top-32 集合保留率。教师替换是 oracle 诊断，不是可部署的量化结果。

| 模型 | 原始量化 | 教师替换 A | 教师替换 B | 教师替换 C | 教师替换 B+C |
|---|---:|---:|---:|---:|---:|
| AD W4A8 | 28.81% | 30.86% | 44.63% | 48.83% | 79.00% |
| AD W8A8 | 70.85% | 74.85% | 79.79% | 76.81% | 87.79% |
| Video W4A8 | 56.49% | 61.47% | 68.60% | 66.06% | 82.71% |
| Video W8A8 | 81.45% | 85.16% | 86.18% | 85.25% | 91.60% |

这支持优先关注 B/C 的条件决策，但不意味着训练能达到 oracle 数值。联合替换的效应也不能当成两个单阶段效应的线性相加。

实际 activation 干预下，最终 ABC 保留率的变化：

| 模型 | 原始量化 | 仅 B/C 激活 A16 | 全部激活 A16 | B/C 干预增量及 95% 配对 CI |
|---|---:|---:|---:|---|
| AD W4A8 | 28.81% | 30.32% | 30.57% | +1.51 pp [0.63, 2.34] |
| AD W8A8 | 70.85% | 76.42% | 79.64% | +5.57 pp [4.30, 6.89] |
| Video W4A8 | 56.49% | 56.74% | 58.30% | +0.24 pp [-0.68, 1.12] |
| Video W8A8 | 81.45% | 84.18% | 85.21% | +2.73 pp [1.66, 3.81] |

与固定前缀复核一致，W8 的激活干预改善更明确；Video W4 的 B/C 激活干预没有稳定改善。

只修正每个父前缀的一项整体分数偏移，没有出现正向稳定收益：B-offset 在 AD W4/W8 为 -2.20/-1.81 pp，在 Video 两组均约 -0.05 pp；C-offset 在四组为 -4.49/-4.25/-5.13/-0.29 pp。这否定的是本次特定偏移方案，而不是证明跨前缀比较不重要。

### 已有 RecQuant 的 AD test32 检查

为避免用 train1024 内的数据评估 RecQuant，另取 AD test `[0,32)`，分别加载同一位宽的纯 MSE baseline 和 `exp3_abc_boundary0.3_joint`。只做固定 GT 前缀与逐层画像，不据此比较完整推荐精度。

| 位宽 / 方法 | C top-32 保留率 | B/C 激活 A16 后 | C KL | block 27 C 相对 MSE |
|---|---:|---:|---:|---:|
| W4A8 FlatQuant | 41.89% | 42.38% | 1.3923 | 42.79% |
| W4A8 + RecQuant | 40.53% | 40.53% | 1.2195 | 58.79% |
| W8A8 FlatQuant | 67.09% | 75.49% | 0.2105 | 12.00% |
| W8A8 + RecQuant | 65.62% | 73.14% | 0.2088 | 16.85% |

W8A8+RecQuant 中，B/C activation 干预仍带来 C 保留率 +7.52 pp 的描述性改善。W4A8+RecQuant 没有显示同样空间。每个位宽的 3,456 项 prefix 0–26 误差记录在两种方法间完全一致，确认这一部分没有被最后层增训改变。

不能据此说 RecQuant 不如 baseline：它已经有先前全量推荐结果，本表只有 32 条、固定首个 GT 前缀的局部指标。相反，这提醒我们：**隐藏层 MSE、条件 top-32 与最终推荐效果不能互相替代**；下一步阶段校准也不能只凭 MSE 下降宣布成功。

该补充结果在 `ad_test32_methods_20260915/{w4a8,w8a8}_{mse,recquant}/`。

## 5. 哪些方向值得做，哪些先不要做

### 优先：覆盖三个阶段，再让共享变换关注它们

最小验证顺序：

1. **覆盖对照**：普通 prefill-only MSE vs. prompt+A+B 的全 token MSE。固定数据、更新参数、epochs、seed，不加新目标或参数。
2. **统计方式对照**：同样的完整输入，将历史、A、B、C 的重建误差分别统计后组合；必要时按各阶段的参考能量归一化，避免历史长度和能量主导目标。保留历史项，不能直接丢弃上下文校准。
3. **与现有方法衔接**：若前两项有效，再使用新的 prefix，最后 block 继续同样的 ABC CE + 0.3 boundary，检查收益能否叠加。

研究对象是现有矩阵/对角缩放/LWC/LAC 的联合校准，不是三套阶段矩阵或三套权重。推理仍是一份静态量化权重和原有 FlatQuant 路径。与过去 LFQ-ALL 的区别是：仍做逐层局部重建，不把全模型的 ABC loss 梯度直接传到所有 block。

这条路线与推荐固定三步的联系在于：可以无歧义、低额外 token 成本地补齐全部决策状态，并把校准预算按决策角色分配。但“补齐 decode calibration”本身不能预先宣称为强创新；只有证明结构性误差、设计优于简单覆盖/加权的机制，并获得 held-out/full-eval 收益后，才能判断论文价值。

如果上述简单对照有效，更有方法空间的候选是 **阶段判别方向约束的共享变换**：不按最大 activation/outlier 选择 channel，而是估计各阶段区分竞争 SID 候选所需的敏感方向，让同一套矩阵减少量化噪声在这些方向上的投影。它不要求三套权重，也不在推理时加入补偿。

这只是下一阶段候选，尚未实现或验证。至少要先检验敏感方向是否低秩、跨样本是否稳定，以及等预算下是否优于随机方向保护。如果 B/C 的敏感方向并不共享、跨样本并集接近全空间，那么“子空间保护”也没有简洁有效的依据，应该停止，而不是继续叠加模块。

### 次优先：W8 的决策激活精度敏感性

B/C 激活干预有清晰信号，可以作为后续 activation 设计的上限参照。但若只做按阶段切换 A8/A16，它是混合精度配置，不是纯 W8A8；若只增加几个 LAC 标量，也没有证据说明能获得上述 A16 的改善。

### 暂缓

- 三套权重、三套不兼容的在线矩阵、额外补偿支路：不符合目前的部署诉求。
- 仅按 raw activation MSE 挑 outlier channel：历史随机对照未支持这种规则。
- 只限制 SID 词表：首轮干预几乎不变，原模型对应 slot 的概率质量已经非常接近 1。
- 直接重做泛化 Beam-KL、再次盲目扩 Top-A/AB 样本：历史已有负面记录；本轮没有证明单纯扩量能解决问题。

## 6. 与已有结论和论文的关系

- 旧的 [SID 阶段探针](SID_Generation_Stage_Probe_Summary.md) 基于 real GPTQ W8A8、AD test1000，曾观察到 Stage-A activation rescue 更强，并发现 error-top-1% channel 不优于随机。不能把其阶段排序直接套到当前 FlatQuant fake-QDQ，也不能用本轮小样本分布指标否定原来的真实推荐指标。
- 旧的 [beam margin 探针](../BEAM_MARGIN_256_PROBE.md) 已观察到条件路径误差、BF16 ties 和深度累积；本轮增加的是当前 FlatQuant 的阶段干预、显式 oracle 分解及校准状态核对，不把“B/C 重要”包装成首次发现。
- [FlatQuant](https://arxiv.org/abs/2410.09426) 已经联合学习权重/激活变换；本项目需要证明新增的任务结构如何改变校准，而非仅给现有矩阵重新命名。
- [PMPD](https://arxiv.org/abs/2410.13461) 已讨论阶段感知的精度分配，所以泛泛的 prefill/decode 分精度不是新的研究命题。
- [RAC](https://arxiv.org/abs/2509.12464) 已在 reasoning-model pruning 中指出 prompt-only reconstruction 的局限，补入 on-policy CoT 激活；这不是本仓库的 SID PTQ 结论，但直接限制了“把 decode 加进校准就是创新”的说法。
- [TLQ](https://arxiv.org/abs/2602.07899) 已在 VLM PTQ 中研究 token 重要性感知的校准。仅把视觉/文本换成 A/B/C、做几个 scalar weight，同样不足以证明强创新。
- [TAAL](https://arxiv.org/abs/2608.29179) 和 [BARGE](https://arxiv.org/abs/2607.21028) 已讨论 SID 前缀剪枝、语义漂移及训练/推理结构差异；这些不是 PTQ 的直接结论，但提醒我们不能把普通推荐解码优化等同于量化创新。

## 7. 历史复跑记录（源码已归档）

按要求清理了本轮的一次性入口、启动脚本和测试，结果与本报告保留。
三个源文件可从 `artifacts/results/flat_quant/fixed_sid_stage_probes/legacy_probe_source_20260915.tar.gz` 恢复。
下列命令仅记录历史协议，恢复归档前不能直接执行。后续训练验证不依赖这些源码。

独立入口：`flat_quant/probe_fixed_sid_stages.py`。

默认四卡筛选：

```bash
bash scripts/flat_quant/run_fixed_sid_stage_probes_cuda2345.sh
```

128 条固定前缀复核：

```bash
SAMPLES=128 SAMPLE_OFFSET=768 BEAM_SAMPLES=0 \
ARMS=baseline,history_a16,abc_a16,prefill_a16,bc_a16,all_a16,all_clip_only \
bash scripts/flat_quant/run_fixed_sid_stage_probes_cuda2345.sh
```

64 条逐层画像 + 最终 beam 诊断：

```bash
SAMPLES=64 SAMPLE_OFFSET=768 BEAM_SAMPLES=64 BEAM_ONLY=1 LAYER_PROFILE=1 \
BEAM_ARMS=baseline,abc_a16,bc_a16,all_a16,oracle_a,oracle_b,oracle_c,oracle_bc,oracle_b_offset,oracle_c_offset \
bash scripts/flat_quant/run_fixed_sid_stage_probes_cuda2345.sh
```

默认输出为新时间戳目录。已有 `protocol.json` 的目录会拒绝覆盖。日志记录进度；每 8 条 beam 样本报告一次实际 CUDA allocated 显存。
