# SID 生成阶段量化探针总结

## 1. 问题与目的

此前的 token-gap 探针已经观察到 text token、SID token 以及不同 SID slot
在表示和激活通道上存在差异，但这还不能回答两个更关键的问题：

1. SID 生成过程中的哪个预测阶段最容易受到量化影响？
2. 该阶段的误差主要来自权重量化还是激活量化？

因此，本轮探针不直接修改 GPTQ 目标，而是在保持同一份 Plain GPTQ FP8
权重不变的前提下，逐阶段、逐误差源恢复 BF16 路径，并观察推荐指标的配对变化。

实验协议：OneRec-1.7B、ad、calib128、test1000、beam=32、三 token SID。
除被恢复的局部路径外，其余路径均保持 real FP8 W8A8。

## 2. SID 生成阶段定义

这里的阶段由“当前 hidden state 预测哪一个下一个 SID token”定义，而不是
prompt 中出现的所有同类 SID token：

| 阶段 | 模型处理的位置 | 预测的下一个 token |
| --- | --- | --- |
| A | prefill 中最后一个 `<|sid_begin|>` | SID-a |
| B | decode 中刚生成的 SID-a | SID-b |
| C | decode 中刚生成的 SID-b | SID-c |

这一区分很重要：生成阶段的位置具有直接的因果角色，而历史 SID token 虽然
属于相同 slot，却只是上下文的一部分。

## 3. 阶段级激活恢复

每个 rescue 变体保留 GPTQ FP8-QDQ 权重，只把指定阶段各量化 Linear 的输入
恢复为 BF16，即局部 W8A16。Stage A 仅切分 prefill 的最后一个 token，而不是
将整个 prompt prefill 恢复为 BF16。

| 变体 | Pass@1 | Pass@16 | Pass@32 | Recall@32 | 配对恢复 | 配对回退 | 净恢复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W8A8 | 0.018 | 0.116 | 0.172 | 0.058266 | - | - | - |
| Rescue-A | 0.017 | 0.132 | 0.184 | 0.061013 | 33 | 21 | +12 |
| Rescue-B | 0.018 | 0.120 | 0.177 | 0.061379 | 23 | 18 | +5 |
| Rescue-C | 0.016 | 0.114 | 0.179 | 0.059652 | 10 | 3 | +7 |
| Rescue-All | 0.014 | 0.128 | 0.182 | 0.064141 | 28 | 18 | +10 |

配对恢复以每个样本的 Pass@32 为单位：W8A8 失败、变体成功记为恢复；
W8A8 成功、变体失败记为回退。

### 阶段定位结论

Stage A 是最强的单阶段瓶颈：仅恢复 final `<|sid_begin|>` 用于预测 SID-a
的激活，就使 Pass@32 提升 1.2 个百分点，且配对净恢复为 +12。B、C 也有
一定影响，但证据较弱。

Rescue-All 的 Recall@32 最高，但 Pass@32 低于 Rescue-A。这不是“保护更多
阶段一定更好”的证据，反而说明 beam search 中局部数值误差会改变候选轨迹和
排序；各阶段的指标收益不应被视为可加项。

返回 beam 的 prefix 统计也只用作轨迹代理：它来自最终返回的 beam，不能等价
于生成过程中所有存活 beam 的完整记录。

## 4. 实现校验：共享输入路径

初始实现只 patch 了 `RealFP8Linear.forward`，但 QKV 和 gate-up 使用
`apply.py` 中的共享输入路径，绕过了该函数。最初 smoke test 的 W8A16 调用数
为 0，因此该实现不能用于结论。

修正后，运行时改为通过 `tail_tokens_for_input` 和 `should_use_decode_a16`
进入统一的输入准备路径，覆盖普通 Linear 和共享 QKV/gate-up 路径。Stage-B
smoke test 观测到 196 次 W8A16 调用。相关阶段探针测试通过：`4 passed`。

这一步保证上表的 Stage rescue 确实改变了目标阶段的所有量化 Linear。

## 5. Stage-A 权重/激活归因

由于 Stage A 的激活恢复最稳定，进一步在 **real runtime** 中保留量化前的
BF16 权重快照，比较四种局部路径：

| 模式 | Stage-A 权重 | Stage-A 激活 | Pass@32 | Recall@32 | 配对恢复 | 配对回退 | 净恢复 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| W8A8 | GPTQ FP8-QDQ | FP8-QDQ | 0.172 | 0.058266 | - | - | - |
| A16 | GPTQ FP8-QDQ | BF16 | 0.184 | 0.061013 | 33 | 21 | +12 |
| W16 | 原始 BF16 | FP8-QDQ | 0.174 | 0.057510 | 27 | 25 | +2 |
| WA16 | 原始 BF16 | BF16 | 0.181 | 0.064276 | 36 | 27 | +9 |

其中 W16 只恢复权重，A16 只恢复激活，WA16 同时恢复两者。W16/WA16 为诊断
路径，临时使用 BF16 `F.linear`，其运行时间不代表部署时延。

### 归因结论

1. A16 的收益明确且复现了此前 Rescue-A：Pass@32 `0.172 -> 0.184`，净恢复
   `+12`。这说明 Stage-A 的 FP8 activation QDQ 是当前最可验证的主要误差源。
2. W16 的 Pass@32 仅升至 `0.174`，净恢复 `+2`，同时 Recall@32 下降。因此，
   当前证据不支持优先构造“仅针对 Stage-A 权重 Hessian”的 GPTQ 改进。
3. WA16 的 Recall@32 达到 `0.064276`，但 Pass@32 为 `0.181`，仍低于 A16。
   这表明 beam 轨迹下的权重/激活交互不呈简单单调关系；不能把
   `A16` 与 `W16` 的效果当作线性、可加的误差分解。

## 6. Fake 与 Real 路径的交叉校验

还运行过同协议的 fake-quant Stage-A 归因，但其 W8A8 baseline 已与 real
Plain GPTQ 明显不一致：fake 的 Pass@32 为 `0.184`，real 为 `0.172`。
更关键的是，fake A16 的配对净恢复为 `-4`，与 real A16 的 `+12` 相反。

因此，该 fake 结果不能作为权重或激活重要性的因果证据。它只说明当前 fake
与 real 的数值路径尚未足够对齐，不能替代 real-runtime 归因。后续若要使用
fake 进行算法筛选，需要先单独解决这一对齐问题。

## 7. 对方法设计的影响

本轮探针支持的主线不是“在所有 token 上重新加权 GPTQ Hessian”，而是：

```text
SID slot / 生成位置异质性
    -> Stage A（sid_begin -> SID-a）对量化最敏感
    -> 该敏感性主要表现为 activation QDQ 误差
    -> 应优先设计 Stage-A-aware、token/group/channel-aware 的 activation 方法
```

这也解释了为什么仅从 end-loss 梯度构造 token 权重、再直接加到 GPTQ Hessian
上缺乏稳定性：该路线直接改变权重量化目标，而当前实证定位到的主要误差源在
activation path。

## 8. 下一步：唯一合理的先导方向

下一步不应立即提出新的复杂量化算法，而应先验证 Stage-A activation error 是否
集中在与 SID slot 相关的少量 channel 上。最小实验应在真实运行时收集 Stage-A
各 Linear 的 activation QDQ 误差，并按 token group / channel 统计：

1. Stage-A final `<|sid_begin|>` 的逐 channel 相对 QDQ 误差；
2. 该误差与已有 token-wise outlier-channel / slot-specific channel 的重叠；
3. 若误差确实集中，再比较只对这些 channel 做轻量保护或平滑时，是否复现 A16
   的主要收益。

只有这三点成立，才有自然且与 SID token 异质性直接相连的 activation-side
量化方法动机。否则，本轮工作应止于“Stage-A 定位”这一可靠分析结论，而不应
强行把它包装成新的 GPTQ 权重算法。

## 9. 结果与代码位置

- 阶段恢复结果：
  `real_quant/naive_w8a8/results/probes/sid_generation_stage_ad_1p7b_gptq_calib128_test1000/`
- real Stage-A 权重/激活归因：
  `real_quant/naive_w8a8/results/probes/stage_a_weight_attribution_ad_1p7b_gptq_calib128_test1000/`
- real 归因汇总 JSON：
  `real_quant/naive_w8a8/results/probes/stage_a_weight_attribution_ad_1p7b_gptq_calib128_test1000/stage_a_real_attribution_summary.json`
- 实现：
  `real_quant/naive_w8a8/stage_weight_attribution_runtime.py` 与
  `real_quant/naive_w8a8/run_stage_a_weight_attribution.py`


## 补充实验更新：Stage-A activation channel 验证

本节替代前文“直接保护误差最大的 channel 可能有效”的下一步假设。所有新增结果都在 **real GPTQ W8A8 runtime** 上获得，使用 OneRec-1.7B、ad、calib128、test1000、beam=32。

### 1. 实际 activation-QDQ 误差是否集中

在每个 `RealFP8Linear` 中记录真正送入 fused dynamic FP8 quantizer 的输入，并只取 prefill 最后一个 `<|sid_begin|>` token。对每个输入 channel 计算：

```text
e_c = average_over_calibration_samples (x_c - xhat_c)^2
```

其中 `xhat` 是运行时 FP8 量化再反量化后的值。QKV 和 gate-up 分别共享输入，因此分别只统计一次；总共覆盖 28 层 x 4 类路径，即 112 个真实 activation 路径。

| 统计量 | 所有路径平均值 | 中位数 |
| --- | ---: | ---: |
| top 1% channel 承载的误差质量 | 36.0% | 34.7% |
| top 5% channel 承载的误差质量 | 60.3% | 61.6% |
| 平均相对误差平方 | 0.083% | 0.070% |

误差确实高度非均匀。QKV 输入、gate-up 输入、o_proj 输入、down_proj 输入的 top-1% 误差质量均值分别为 42.8%、26.8%、27.2% 和 47.2%。最极端的 layer 27 down_proj 中，top 1% channel 承载 89.5% 的该路径 QDQ 误差。

与此前 BF16 prompt 上的 token-wise outlier-channel profile 做描述性对齐时，boundary group 的 top-1% channel 平均承载 22.2% Stage-A 误差，高于 text 的 19.2% 与 sid-a/b/c 的 18.1% / 16.6% / 17.3%；112 个路径中 boundary 是误差质量最高 group 共 71 次。这与 Stage-A 正在处理 `<|sid_begin|>` 的语义角色一致，但该对齐跨越 BF16 prompt profile 与 W8A8 Stage-A path，不能视为因果证明。

### 2. 误差最大的 top-1% channel 是否对指标更重要

为进行因果验证，对每个 Linear 的 top 1% 误差 channel 恢复其 BF16 activation contribution。具体是在常规 FP8 输出后加入：

```text
(x_C - xhat_C) Wq_C^T
```

因此权重仍然是 GPTQ FP8-QDQ，未选中的 channel 仍是 FP8-QDQ。对照组在每个 Linear 随机选择数量相同的 1% channel。

| 变体 | Pass@32 | Recall@32 | 配对恢复 | 配对回退 | 净恢复 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W8A8 | 0.172 | 0.058266 | - | - | - |
| 误差 top-1% channel 恢复 | 0.175 | 0.061055 | 27 | 24 | +3 |
| 随机 top-1% channel 恢复 | 0.177 | 0.063256 | 32 | 27 | +5 |

结论是负面的：按 activation QDQ 误差选择 channel 虽有极小收益，但没有优于等预算的随机选择。单一随机 seed 不能证明随机选择本身更优，但已经足以否定 `error-top-k` 是一个有说服力的 channel 选择规则。

根本原因是 `e_c` 只度量局部数值扰动，不能度量该扰动经过后续层、SID-a logits 排序和 beam pruning 后对 Pass@32 的影响。beam search 中微小变化会改变候选轨迹，局部 MSE 最大的 channel 不必是推荐指标最重要的 channel。

### 3. 更新后的边界与下一步

当前可以可靠保留的结论是：

1. Stage A 是 SID 三步生成中最强的 activation quantization 敏感位置；
2. Stage-A 的 activation QDQ 误差在 channel 上高度集中，并在描述性上更接近 boundary token 的 channel profile；
3. 但“误差最大 channel = 推荐指标最重要 channel”不成立，不能直接做 top-error channel protection 或 smoothing。

因此当前不应继续调 error-top-k 的比例，也不应将其包装成论文方法。若继续探索 activation-side 方法，必须先构造一个任务相关准则：它需要在 Stage-A 的 SID-a 候选排序或 beam margin 上测量 channel 扰动影响，并在多个随机对照下稳定优于随机选择。没有这一步，本轮工作应停留在可靠的 Stage-A 定位与误差分析，而不是强行提出新的 GPTQ/channel 保护算法。

## 8. 保留记录

本文件是阶段探针唯一保留的记录。原始探针实现、测试和非全量输出已于
2026-07-24 清理；上文表格和结论应作为后续研究引用这些结果的唯一依据。
