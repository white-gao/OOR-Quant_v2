# Token-role-aware LAC 先导探针

## 1. 目的

本探针回答一个实现前问题：是否有证据支持把 FlatQuant 风格的单组 Learnable
Activation Clipping（LAC）扩展成推荐任务专用的 token-role-aware LAC？

候选设计为每个 Linear 根据当前 token 的因果角色选择一组截断参数：

| 角色 | 当前输入 token | 该位置预测的下一个 token |
| --- | --- | --- |
| `TEXT` | 普通文本 token | 普通上下文 token |
| `PREDICT_A` | `<|sid_begin|>` | SID-a |
| `PREDICT_B` | SID-a | SID-b |
| `PREDICT_C` | SID-b | SID-c |
| `OTHER` | SID-c、边界、控制或其他特殊 token | SID-end 或其他 token |

这里按“当前位置预测什么”定义角色，而不是按 token 的名称直接命名。例如 SID-a
本身属于 `PREDICT_B`，因为它的 hidden state 用来预测 SID-b。该定义同时适用于
prompt 中的历史 SID 和自回归 decode；它只依赖已经输入模型的 token，不使用
ground-truth 未来 token，因此没有标签泄漏。

本探针需要区分两个问题：

1. 不同角色的激活分布和最佳截断比例是否真的不同？
2. 即使不同，角色分组相对普通单组 LAC 的额外收益是否足以支持新增实现复杂度？

## 2. 实验协议

| 项目 | 设置 |
| --- | --- |
| 模型 | OneRec-1.7B BF16 |
| 任务 | AD calibration |
| 样本 | 前 128 条；前 96 条拟合截断比例，后 32 条 held-out 验证 |
| Prompt 长度 | min 476，mean 724.60，max 1095 |
| Transformer 层 | 0、9、18、27 |
| Linear | q/k/v/o、gate/up/down，共 `4 × 7 = 28` 个层-Linear 单元 |
| 唯一激活位置 | QKV input、O input、gate/up input、down input |
| 激活量化 | INT8、symmetric、dynamic per-token |
| 局部输出探针的权重量化 | INT4 asymmetric per-output-channel RTN |
| 截断比例网格 | 1.0 至 0.5，共 15 个候选值 |
| 激活采样 | 每条 prompt、每个角色、每个激活位置最多均匀取 4 个 token row |
| 输出采样 | 每个 Linear 固定随机采样 192 个 output channel |
| 随机种子 | 42 |

128 条 prompt 的角色覆盖如下。五类角色均有足够样本，不存在因为某个角色太少而
无法估计截断比例的问题。

| Split | TEXT | PREDICT_A | PREDICT_B | PREDICT_C | OTHER | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit 96 | 6,546 | 12,339 | 12,243 | 12,243 | 24,966 | 68,337 |
| Held-out 32 | 2,165 | 4,443 | 4,411 | 4,411 | 8,982 | 24,412 |

### 2.1 截断和量化误差

对一个 token row `x`，候选比例为 `r`，本探针使用当前仓库一致的 symmetric
dynamic per-token INT8 QDQ：

$$
t_r = r \max_i |x_i|, \qquad
s_r = \frac{t_r}{127}, \qquad
\hat{x}_r = s_r\,\mathrm{clip}\left(\mathrm{round}(x/s_r), -127, 127\right).
$$

由于当前激活量化是 symmetric，正负方向最终共享一个有效绝对阈值。因此本轮
只搜索一个比例；它不能验证 asymmetric activation LAC 中独立上下界是否有益。

除输入 activation MSE 外，还计算两种局部 Linear 输出误差：

$$
y_{\mathrm{fp}} = xW^T,
$$

$$
\hat y_{\mathrm{A8}}(r) = \hat x_r W^T, \qquad
\hat y_{\mathrm{W4A8}}(r) = \hat x_r Q_4(W)^T.
$$

W4A8 局部目标为

$$
E_r = \mathrm{MSE}\left(\hat y_{\mathrm{W4A8}}(r), y_{\mathrm{fp}}\right).
$$

它允许 activation clipping 在局部上补偿一部分 W4 weight error，比只看输入
activation MSE 更接近 W4A8 使用场景。每个单元先在 96 条 fit prompt 上选比例，
然后固定比例，在 32 条 held-out prompt 上报告误差。

## 3. 不同角色的激活分布确实不同

下表先对 QKV、O、gate/up、down 四个唯一激活位置去重，再对 4 层共 16 个位置
取 held-out 均值。

| 角色 | token absmax | token RMS | peak/RMS | top-1% channel 能量占比 |
| --- | ---: | ---: | ---: | ---: |
| TEXT | 58.94 | 2.853 | 17.30 | 42.97% |
| PREDICT_A | 68.16 | 2.933 | 15.54 | 43.20% |
| PREDICT_B | 33.41 | 2.228 | 14.53 | 40.08% |
| PREDICT_C | 34.38 | 2.428 | 13.50 | 36.65% |
| OTHER | 74.05 | 2.496 | 20.38 | 52.86% |

主要观察：

1. `PREDICT_A` 的平均 absmax 大约是 `PREDICT_B/C` 的两倍，A/B/C 不是同一种
   激活尺度。
2. `OTHER` 的 peak/RMS 和 top-1% channel 能量占比最高，控制/边界/SID-c 等
   token 的分布最尖锐。
3. `PREDICT_C` 的 top-1% channel 能量占比最低，分布相对更均匀。

因此，“所有 token 共享完全相同的 clipping prior”并不是很强的假设。角色感知
设计具有数据分布上的依据。

## 4. 各角色的 W4A8 最佳截断比例

下表对 28 个层-Linear 单元统计 fit 集最优比例，并用该比例在 held-out 集上测量
W4A8 局部输出误差。`误差下降` 都相对不截断的 `r=1`。

| 角色 | 最佳比例中位数 | IQR | 最佳比例 `<0.98` | Fit 误差下降 | Held-out 误差下降 | Held-out 改善单元 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TEXT | 0.940 | [0.869, 0.965] | 75.0% | 2.31% | 2.20% | 25/28 |
| PREDICT_A | 0.930 | [0.838, 0.940] | 89.3% | 2.94% | 2.82% | 24/28 |
| PREDICT_B | 0.940 | [0.894, 0.965] | 75.0% | 1.11% | 1.04% | 24/28 |
| PREDICT_C | 0.940 | [0.894, 0.960] | 82.1% | 1.18% | 1.11% | 26/28 |
| OTHER | 0.940 | [0.850, 0.980] | 71.4% | 2.81% | 2.66% | 25/28 |

这一结果有两面性：

- 正面：截断在 held-out 上仍普遍降低局部误差，说明 LAC 本身有稳定信号；
  `PREDICT_A` 的最佳比例分布也更偏向强截断。
- 限制：五类角色的全局中位数都在 0.93–0.94，不能声称它们需要完全不同的
  全局阈值。差异主要发生在具体层和具体 Linear，而不是简单的角色常数。

Fit 选择的 W4A8 比例在 held-out 上没有明显失效。五类角色中，最优比例精确一致
的单元分别为 19/28、17/28、9/28、8/28、19/28；比例差不超过 0.02 的单元分别
为 24/28、22/28、20/28、21/28、23/28。相对 held-out oracle 的平均误差 regret
均小于 0.17%。B/C 的比例更不稳定，但整体泛化误差仍小。

## 5. 普通 LAC 与角色感知 LAC 的 held-out 对比

比较三个方案：

1. `No clipping`：固定 `r=1`；
2. `Shared`：每个 Linear 一组比例，所有角色共享；
3. `Role-aware`：每个 Linear 为五个角色分别选择比例。

所有比例只使用 fit 96 选择。下表的前两列表示相对 `No clipping` 的 held-out
局部输出 MSE 下降；“角色额外收益”表示 Role-aware 相对 Shared 的误差下降，
不是两个百分数的简单百分点差。

| 输出设置与加权 | Shared 相对 r=1 | Role-aware 相对 r=1 | Role-aware 相对 Shared | Role-aware 胜出单元 |
| --- | ---: | ---: | ---: | ---: |
| A8-only，全 token 按频次加权 | 3.59% | 4.32% | 0.80% | 18/28 |
| A8-only，A/B/C 等权 | 4.96% | 5.31% | 0.43% | 11/28 |
| W4A8，全 token 按频次加权 | 1.90% | 2.22% | 0.33% | 24/28 |
| W4A8，A/B/C 等权 | 1.39% | 1.58% | 0.20% | 23/28 |

W4A8 下，14/28 个单元的 A/B/C 最佳比例跨度至少为 0.05；若考虑全部五类角色，
则为 19/28。相比之下，A8-only 下这两个数字只有 5/28 和 8/28。这说明 W4
weight error 与 activation clipping 的交互放大了角色差异，但也意味着该信号
不完全是纯激活分布造成的。

总体方向是稳定正向的，但量级较小：普通单组 LAC 已经取得大部分 clipping
收益，五组角色参数带来的 held-out 局部误差额外下降只有约 0.2%–0.3%。

## 6. 信号主要集中在最后一层

下表使用 W4A8、A/B/C 等权目标，并按层对 7 个 Linear 求平均。

| Layer | Shared 相对 r=1 | Role-aware 相对 r=1 | Role-aware 相对 Shared | 胜出 Linear | A/B/C 比例跨度 `>=0.05` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.83% | 2.06% | 0.23% | 7/7 | 3/7 |
| 9 | 1.21% | 1.28% | 0.07% | 5/7 | 1/7 |
| 18 | 1.49% | 1.56% | 0.07% | 4/7 | 3/7 |
| 27 | 1.02% | 1.43% | **0.42%** | **7/7** | **7/7** |

第 27 层是唯一一个所有 Linear 都出现明显角色比例分离、且 7/7 都在 held-out
上获得正增益的层。这与当前 ABC-LFQ 只调整最后一个 block 的路线具有直接联系。

第 27 层各 Linear 的 fit 比例和 held-out 结果如下：

| Linear | Shared | A | B | C | Shared 相对 r=1 | Role-aware 相对 r=1 | 角色额外收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 0.940 | 0.875 | 0.940 | 0.960 | 0.34% | 0.44% | 0.09% |
| k_proj | 0.960 | 0.875 | 0.940 | 0.960 | 0.35% | 0.37% | 0.02% |
| v_proj | 0.700 | 0.800 | 0.700 | 0.700 | 1.99% | 2.05% | 0.06% |
| o_proj | 0.980 | 0.900 | 0.960 | 0.980 | 0.07% | 0.22% | 0.15% |
| gate_proj | 0.850 | 0.700 | 0.800 | 0.850 | 2.74% | 3.66% | 0.95% |
| up_proj | 0.900 | 0.700 | 0.900 | 0.900 | 0.69% | 2.21% | **1.53%** |
| down_proj | 0.940 | 0.940 | 0.875 | 0.850 | 0.94% | 1.09% | 0.15% |

最后一层最明确的模式出现在 gate/up：`PREDICT_A` 选择 0.70，而 B/C 选择
0.80–0.90。也就是说，至少在局部 W4A8 输出 MSE 下，SID-a 预测位置更能从强
clipping 中获益。

## 7. 结论与实现建议

### 7.1 可以确认的结论

1. **角色异质性真实存在。** A/B/C 的激活范围、异常能量和部分 Linear 的最优
   clipping ratio 不同，因此角色感知 LAC 不是没有数据依据的任意分组。
2. **截断信号能够迁移到 held-out。** 普通 LAC 和角色感知 LAC 的局部误差收益
   都没有在 32 条 held-out prompt 上消失。
3. **角色感知的增量总体较小。** 在 28 个单元上，W4A8 held-out 局部误差相对
   普通 LAC 只额外下降约 0.20%–0.33%。当前证据不支持直接给全部 28 层增加
   五组参数和 role-mask runtime。
4. **最后一层是最值得做最小实验的位置。** Layer 27 的 7/7 Linear 均为正增益，
   A/B/C 比例也全部出现至少 0.05 的跨度；这与 ABC-LFQ 的优化范围完全一致。

### 7.2 推荐的下一步

若继续实现，建议只做一个可回退的 final-block 对照，不先扩展到全模型：

| 组别 | Layer 0–26 | Layer 27 |
| --- | --- | --- |
| A | 同一 LWC prefix | LWC + ABC-LFQ |
| B | 同一 LWC prefix | LWC + 普通 per-Linear LAC + ABC-LFQ |
| C | 同一 LWC prefix | LWC + role-aware per-Linear LAC + ABC-LFQ |

比较 train/validation 的 ABC CE、最后一层 MSE、各角色 clipping ratio、clip rate
和最终推荐指标。只有 C 在独立验证和完整推荐评测上稳定优于 B，才能说明收益来自
角色条件化，而不是普通 LAC 扩大优化空间。

从 probe 看，优先关注最后一层 gate/up 是合理的，但正式第一版仍建议保留 7 个
Linear，避免根据同一份 probe 数据再次人工筛选而引入选择偏差。

## 8. 适用边界

本结果是实现筛选证据，不是端到端算法验证：

- 捕获的是 BF16 layer input，没有模拟 W4A8 误差逐层传播；
- W4A8 输出只计算单个 Linear 的局部误差，没有覆盖 attention、SiLU、乘法和
  residual 等完整 block 非线性；
- 每个 Linear 只采样 192 个 output channel；
- held-out 仍来自同一 calibration 文件，而不是真正的 test set；
- 局部 MSE 改善不能直接等价为 Pass/Recall 改善，仓库中的既有实验已经证明两者
  并非单调对应；
- 当前 activation QDQ 是 symmetric，因此只验证一个有效 absmax ratio，未验证
  独立上下界参数。

因此最稳妥的判断是：**role-aware LAC 有可复现但偏小的局部正信号，足以支持
“只在最后一个 block 做一次受控实现”，不足以支持直接把它当成全模型新方法。**

## 9. 清理说明

本探针的临时 Python 实现、smoke/full 原始 JSON 和 smoke test 自动生成的 2 条
样本缓存均已在文档完成后删除。仓库只保留本总结。
