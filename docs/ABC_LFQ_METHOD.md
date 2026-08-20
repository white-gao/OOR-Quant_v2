# ABC-LFQ：面向三段 SID 推荐的最后块 Logit-aware PTQ

## 1. 方法定位

ABC-LFQ 是本仓库在 LFQ（Logit-aware Final-block Quantization）思路上的推荐任务扩展。它不是 LFQ 原论文代码的原样复现，而是针对 OneRec 的三段式 SID：

```text
<s_a_*> → <s_b_*> → <s_c_*>
```

将最后一个 Transformer block 的优化目标从 hidden-state MSE 改为 SID-A、SID-B、SID-C 三个位置的全精度模型软分布蒸馏。

它要解决的问题是：逐层 MSE 让量化 hidden state 接近全精度 hidden state，但推荐指标最终由 LM head 输出的 SID 候选分布和三步自回归路径排序决定。两者并不完全等价，因此最后 block 应当直接优化更接近推荐决策的输出分布。

当前方法仍属于 PTQ：

- 全精度模型权重不更新；
- LM head 和 final norm 冻结；
- 前面 block 使用已经完成的 PTQ checkpoint；
- 只优化最后 block 的量化参数，例如 LWC 截断参数以及可选的 LET scale；
- teacher 提供软分布，不对模型权重做常规监督微调。

需要注意，ABC-LFQ 使用 ground-truth SID 构造 SID-B/SID-C 的条件前缀，因此属于 **GT-prefix-conditioned、teacher-soft-target 的任务感知 PTQ**，不是完全无标签 PTQ。

## 2. 受控的模型结构

以 28 个 Transformer block 的 1.7B 模型为例：

```text
Layer 0–26：加载同一个 MSE-LWC prefix checkpoint
Layer 27：重新从量化参数初始点出发，使用 ABC-LFQ 优化
Final norm：冻结
LM head：冻结
```

在严格对照实验中，MSE-LWC 与 ABC-LFQ 的 layer 0–26 完全相同，只有 layer 27 的优化目标和最终量化参数不同。因此最终推荐指标变化可以归因于最后 block 的目标调整，而不是前面层发生变化。

训练时同时维护两条 hidden-state 轨迹：

- teacher 轨迹：输入依次经过原始全精度 block；
- student 轨迹：输入依次经过恢复的量化 prefix block；
- 到 layer 27 时，teacher 接收全精度 prefix 的输出，student 接收量化 prefix 的输出。

因此，最后 block 的 ABC-LFQ 不仅看到本层量化误差，也会面对 layer 0–26 已经传播下来的量化轨迹偏移。

## 3. Calibration 样本构造

一条 calibration 样本的 target SID 记为：

\[
y^*=(a^*,b^*,c^*).
\]

模型 prompt 已以 `<|sid_begin|>` 结束。在其后追加 ground-truth 的 `a*` 和 `b*`：

```text
prompt + <|sid_begin|> + a* + b*
```

由于 causal LM 的位置错位关系，这个序列最后三个 hidden state 分别用于预测：

\[
\begin{aligned}
\text{SID-A}:&\quad P(a\mid x),\\
\text{SID-B}:&\quad P(b\mid x,a^*),\\
\text{SID-C}:&\quad P(c\mid x,a^*,b^*).
\end{aligned}
\]

这是一种 strict ground-truth teacher forcing：

- A 不依赖 SID 前缀；
- B 使用正确的 A 前缀；
- C 使用正确的 AB 前缀；
- 它隔离了每一步条件分布的量化误差，但没有模拟量化模型使用自身错误前缀继续解码的 exposure effect。

当前 `ad_calib.parquet` 的 1024 条样本每条都只有一个 target SID，因此当前实现每条样本只构造一条 GT 路径。它不能从现有 calibration 中获得多正例扩展。

## 4. Slot-specific 输出分布

对于每个 slot \(s\in\{a,b,c\}\)，只选择该 slot 对应的合法 SID 子词表：

\[
\mathcal V_a=\{ \texttt{<s\_a\_*>} \},\quad
\mathcal V_b=\{ \texttt{<s\_b\_*>} \},\quad
\mathcal V_c=\{ \texttt{<s\_c\_*>} \}.
\]

当前每个集合各有 8192 个 token。令冻结的 final norm 和 LM-head 相应行组成投影函数 \(G_s\)，则 teacher/student logits 为：

\[
z_s^T=G_s(h_s^T),\qquad z_s^Q=G_s(h_s^Q).
\]

温度默认为 1：

\[
p_s^T=\operatorname{softmax}(z_s^T),\qquad
p_s^Q=\operatorname{softmax}(z_s^Q).
\]

这里的 softmax 只在对应的 8192 个 slot token 内重新归一化，不包含普通词、特殊 token 或其他 SID 类型。

## 5. ABC-LFQ 优化目标

每个 slot 使用全精度模型软分布作为 target，计算 soft cross-entropy：

\[
\mathcal L_s
=-\sum_{v\in\mathcal V_s}
p_s^T(v)\log p_s^Q(v).
\]

三个 slot 的总目标为：

\[
\mathcal L_{\mathrm{ABC}}
=w_a\mathcal L_a+w_b\mathcal L_b+w_c\mathcal L_c,
\qquad
w_a+w_b+w_c=1.
\]

当前默认输入权重为 `(1,1,1)`，代码归一化后得到：

\[
w_a=w_b=w_c=\frac{1}{3}.
\]

由于 teacher 分布固定：

\[
\mathcal L_s
=H(p_s^T)+D_{\mathrm{KL}}(p_s^T\Vert p_s^Q).
\]

其中 teacher entropy \(H(p_s^T)\) 与量化参数无关。因此，在同一批样本和同一 slot 上，最小化当前 soft CE 与最小化 forward KL `KL(FP || Quant)` 完全等价。训练日志显示 CE，是为了直接计算稳定；它并不是与 KL 不同的另一个对齐目标。

当前目标不包含：

- ground-truth token 的 one-hot hard CE；
- hidden-state MSE；
- full-vocabulary CE；
- beam-path KL；
- top-k 排序或 margin loss。

训练日志中的 MSE 只是优化前后的诊断量，不参与反向传播。

## 6. 优化哪些参数

ABC-LFQ 使用与当前 OmniQuant runtime 相同的可训练量化模块。具体参数取决于实验配置：

- asymmetric LWC、无 LET：优化最后 block 七个 Linear 的逐输出通道上下截断比例；
- LWC + learned LET：除 LWC 外，还优化 QKV、VO、MLP 的通道 scale；
- final norm 与 LM head 始终冻结；
- 原始全精度权重始终冻结。

### 6.1 Learnable Weight Clipping

按照 [OmniQuant 原文 Eq. (2)](https://proceedings.iclr.cc/paper_files/paper/2024/file/c6483c8a68083af3383f91ee0dc6db95-Paper-Conference.pdf)，对第 \(i\) 个输出通道的权重 \(W_i\) 进行 \(N\)-bit asymmetric LWC：

\[
W_i^q=\operatorname{clamp}\!\left(
\left\lfloor\frac{W_i}{h_i}\right\rceil+z_i,\,
0,\,2^N-1
\right),
\quad
h_i=\frac{\gamma_i\max(W_i)-\beta_i\min(W_i)}{2^N-1},
\quad
z_i=-\left\lfloor\frac{\beta_i\min(W_i)}{h_i}\right\rceil.
\tag{1}
\]

其中 \(\lfloor\cdot\rceil\) 表示取整，\(h_i\) 为量化步长，\(z_i\) 为零点。对应的反量化权重为：

\[
\widehat W_i=h_i(W_i^q-z_i).
\tag{2}
\]

可学习的 clipping strength 为：

\[
\gamma_i=\sigma(\theta_i^+),\qquad
\beta_i=\sigma(\theta_i^-),
\qquad \gamma_i,\beta_i\in(0,1).
\tag{3}
\]

因此实际量化区间就是：

\[
\left[\beta_i\min(W_i),\ \gamma_i\max(W_i)\right].
\tag{4}
\]

\(\gamma_i\) 控制上界，\(\beta_i\) 控制下界；二者越小，对应一侧截断越强。当 \(\gamma_i=\beta_i=1\) 时，LWC 退化为普通 MinMax 量化。当前代码将两个 logit 初始化为 4，即初始 \(\gamma_i=\beta_i=\sigma(4)\approx0.982\)，并使用 STE 对取整操作反向传播。

当前实现是 per-output-channel LWC。一个 Linear 学习 \(2C_{\mathrm{out}}\) 个标量，一个含七个 Linear 的 block 共学习：

\[
N_{\mathrm{LWC}}=2\sum_{k=1}^{7}C_{\mathrm{out}}^{(k)}.
\tag{5}
\]

这里不是 \(7\times C_{\mathrm{in}}\times2\)：LWC 优化的是每个输出通道的上下 clipping strength。若使用 symmetric LWC，则每个输出通道只学习一个绝对值截断比例，零点固定为 0；本文主要 W4A8 ABC-LFQ 实验采用 asymmetric LWC。

### 6.2 当前主要实验设置

当前 W4A8 的主要受控实验采用：

| 项目 | 设置 |
|---|---|
| Weight | INT4，per-output-channel asymmetric LWC |
| Activation | INT8，dynamic per-token |
| LET | 关闭 |
| Layer 0–26 | 128-calib MSE-LWC checkpoint |
| Layer 27 | ABC-LFQ |
| Slot 权重 | A:B:C = 1:1:1 |
| Optimizer | AdamW |
| LWC learning rate | 0.01 |

## 7. 与联合序列 KL 的差距

理想的完整 SID 联合分布 KL 可以按链式法则写为：

\[
\begin{aligned}
D_{\mathrm{KL}}\!\left(P_T(a,b,c\mid x)\Vert P_Q(a,b,c\mid x)\right)
=&D_{\mathrm{KL}}\!\left(P_T(a\mid x)\Vert P_Q(a\mid x)\right)\\
&+\mathbb E_{a\sim P_T}
D_{\mathrm{KL}}\!\left(P_T(b\mid x,a)\Vert P_Q(b\mid x,a)\right)\\
&+\mathbb E_{a,b\sim P_T}
D_{\mathrm{KL}}\!\left(P_T(c\mid x,a,b)\Vert P_Q(c\mid x,a,b)\right).
\end{aligned}
\]

当前 ABC-LFQ 用唯一 GT 前缀代替了后两个期望：

\[
\mathbb E_a[\cdot]\rightarrow[\cdot]_{a=a^*},
\qquad
\mathbb E_{a,b}[\cdot]\rightarrow[\cdot]_{a=a^*,b=b^*}.
\]

所以它是联合 SID 分布 KL 的单路径、teacher-forcing 近似，不是完整 sequence KL，也不直接优化真实 beam-search 的路径集合和截断边界。

## 8. 已验证的 W4A8 AD-full 结果

layer 0–26 均复用同一个 128-calib LWC-only checkpoint：

| 方法 | Pass@1 | Pass@4 | Pass@8 | Pass@16 | Pass@32 | Recall@32 |
|---|---:|---:|---:|---:|---:|---:|
| LWC-only，128 calib | 1.1562% | 2.9411% | 4.4839% | 6.6662% | 9.1845% | 2.7430% |
| ABC-LFQ，128 calib | 1.3043% | 3.1795% | 4.7585% | 7.0058% | 9.5639% | 2.8938% |
| ABC-LFQ，512 calib | **1.3477%** | **3.3602%** | **4.9247%** | **7.4286%** | **10.5936%** | **3.3146%** |

512-calib ABC-LFQ 相对 LWC-only：

- Pass@32：`9.1845% → 10.5936%`，绝对提高 `1.4091 pp`；
- Recall@32：`2.7430% → 3.3146%`，绝对提高 `0.5716 pp`；
- PID Pass@32：`7.9019% → 9.3435%`；
- PID Recall@32：`2.3752% → 2.9391%`。

128→512 的逐样本配对结果在大 K 上也具有明确证据：SID Pass@32 新增命中 683 条、丢失 398 条，净增加 285 条，双侧配对 binomial `p=3.68e-18`。

## 9. 为什么它不是“换一种 MSE”

W4A8 512-calib ABC-LFQ 的最后 block：

| 量 | 优化前 | 优化后 |
|---|---:|---:|
| ABC soft CE | 4.5500 | 4.2617 |
| SID-A CE | 6.2767 | 6.2277 |
| SID-B CE | 4.0020 | 3.8608 |
| SID-C CE | 3.3714 | 2.6967 |
| Hidden-state MSE | 213.83 | 228.39 |

Hidden-state MSE 明显变差，但所有 AD-full 推荐指标提高。这说明 ABC-LFQ 找到的是对最终 SID 分布更有价值的量化参数方向，而不是简单地用另一种方式降低 block reconstruction error。

同时，不能跨不同 calibration 集直接比较 soft CE 的绝对大小，因为 teacher entropy 和样本构成都发生了变化。判断 128 与 512 哪个更好仍然需要独立测试集生成评测。

## 10. 当前已知局限

### 10.1 Equal loss weight 不等于 equal gradient influence

已有 W4A16 probe 表明，在纯 LWC 初始点，SID-C 的 mean-gradient norm 是 SID-A 的 2.82 倍，等权联合梯度与 C 的 cosine 为 0.884、与 A 只有 0.349。训练后 A/B/C 的 mean gradient 两两 cosine 全部变负，说明共享量化参数存在真实的多目标冲突。

因此 `1:1:1` 只表示 loss 标量等权，不表示三个 slot 获得相同优化预算。

### 10.2 B/C 前缀覆盖有限

SID-B/C 的梯度依赖不同的 A/AB 条件前缀。128 条 calibration 下，跨样本梯度一致性只有约 12.0%/10.4%，明显低于 SID-A 的 70.5%。512 条带来的显著提升说明，ABC-LFQ 比逐层 MSE 更依赖 calibration 前缀覆盖。

### 10.3 只对齐 slot 内部形状

当前 loss 在 8192-token slot 子词表内重新归一化，因此不直接惩罚概率泄漏到普通词或错误 SID 类型。既有诊断中正确 SID 类型的全词表概率质量保持在 99% 以上，暂未成为主要问题，但这是目标定义上的边界。

### 10.4 不是 beam-aware objective

ABC-LFQ 不对齐完整路径概率、beam 内候选并集、top-32 截断边界或候选间 margin。它只通过改善三个条件 token 分布间接影响 beam search。此前纯 beam-path KL 虽能显著降低 calibration loss，却造成推荐负优化，因此当前没有将 beam loss 纳入默认方案。

### 10.5 只调整最后 block

最后 block 只能重新映射前面已经形成的量化 hidden trajectory，不能消除所有早期层误差。它的优势是优化范围小、对照干净；表达能力也因此受到限制。

## 11. 当前判断与下一步

现有证据支持以下判断：

1. 最后 block 的 logit-aware 对齐方向成立，至少在 W4A8 上稳定优于同 prefix 的 MSE-LWC；
2. calibration 数量对 ABC-LFQ 的影响明显大于对传统逐层 MSE 的影响；
3. 当前主要瓶颈更可能是条件前缀覆盖与 slot 梯度冲突，而不是 CE/KL 的数学形式；
4. 在完成 `1024 calib × 5 epochs` 的全量评测前，不应同时引入新 loss，以免混淆 calibration diversity 与目标调整的贡献；
5. 若继续改目标，优先考虑基于梯度范数的 slot balance 或带验证集的 checkpoint selection，再考虑 teacher-prefix chain KL。

## 12. 实现与相关文档

核心实现：

- calibration batch：`fake_quant/run_m1_onerec_ad.py::build_lfq_sid_slot_batches`
- frozen projector：`fake_quant/omniquant/runtime.py::_LFQOutputProjector`
- soft CE：`fake_quant/omniquant/runtime.py::_lfq_soft_cross_entropy`
- prefix checkpoint 与最后 block 优化：`fake_quant/omniquant/runtime.py::apply_omniquant_layers`

相关诊断：

- [SID-A LFQ 512-sample diagnostics](SID_A_LFQ_512_DIAGNOSTICS.md)
- [SID-B/C conditional KL diagnostics](SID_BC_CONDITIONAL_KL_512_DIAGNOSTICS.md)
- [SID-A/B/C ABC-LFQ diagnostics](SID_ABC_LFQ_512_DIAGNOSTICS.md)
- [ABC gradient-conflict diagnostics](LFQ_ABC_GRADIENT_CONFLICT_128_DIAGNOSTICS.md)
- [Beam prefix survival probe](BEAM_PREFIX_SURVIVAL_PROBE.md)
- [Beam margin probe](BEAM_MARGIN_256_PROBE.md)

当前 `1024 calib × 5 epochs` 串行优化与四卡评测脚本：

```bash
bash scripts/fake_quant/run_1p7b_ad_full_w4a8_lfq_abc_calib1024_ep5_cuda4567.sh
```
