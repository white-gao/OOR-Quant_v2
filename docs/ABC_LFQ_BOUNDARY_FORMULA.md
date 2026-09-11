# ABC-LFQ Boundary Loss 核心公式

本文只记录当前代码中的核心定义；公式使用标准 LaTeX，可由 Markdown 数学渲染器编译。

## 1. 符号

| 符号 | 含义 |
| --- | --- |
| $q$ | SID slot，$q \in \{A,B,C\}$ |
| $t_i^q$ | FP teacher 对 token $i$ 的 logit |
| $s_i^q$ | 量化 student 对 token $i$ 的 logit |
| $K$ | teacher top-$K$ 候选数，默认 32 |
| $N$ | teacher 边界负候选数，默认 32 |
| $\tau$ | tie 过滤阈值，默认 0.01 |
| $\gamma$ | teacher gap 权重尺度，默认 1.0 |
| $B$ | batch size |

以下公式先描述单条样本，并省略样本下标。

## 2. Teacher 候选集合

每个 slot 的正、负候选集合完全由 FP teacher 决定：

$$
\begin{aligned}
\mathcal{P}_q &= \operatorname{TopK}_K(t^q), \\
\mathcal{Q}_q &= \operatorname{Ranks}_{K+1:K+N}(t^q).
\end{aligned}
$$

$\mathcal{Q}_q$ 不是随机负样本，而是紧邻 top-$K$ 边界的候选。

## 3. Teacher gap 与 student gap

对任意 $i \in \mathcal{P}_q$、$j \in \mathcal{Q}_q$：

$$
\begin{aligned}
g_{ij}^{T,q} &= t_i^q-t_j^q, \\
g_{ij}^{S,q} &= s_i^q-s_j^q, \\
e_{ij}^q &= g_{ij}^{S,q}-g_{ij}^{T,q}.
\end{aligned}
$$

## 4. Tie-aware pair 权重

$$
w_{ij}^q =
\begin{cases}
0, & g_{ij}^{T,q} \le \tau, \\
\min\left(g_{ij}^{T,q}/\gamma,\,1\right), & g_{ij}^{T,q} > \tau.
\end{cases}
$$

teacher 无法可靠区分的近似 tie pair 不参与训练；其余 pair 随 teacher gap 增大而加权，达到 $\gamma$ 后权重饱和为 1。

## 5. 单个 pair 的损失

$$
\ell_{ij}^q = \operatorname{SmoothL1}\!\left(e_{ij}^q\right).
$$

当前使用 PyTorch 默认参数，对应：

$$
\operatorname{SmoothL1}(e) =
\begin{cases}
\frac{1}{2}e^2, & |e| < 1, \\
|e|-\frac{1}{2}, & |e| \ge 1.
\end{cases}
$$

因此该损失不仅要求 student 保持正确边界顺序，还会让 student 恢复 teacher 的实际 gap。

## 6. 单个 slot 的 boundary loss

对样本 $b$ 的所有 $K \times N$ 个 pair 加权归一化：

$$
L_{\mathrm{bdry},b}^q =
\frac{
\sum_{i \in \mathcal{P}_q}\sum_{j \in \mathcal{Q}_q}
w_{ij}^q\ell_{ij}^q
}{
\max\left(
\sum_{i \in \mathcal{P}_q}\sum_{j \in \mathcal{Q}_q}
w_{ij}^q,\,1
\right)
}.
$$

再对 batch 内样本取平均：

$$
L_{\mathrm{bdry}}^q =
\frac{1}{B}\sum_{b=1}^{B}L_{\mathrm{bdry},b}^q.
$$

## 7. A/B/C 合并

$$
L_{\mathrm{bdry}} =
\sum_{q \in \{A,B,C\}}\alpha_q L_{\mathrm{bdry}}^q,
\qquad
\alpha_A=\alpha_B=\alpha_C=\frac{1}{3}.
$$

## 8. 最终训练目标

$$
L_{\mathrm{total}} =
\lambda_{\mathrm{ABC}}L_{\mathrm{ABC}}
+\lambda_{\mathrm{bdry}}L_{\mathrm{bdry}}.
$$

当前受控实验：

| 分支 | $\lambda_{\mathrm{ABC}}$ | $\lambda_{\mathrm{bdry}}$ |
| --- | ---: | ---: |
| 原 ABC-LFQ | 1.0 | 0.0 |
| ABC-LFQ + boundary | 1.0 | 0.1 |
sss
当 $\lambda_{\mathrm{bdry}}=0$ 时，代码不会构造 top-$(K+N)$ boundary target，行为回退到原 ABC-LFQ。

## 9. Held-out 诊断

训练使用连续的 gap matching 损失；以下指标只用于 held-out 诊断，不参与反向传播：

- teacher top-32 保留率；
- rank 33--64 候选侵入率；
- rank 64 以后候选侵入率；
- $g_{ij}^{S,q}\le 0$ 的 pair violation rate；
- student gap 与 teacher gap 的 MAE。

当前协议：Layer 0--26 使用前 128 条 calibration；三个 Layer 27 分支使用前 512 条训练；最后 512 条仅用于 held-out 检查。
