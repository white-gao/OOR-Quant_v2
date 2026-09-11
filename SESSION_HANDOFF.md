# OOR-Quant 会话迁移说明

更新日期：2026-08-24。本文件记录当前研究主线、有效结果、代码状态与下一步
实验，供新会话直接接手。数值以 `artifacts/results/` 中的最终结果为准。

## 研究目标

项目研究 LLM-based generative recommendation 的 post-training quantization。
OpenOneRec 自回归生成三个 Semantic ID token（`SID-a → SID-b → SID-c`）来表示
推荐 item。当前核心问题是：

1. 为什么相同量化在通用语言 benchmark 上接近无损，却会让推荐指标明显下降；
2. FP8 W8A8 是否适合作为低成本部署方案；
3. 当权重降到 INT4 后，LWC/LET 等 OmniQuant 组件能否恢复推荐能力。

项目现在明确区分两条路径：

- `real_quant/`：真实 FP8 kernel，只用于 FP8 W8A8 精度、时延和部署结论；
- `fake_quant/`：FP8/INT8/INT4 的 QDQ 数值实验，是当前低比特方法研究主线，
  不能用于宣称 packed memory 或真实 INT4/INT8 加速。

## 当前仓库结构

```text
fake_quant/                 混合精度 fake-QDQ、SmoothQuant、GPTQ、OmniQuant
real_quant/full_precision/  BF16 推荐与通用 benchmark runner
real_quant/naive_w8a8/      真实 FP8 W8A8、plain GPTQ/GPTAQ
benchmarks/benchmark/       OpenOneRec/RecIF-Bench evaluator
scripts/fake_quant/         fake-quant 串行实验入口
scripts/real_quant/         real-quant 串行实验入口
shared/                     模型、数据和结果路径的统一定义
docs/              研究表格与保留的 probe 总结
artifacts/                  生成结果与探测产物（Git 忽略）
```

实验 launcher 只保留在 `scripts/` 下。重复脚本、decode-A16 fake-quant
ablation 和已完成的一次性 probe 可以继续清理。当前 `tests/` 下有 unittest
suite；修改 fake-quant 核心后至少执行：

```bash
/home/yhhuang/miniconda3/envs/benchmark2/bin/python -m unittest discover -s tests
python -m pytest -q
```

缓存目录可以随时删除。

清理补充（2026-08-18）：主代码只保留 OmniQuant 的逐层 MSE 与最后一层
ABC-LFQ；实验效果不佳的 LFQ-ALL、Beam-KL 训练目标、低秩补偿分支和一次性
绘图/实验 launcher 已删除。普通推荐 beam-search 解码、多卡分片测评及历史
结果文档不受影响。

权重与 activation 探测代码已经删除，保留的完整结果归档为：

```text
artifacts/archives/onerec_weight_activation_probes.tar.gz
```

## 模型、数据与推荐评测协议

默认模型：

```text
/root/dataDisk/guowei/models/1.7B
/root/dataDisk/guowei/models/8B
```

推荐任务今后统一使用：

```text
/root/dataDisk/guowei/data/onerec_data/benchmark_data
```

不要再使用旧的 `benchmark-data-calib1024`。该目录下 calibration 与 test 是
独立划分；快速方法实验通常使用 `calib=128`。当前低比特主对比使用 AD test
前 3000 条，所有方法保持相同数据顺序和 `seed=42`。

HF 推荐 runner 与 OpenOneRec 官方 vLLM 解码协议已经核对到可比较程度。当前
固定为 greedy beam search（`do_sample=False`）、`num_beams=32`、
`num_return_sequences=32`、`max_new_tokens=3`。不要给不同量化方法改变 beam
或 sample 设置。

## Real FP8 路径

核心入口：

```bash
python -m real_quant.naive_w8a8.run_hf_naive_w8a8 --help
```

`RealFP8Linear` 使用 FP8 E4M3FN per-output-channel 权重 scale、运行时
per-token dynamic activation scale 和 `torch._scaled_mm`。Q/K/V 与 gate/up
共享一次输入 activation quantization。

保留的权重量化模式：

- `minmax`：naive/RTN FP8 W8A8；
- `gptq`：plain GPTQ，Hessian 为校准输入的 `XᵀX`；
- `gptaq`：plain GPTAQ，可选择 activation-aware target。

`--decode_a16_single_token` 是部署选项，不属于纯 W8A8 算法比较。实验表中写
pure W8A8 时必须关闭它。真实 FP8 路径依赖可用的 CUDA 12 runtime、
`torch._scaled_mm` 和 vLLM fused `scaled_fp8_quant` custom op；PyTorch、
torchvision 和 vLLM 的 CUDA build 必须一致。

真实 FP8 推荐 runner 支持 deterministic round-robin 多卡测评：每张卡独立加载
模型并使用 `--eval_num_shards/--eval_shard_id` 处理一个 shard；单 shard 禁止
直接 `--evaluate`，全部完成后由 `python -m real_quant.merge_eval_shards`
恢复原样本顺序、校验配置一致性、聚合 latency 并只计算一次 benchmark metric。
正式入口为 `scripts/real_quant/run_1p7b_ad_full_rtn_fp8w8a8_sharded.sh`，支持
`GPUS`、`OUTPUT_DIR`、`OVERWRITE` 和 `DRY_RUN` 覆盖，并在 INT/TERM 时回收子进程。

## Fake-QDQ 与 OmniQuant

主入口：

```bash
python -m fake_quant.run_m1_onerec_ad --help
```

`baseline_qdq` 支持权重和 activation 独立选择：

```text
none / fp8_e4m3fn / fp4_e2m1 / int8 / int6 / int4
```

入口默认仍是 `baseline_qdq + asymmetric INT8-W + dynamic per-token
symmetric INT8-A`，但当前新方法实验主线优先使用浮点 QDQ：W4A8 表示
FP4-E2M1-W/FP8-E4M3-A，W8A8 表示 FP8-E4M3-W/A。INT4/INT8 结果保留为
历史和数据类型消融，不再为新版 ABC-LFQ 重跑完整 INT sweep。整数权重默认
采用 affine asymmetric zero-point QDQ；FP4/FP8 权重只能使用 zero-centered
symmetric QDQ。activation 仍为 dynamic per-token，fake-QDQ 最终以模型 dtype
执行 `F.linear`。当前没有 FP6 codebook，因此尚不宣称 FP W6A6 支持。

统一数值契约（2026-08-18 起）为 deployment-matched，适用于 RTN、
SmoothQuant、GPTQ、OmniQuant 和 LFQ：FP32 只用于 master weight、量化参数、
scale/zero-point/QDQ 算术及 loss；QDQ 后的权重与 activation 必须先转回模型
dtype，再执行 Linear、RMSNorm、attention、MLP、残差和 LM head。该契约没有
兼容开关。旧 OmniQuant checkpoint 属于 FP32-surrogate 校准路径，元数据校验
会拒绝加载，必须重新校准。

OmniQuant 实现在 fake_quant/omniquant/runtime.py，采用逐 block 的输出重构
MSE。当前包含：

- LWC：每个 Linear、每个输出通道学习 clipping 参数；INT4/INT6/INT8 支持 symmetric
  和 asymmetric，FP4 E2M1/FP8 E4M3FN 支持 zero-centered symmetric clipping；
- LET-QKV：input norm 到共享 Q/K/V scale；
- LET-MLP：post-attention norm 到共享 gate/up scale；
- LET-V/O：适配 Qwen3 GQA head 排列的 V→O scale；
- `--omni_let_mode none|fixed|learned`：分别表示关闭、固定 SmoothQuant
  初始化、学习 LET。

默认校准协议为128条样本；OmniQuant 默认优化20 epochs，LWC/LET 学习率
分别为1e-2/5e-3，weight decay 为0且不做梯度裁剪。固定 SmoothQuant
与 OmniQuant 的 SQ-init 共用 alpha=0.4，该值由 AD calibration block MSE
搜索得到。

### ABC-LFQ top-K boundary 排序增强（2026-08-24）

当前 ABC-LFQ 保留 SID-A/B/C 三个 slot 内的 teacher-soft CE，并新增了
一个可选的 teacher-only top-K boundary gap loss。它用 FP teacher 的 ranks
1–K 作为正候选，紧随其后的 N 个 token 作为边界负候选，通过 weighted
SmoothL1 对齐跨边界 logit gap。teacher gap 不大于 tie threshold 的 pair
不训练，避免强行排序高熵平坦区域中的近似并列 token。

总目标为：

```text
L_total = lfq_loss_weight * L_ABC_CE
        + lfq_boundary_loss_weight * L_boundary
```

新增 CLI：

```text
--omni_lfq_boundary_loss_weight       默认 0
--omni_lfq_boundary_topk              默认 32
--omni_lfq_boundary_negative_count    默认 32
--omni_lfq_boundary_tie_threshold     默认 0.01
--omni_lfq_boundary_gap_scale         默认 1.0
```

boundary weight 为 0 时不构建 top-64 target，行为回退到原 ABC-LFQ。
checkpoint、epoch log 和 `omniquant_config.json` 会保存总 boundary loss 以及
A/B/C 分 slot 诊断。该实现只保护 teacher top-32 与 ranks 33–64 的边界，
暂不做 student intruder mining，也不强制 top-32 内部的全排序。详细定义见
`docs/ABC_LFQ_METHOD.md`。

实现文件：

```text
fake_quant/omniquant/runtime.py
fake_quant/run_m1_onerec_ad.py
fake_quant/evaluate_lfq_boundary_diagnostics.py
scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_lwc_abc_boundary_calib1024.sh
tests/test_fake_quant_deployment_matched.py
tests/test_lfq_boundary_diagnostics.py
```

已验证：核心文件 `py_compile` 通过，`python -m unittest discover -s tests`
30/30 通过，pytest 33/33 通过；小型端到端训练测试能产生非空 train/validation boundary 诊断。
尚未进行真实模型 calibration 或推荐测评。

### Asymmetric LWC 的当前实现

`--weight_quant_scheme` 支持：

- `symmetric`：zero-centered、单 clipping 参数实现；INT4/INT6/INT8 使用 signed
  uniform QDQ，FP8 使用 E4M3FN cast QDQ；
- `asymmetric`：论文式双边 LWC，每个输出通道分别学习 upper/lower clipping，
  使用 zero point 和 `[0, 2^N-1]` 整数码域，仅适用于 INT4/INT6/INT8。

默认规则为 INT4/INT6/INT8 权重使用 `asymmetric`，FP8 权重使用
`symmetric`。该规则同时用于普通 RTN 和 OmniQuant；旧参数名
`--omni_weight_quant_scheme` 保留为兼容别名。

对一个输出通道，asymmetric 路径为：

```text
upper = sigmoid(up_logit)  * max(W)
lower = sigmoid(low_logit) * min(W)
scale = (upper - lower) / (2^N - 1)
zero_point = -round(lower / scale)
Q = clamp(round(W / scale) + zero_point, 0, 2^N - 1)
W_qdq = (Q - zero_point) * scale
```

训练 QDQ 与最终静态固化共用同一公式。自动回退代码已物理删除：即使某层
`final_loss > initial_loss`，也会保留真实训练后参数并传播到下一层；NaN/Inf
仍然直接报错。checkpoint 保存上下界两组 logits。

需要复现历史 symmetric INT 结果时，必须显式写
`--weight_quant_scheme symmetric`。

### Group-wise 权重量化（RTN / SmoothQuant / OmniQuant-LWC）

RTN、SmoothQuant 与 OmniQuant-LWC 支持 `--weight_group_size G`。这里的
group 位于每个 output-channel 行内部，沿 Linear 的 `in_features` 连续分组；每个
`(output_channel, input_group)` 独立计算 scale/zero point。`G=0` 或 `None`
严格保留历史 per-output-channel 路径，尾部不足一个 group 时 asymmetric 路径
用 mask 排除 padding。

OmniQuant-LWC 在每组上独立学习 clipping：symmetric INT/FP 每组一个 logit，
asymmetric INT 每组一对 upper/lower logits；每组由学习后的范围独立产生 scale
（以及 asymmetric zero point）。checkpoint 会记录 `weight_group_size`，旧
per-channel checkpoint 缺失该字段时按 `0` 兼容恢复。

支持链路包括 `fake_quant.quant`、普通 RTN wrapper、SmoothQuant fold/runtime、
OmniQuant train/finalize/checkpoint、CLI metadata 和测试。正式 W4A8
per-channel/g128 串行全量 launcher 为：

```text
scripts/fake_quant/run_1p7b_ad_full_fp4w_fp8a_rtn_pc_g128_cuda0123.sh
```

### 保留的诊断与实验工具

- `docs/QUANT_FORMAT_LAYERWISE_MSE_AD128.md` 已同时记录局部 block MSE 与两流
  prefix-quantized 累计 MSE；累计流分别传播 FP 和量化 hidden state。
- `fake_quant/probe_activation_quant_patterns.py`：压缩完整 SID-ABC 组后，绘制
  Layer-27 q_proj/o_proj 的 BF16 输入、局部 activation QDQ 与误差矩阵。
- `fake_quant/probe_w8a8_projection_output_patterns.py`：在相同 prompt 上比较
  BF16 与端到端 W8A8 的 q_proj/o_proj 输出和累计误差。
- 两个 probe 的 launcher 位于 `scripts/fake_quant/run_layer27_*_probe_ad.sh`。
- Product 域 FP8 W8A8 SmoothQuant alpha 搜索入口为
  `scripts/fake_quant/search_product_smoothquant_alpha_mse_fp8w8a8.sh`。
- 已完成的单文件 FP4/INT4 MSE 比较器及 launcher 已删除；结论保留在文档和
  `artifacts/results/` 中。

## 已完成的关键结果

### 通用能力：真实 RTN FP8 W8A8 基本无损

| 模型 | Benchmark | BF16 | RTN W8A8 |
|---|---:|---:|---:|
| 1.7B | WikiText-2 PPL | 11.8665 | 11.9563 |
| 1.7B | C4-val 256×2048 PPL | 17.8754 | 18.0039 |
| 1.7B | GSM8K non-thinking | 67.17% | 69.29% |
| 1.7B | MATH-500 non-thinking | 63.60% | 63.80% |
| 8B | WikiText-2 PPL | 8.1256 | 8.1689 |
| 8B | C4-val 256×2048 PPL | 13.3423 | 13.4054 |
| 8B | GSM8K non-thinking | 90.22% | 90.98% |
| 8B | MATH-500 non-thinking | 73.80% | 75.60% |

量化后个别离散 accuracy 上升不能解释为能力提升；确定性解码下，量化扰动会
让部分边界样本由错变对、也会由对变错。PPL 的一致小幅上升更适合描述整体
扰动。结论只能是 W8A8 对这些通用 benchmark 近似无损。

### INT4 明显破坏推荐能力，LWC 能恢复一部分

统一新数据、AD-3000、1.7B：

| 方法 | Activation | P@1 | P@32 |
|---|---:|---:|---:|
| BF16 | BF16 | 0.0157 | 0.2053 |
| RTN INT4 | BF16 | 0.0030 | 0.0277 |
| 旧 symmetric LWC | BF16 | 0.0133 | 0.0863 |
| 旧 learned LET + LWC INT4 | FP8 | 0.0000 | 0.0000 |

旧 symmetric LWC 结果在
`artifacts/results/fake_quant/omniquant_w4a16_ad3000_v2/`。它相对 RTN 提升
显著，但仍远低于 BF16。INT4-W/FP8-A 的 learned LET 结果全零，说明当前 LET
组合尚不可靠，不能作为正向结果。

其他低比特控制也支持相同趋势：1.7B WikiText-2 从 BF16 PPL 11.87 上升到
INT4-W/BF16-A 27.22；GSM8K 从 67.17% 降到 INT4-W/FP8-A 47.99%。因此低比特
并非只损害推荐任务，但推荐 SID 指标对 naive INT4 的崩塌尤其明显。

## Group-wise OmniQuant-LWC（2026-08-25）

FP symmetric LWC 已支持标准 weight group-wise 截断：`weight_group_size=128`
时，每个 output channel 的每个连续 128-input 权重组分别学习 clipping
参数；尾组显式屏蔽 padding，checkpoint 与 run config 都记录 group size。

首个受控实验入口为：

```text
scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_groupwise_g128_lwc_abc_boundary.sh
```

它只训练 g128 的 ABC-CE + 0.3 boundary 组：Layer 0--26 使用 calib 前
128 条，Layer 27 使用前 512 条训练。训练后自动在未参与反向传播的后 512
条上输出 A/B/C 的 CE、KL、top-1 一致率、top-5/10/32 保留率、越界率及
near/far intruder rate，并写入同一 `RUN_ROOT` 下的 JSON；可用
`RUN_HELDOUT_DIAGNOSTICS=0` 关闭。诊断入口新增 single-checkpoint 模式，
原三组受控比较模式保持不变。

g128 实验已完成，并与相同协议的 per-channel ABC+0.3 boundary 对齐：
Layer 26 prefix MSE 从 76.2231 降到 36.6217；held-out512 macro CE 从
4.3671 降到 3.9436，top-1 从 59.31% 升到 72.98%，top-32 从 60.89%
升到 69.42%，boundary violation 从 21.24% 降到 15.78%。A/B/C 三个
slot 均同向改善。需要保留的反例是 Layer 27 在 ABC+boundary 优化后
block MSE 从 88.55 上升到 246.94，说明任务目标仍会牺牲全局重构。

下一轮六组 AD-full 核心矩阵入口为：

```text
scripts/fake_quant/run_1p7b_ad_full_w4w8a8_g128_core6_cuda67.sh
```

它对 W4A8/W8A8 分别运行 RTN-g128、MSE-LWC-g128 和
ABC+0.3-boundary-g128。两个 learned arm 共享各自的 prefix，Layer 27
使用完整 calib1024；W4 默认复用上述已完成 prefix，最后层两分支分配到
前两张 GPU 并行训练，六组 AD-full 评测完成后写出统一汇总 JSON。

## 当前下一步（2026-08-24）

旧 INT4-W/INT8-A ABC-LFQ 实验表明：在旧数值路径下，最后一层的
calibration 从 128 增加到 512/1024，并给予足够 epoch 时，AD-full
推荐指标整体改善。但这些 checkpoint 早于当前 deployment-matched
训练语义，只能作为历史趋势，不能与新 boundary loss 做严格对照。

下一轮不重跑旧 INT sweep，而在当前代码下进行 FP W4A8 三组受控对比：

1. 使用 calibration 的前 128 条重新训练 FP4-E2M1-W/FP8-E4M3-A 的
   LWC-only Layer 0--26 prefix；
2. 从同一 prefix 和最后一层初始点分出 MSE-LWC control、原 ABC-LFQ 与
   `ABC-LFQ + boundary` 三组；
3. 三个 Layer 27 分支使用同一 train512/held-out512 划分、epoch、seed；
   两个 LFQ 分支额外保持相同 slot 权重；
4. 先在独立 512 条样本上检查 ABC KL/CE、teacher top-32 保留率与越界率，
   再进入 AD-3000，只在两者都正向时跑 AD-full；
5. FP W4A8 成立后再迁移至 FP W8A8，检查高精度场景是否也有收益。

上述训练与 held-out 诊断已经固化到：

```text
scripts/fake_quant/run_1p7b_ad_fp4w_fp8a_lwc_abc_boundary_calib1024.sh
```

launcher 先在 `GPUS` 首卡训练共享 prefix，再将三个最后层分支轮转分配到多卡，
全部完成后回到首卡执行 held-out 诊断。单卡模式仍受支持。
脚本和自动协议校验已通过 dry-run 与完整测试，但尚未启动真实 1.7B calibration。

FP LWC 是 symmetric 单截断参数，可调空间小于旧 INT asymmetric
双边 LWC。若 boundary proxy 能改善而推荐指标不变，应优先怀疑最后一层
参数空间不足；若 proxy 本身也无法改善，再检查 loss 定义和系数尺度。

## 接手检查清单

- 从仓库根目录执行命令，launcher 使用 `scripts/` 下的正式路径；
- 推荐数据只使用 `/root/dataDisk/guowei/data/onerec_data/benchmark_data`；
- 核对模型大小、calib/test 数量、seed、beam 和 activation 格式；
- real FP8 与 fake-QDQ 不得混作时延、显存或真实吞吐结论；
- generation 时延不包含离线 Hessian/OmniQuant calibration；
- 运行后检查 config、逐层 loss、checkpoint 数量、生成样本数和最终
  `eval_results.json`；
- 新 asymmetric 实验中 loss 变差不会回退，这是预期行为，不要误判为仍在
  使用初始化参数。
