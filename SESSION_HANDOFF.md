# OOR-Quant 会话迁移说明

更新日期：2026-07-30。本文件记录当前研究主线、有效结果、代码状态与下一步
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
artifacts/                  模型、数据、结果和探测归档（Git 忽略）
```

实验 launcher 只保留在 `scripts/` 下。测试套件、pytest 配置、缓存、重复脚本、
decode-A16 fake-quant ablation 和一次性 profile/分布探测代码已删除。当前没有
可运行的 pytest suite；修改后至少应做核心模块 import、CLI smoke test 和小样本
实验。缓存目录可以随时删除。

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
artifacts/models/1.7B
artifacts/models/8B
```

推荐任务今后统一使用：

```text
artifacts/data/onerec_data/benchmark_data
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

## Fake-QDQ 与 OmniQuant

主入口：

```bash
python -m fake_quant.run_m1_onerec_ad --help
```

`baseline_qdq` 支持权重和 activation 独立选择：

```text
none / fp8_e4m3fn / fp4_e2m1 / int8 / int6 / int4
```

主实验协议统一使用整数权重与整数 activation。入口默认是
`baseline_qdq + asymmetric INT8-W + dynamic per-token symmetric INT8-A`；
例如 W4A8 只需把权重格式指定为 `int4`。INT4-W/BF16-A 和 FP8 组合只作为
显式消融或历史/部署对照，不与整数 PTQ 主结果混用。整数权重采用
per-output-channel QDQ，整数 activation 采用 dynamic per-token symmetric
QDQ；fake-QDQ 最终仍以模型 dtype 执行 `F.linear`。`fp4_e2m1` 使用标准有限
E2M1 codebook、round-to-nearest-ties-to-even、权重 per-output-channel scale 与
activation dynamic per-token scale；浮点权重格式只允许 symmetric QDQ。

统一数值契约（2026-08-18 起）为 deployment-matched，适用于 RTN、
SmoothQuant、GPTQ、OmniQuant 和 LFQ：FP32 只用于 master weight、量化参数、
scale/zero-point/QDQ 算术及 loss；QDQ 后的权重与 activation 必须先转回模型
dtype，再执行 Linear、RMSNorm、attention、MLP、残差和 LM head。该契约没有
兼容开关。旧 OmniQuant checkpoint 属于 FP32-surrogate 校准路径，元数据校验
会拒绝加载，必须重新校准。

OmniQuant 实现在 fake_quant/omniquant/runtime.py，采用逐 block 的输出重构
MSE。当前包含：

- LWC：每个 Linear、每个输出通道学习 clipping 参数；INT4/INT6/INT8 支持 symmetric
  和 asymmetric，FP8 E4M3FN 支持 zero-centered symmetric clipping；
- LET-QKV：input norm 到共享 Q/K/V scale；
- LET-MLP：post-attention norm 到共享 gate/up scale；
- LET-V/O：适配 Qwen3 GQA head 排列的 V→O scale；
- `--omni_let_mode none|fixed|learned`：分别表示关闭、固定 SmoothQuant
  初始化、学习 LET。

默认校准协议为128条样本；OmniQuant 默认优化20 epochs，LWC/LET 学习率
分别为1e-2/5e-3，weight decay 为0且不做梯度裁剪。固定 SmoothQuant
与 OmniQuant 的 SQ-init 共用 alpha=0.4，该值由 AD calibration block MSE
搜索得到。

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

## 当前下一步

最优先实验是 LWC-only、asymmetric INT4-W/BF16-A，在同一 AD-3000 协议下与
旧 symmetric LWC 结果比较。命令：

```bash
python -m fake_quant.run_m1_onerec_ad \
  --task ad \
  --mode omniquant \
  --model_path artifacts/models/1.7B \
  --weight_quant_format int4 \
  --activation_quant_format none \
  --omni_weight_quant_scheme asymmetric \
  --omni_let_mode none \
  --calib_sample_size 128 \
  --eval_sample_size 3000 \
  --device cuda:7 \
  --output_dir artifacts/results/fake_quant/omniquant_asym_lwc_w4a16_ad3000 \
  --evaluate \
  --overwrite
```

比较基线：

```text
artifacts/results/fake_quant/omniquant_w4a16_ad3000_v2
```

该旧结果中所有层的训练后 loss 都低于初始化 loss，因此旧自动回退实际上没有
触发；新实验与它的主要有效差异应是 asymmetric 双边 clipping。先完成这一
对比，再决定是否继续修正 LET。不要同时改变 activation 格式、LET、epoch、
calibration 样本或评测子集。

## 接手检查清单

- 从仓库根目录执行命令，launcher 使用 `scripts/` 下的正式路径；
- 推荐数据只使用 `artifacts/data/onerec_data/benchmark_data`；
- 核对模型大小、calib/test 数量、seed、beam 和 activation 格式；
- real FP8 与 fake-QDQ 不得混作时延、显存或真实吞吐结论；
- generation 时延不包含离线 Hessian/OmniQuant calibration；
- 运行后检查 config、逐层 loss、checkpoint 数量、生成样本数和最终
  `eval_results.json`；
- 新 asymmetric 实验中 loss 变差不会回退，这是预期行为，不要误判为仍在
  使用初始化参数。
