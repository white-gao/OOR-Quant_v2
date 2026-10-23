# 推理加速实验方案初稿

**已确认：只测 OneRec-1.7B；RTN-W8A8 与 RecQuant-W8A8 两组柱，BF16 为 1× 基准线。左图 Prefill、右图 Decode，各六根柱，共十二根；不展示 Overall，不安排显存实验，不展示绝对毫秒数。**

配套 [图稿](figures/inference_speedup_mockup.png) 仅演示布局：**柱高全部是人为示意值，不是实测或性能预测，不能用于论文结论。** 当前未启动 GPU 实验。

## 1 图像与实验设置

| 项目 | 左图：Prefill | 右图：Decode |
| --- | --- | --- |
| 横轴（建议档位，待可运行性检查） | 输入长度 512 / 1024 / 2048 tokens | 请求 batch 1 / 2 / 4 |
| 固定设置 | 请求 batch=1 | 输入长度=1024 tokens |
| 每档两根柱 | RTN、RecQuant | RTN、RecQuant |
| 计时阶段 | 完整 prompt 的首轮生成迭代，得到 SID-A 候选 | 后续预测 SID-B、SID-C 的两轮合计 |
| 纵轴 | 相对同档 BF16 的阶段加速比 | 相对同档 BF16 的阶段加速比 |

共同设置：1.7B、真实 W8A8（FP8 E4M3FN）算子、per-output-channel 权重 scale、beam=32、返回 32 个候选、固定生成三个 token、完整词表、非采样。**本版 BF16/RTN 两阶段实测；RecQuant 两阶段均为算子实测支撑的估算，不是已部署模型时延。** batch 指请求数，不是 beam 数；从真实生成中捕获展开后的形状，不把 decode 当作 greedy decode。

两个子图共享纵轴，柱状图从零开始；BF16 画 1× 虚线，柱顶标倍数，正式结果加 95% 区间。横轴不同是有意设计：左侧观察输入长度影响，右侧观察请求并行度影响。图注写清固定设置，不将左右同序号柱当成同一负载。

## 2 对比口径与实现前提

- 使用同一卡、同一软件环境和 SDPA backend，不并发测速。BF16 和 RTN 测完整阶段；RecQuant 的独立算子探测采用对应的真实 FP8 GEMM。覆盖范围和生成设置对齐，LM head、embedding、KV cache 保持 BF16。关闭单 token A16 回退和实验性 QKV/gate-up GEMM 融合；仍保留共享输入量化。
- RTN 沿用 `real_quant/naive_w8a8` 的 minmax 权重量化及真实 FP8 GEMM。RecQuant 必须使用已训练 checkpoint 的变换、clipping、权重码点与 scale；离线 folding 不计时，实际在线矩阵变换与激活量化开销必须计入。
- **当前实现路径：不先构建完整 RecQuant 部署后端。** 恢复已训练的第 0、13、27 层，导出 learned FP8 码点和 scale，核对其重构等于原 checkpoint 的 QDQ 权重，再测完整共享投影组。每组计入在线矩阵变换、LAC、一次共享 dynamic quant、GEMM；离线参数折叠不计入。没有用 fake-quant 完整模型时间冒充 real-quant。
- 两个阶段统一使用 `T_RecQuant_est = T_RTN_measured + sum(实际调用次数 × mean_probe_layers(T_RecQuant_group − T_RTN_group))`。RTN 实测保留 attention、LM head、beam 等未量化部分。该估计假设这些部分耗时近似不变，并用代表层替代所有层；缓存状态、算子调度、融合和数据值差异会引入误差。它不是严格可加的部署保证，不能与实测混称。
- 这张图只比较低精度推理路径，不比较 FlatQuant 与 RecQuant，不将加速归因于对齐 loss。RecQuant 相对 RTN 的差异由完整在线路径决定，不预设谁更快。

## 3 请求、阶段计时与统计

1. **固定负载。** 从 AD test 的 messages 中取上下文，移除可能的末尾 assistant 答案，以 seed=42 抽取记录并构造受控 token 流。长上下文截取后缀，短上下文重复填满，最后附 SID 起始标记；无 padding。这是 shape benchmark，不是原始聊天模板或完整真实请求的质量测评。默认每档 8 个不同请求批次，可用 `--samples 32` 扩展；记录完整 token IDs 和输入构造方式。各方法输入相同，不按结果筛选。
2. **阶段边界。** BF16/RTN 以第一次 decode forward 之前为分界。Prefill 包含 generate 初始化、首轮 prompt 前向、LM head 和第一步 beam 选择；Decode 包含后两轮前向、beam、缓存和候选收尾。输入已在 GPU 上，tokenization、加载、离线校准和转字符串不计时。RTN 的缓存由自己的 prefill 生成；RecQuant 不执行完整缓存生成，而是做上述阶段估计。
3. **计时实现。** 新增阶段记录；当前 `generator.py` 的 `decode_time` 是转字符串耗时，不可用作模型 Decode。阶段计时覆盖完整迭代的墙钟时间，正确同步 CUDA，不只统计 GEMM 或只累计 kernel 时间。分段同步可能扰动执行，需用不插入中间同步的完整生成计时做内部核验；总体时间不出图。正式测速关闭 profiler。
4. **固定工作量。** 确认每次完成相同的三个 SID 生成步骤，记录提前停止情况；若需固定步数测试，所有方法使用一致配置并注明。不要把提前终止产生的少算一步解释为加速。
5. **预热与重复。** 阶段每档预热 3 次、8 个请求批次各重复 5 次；算子预热 10 次、重复 50 次，并随机轮换 RTN/RecQuant 算子顺序。完整模型当前固定先 BF16 后 RTN，不能宣称消除了硬件时间漂移；需保持独占 GPU。三次不分段同步的生成作为内部核验，若分段与不分段总时间差异超过 10%，汇总给出警告。原始毫秒数保留，图只展示比值。
6. **加速比。** 对每个请求批次先平均重复时间，再平均同档批次。每根柱为 `mean(T_BF16_stage) / mean(T_method_stage)`，BF16 参照必须匹配阶段、输入长度和 batch。不要反写比值，也不要平均逐样本倍数。
7. **误差区间。** 以请求批次为配对单位做 2,000 次 bootstrap；重复计时不是独立请求。RecQuant 区间以算子均值为条件，不涵盖算子估计和代表层近似的系统误差，图注需说明。保留低于 1× 的结果；估计时间非正则报错，不截成“合理”数值，不平均两阶段倍数推导总体。

## 4 执行顺序与写作边界

1. **CPU 预检：** `--check-only` 检查模型、数据、checkpoint 和配置，不初始化 CUDA、不写结果。本次已通过 CPU 单元测试，未执行 GPU 测速。
2. **有空卡后：** 先 `--smoke` 验证链路；再跑默认设置。入口会先实际调用 FP8 kernel，不支持则直接报错，不回退到 fake-quant 或 CPU。Smoke 的输入长度和重复次数都缩小，不能充当论文数据。
3. **出图与分析：** 本版原始记录、CSV/JSON 和图像统一存 `paper/efficiency/results/<run_tag>/`；不覆盖旧示意图。区分实测与估算，结合分段同步核验解释异常，不预设单调趋势。

图题建议：*Prefill and decoding speedups of OneRec-1.7B.* 图注交代 W8A8、BF16 归一化、GPU、beam、负载设置，并明确 RTN 为实测、RecQuant 为 operator-based estimate 及区间含义。并非生产服务响应时延，也没有部署端到端 RecQuant 性能保证。

## 5 当前布局预览

```bash
python paper/efficiency/scripts/plot_speedup_mockup.py
```

示意值存于 `mockup/illustrative_speedups.json`，绘图器强制检查 `data_kind=illustrative`。按排版要求，图内仅保留子图标题、图例、坐标轴与柱顶数值，不显示警示文字、模型配置、固定设置说明或水印。文件名保留 `mockup`，PNG/PDF/SVG 元数据及本文件继续注明 **SYNTHETIC DATA — NOT MEASURED**；去掉图内文字不改变数据的示意性质。展示这份预览时在图外注明“布局示意，非实测结果”；正式结果另设实测数据入口并替换数值。

## 6 启动与结果

CPU 检查（不用等空卡）：

```bash
bash paper/efficiency/scripts/run_stage_speedups.sh --check-only
```

选一张空卡先测通链路，再正式收集；以下 GPU=6 可替换，默认输出目录带时间戳：

```bash
GPU=6 bash paper/efficiency/scripts/run_stage_speedups.sh --smoke
GPU=6 bash paper/efficiency/scripts/run_stage_speedups.sh \
  --output-dir paper/efficiency/results/1p7b_w8a8_stage_speedups
```

有逐阶段日志及进度条。入口发现卡上已有计算进程会退出；不要为了测速绕过此检查。中断后，保持原参数、同一卡与环境，添加 `--resume` 并指定原输出目录；复用已完成的负载和算子探测，未完成负载重新测。文件校验不一致则拒绝混用。

输出包括：`inputs.json`（固定输入）、`manifest.json`（配置、版本和 checkpoint 哈希）、`raw/`（BF16/RTN 两阶段原始时间）、`operators.json`（分组、形状、代表层的两种 FP8 路径时间）、`summary.csv/json`（12 个点及其数据类型）、`stage_speedups_estimated.png/pdf/svg` 和 `.caption.txt`。不保存大模型副本或生成结果列表。

只重新汇总、出图，不用 GPU：

```bash
bash paper/efficiency/scripts/run_stage_speedups.sh \
  --output-dir paper/efficiency/results/1p7b_w8a8_stage_speedups --summarize-only
```
