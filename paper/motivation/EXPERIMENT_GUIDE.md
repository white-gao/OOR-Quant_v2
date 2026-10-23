# Motivation：实验执行指南

本文保留可执行的动机实验方案，单独维护配置、命令和结果管理。
完整写作规划 `MOTIVATION_PLAN.md` 为本地私人草稿，不随迁移提交上传，也不是运行脚本的依赖。

更新：2026-10-16。对应 `3_Preliminaries.tex / Motivating Observations`。**AD 已完成并用于正文初稿；Product/Video 尚待补齐。绘图仅聚合已有记录，不占 GPU、不重训。**

当前主图文本参照已改为 **WikiText-2**，其余对照择要放补充材料。采集、长度控制及绘图均已完成，命令见第 5.5 节；正文已采用新实测数值，`paper/Appendix.tex` 精简为必要协议与两张关键结果表，目标一页多。完整统计和诊断图仍保存在实验目录，不全部排入论文。真实 SID-A 数值核查见第 6 节。**v1 图和第 1–4 节的旧启动器仍使用 CMRC2018，保留为历史版本，不与新主图混用。**

## 1. 论证线与配图

**候选概率分散 → 有限量化误差可能改变关键比较 → 完整 SID 路径偏移 → 校准需要兼顾条件分布与候选边界。**

正文写三个观察：①候选竞争有多分散；②数值接近与决策保持的关系；③完整路径的偏移。最后一小段引出两个对齐目标，不提前展开公式。首页图交代挑战，本节提供统计证据，消融实验验证方法贡献。

主图采用紧凑横排三联图，突出“首步分散 → 比较易变 → 路径偏移”。优先 AD，Product/Video 验证跨域一致性，不按结果挑选展示域。完整条件诊断移入补充图，不删除记录或改动统计口径。

| 面板 | 图像与数据 | 要回答的问题 |
|---|---|---|
| (a) 首步候选集中度 | 中文文本与 SID-A 的累计 Top-k 概率曲线、95% CI，标注 Top-32 概率质量；同一 BF16 模型、完整词表 | 初始 SID 决策是否更分散？明确只谈 A，不将 B/C 更集中的条件分布写成相反结论 |
| (b) 跨 cutoff 比较 | 热力图：行是 W8A8/W4A8 × A/B/C，列是固定 logit-gap 桶，格内为严格反转率（%）；教师 ranks 1–32 对 33–64 | 较小间隔是否更易翻转？保留全部位置/精度，只省略所有行都无样本的列；无样本格标破折号，有样本的零值保留 |
| (c) 路径与任务后果 | 实际 A、AB、ABC beam 保持率及 CI，图内附 BF16/W8A8/W4A8 的同子集 SID Recall@32 | 完整路径偏移多大、是否伴随任务损失？A/AB 用真实 live beams，ABC 用实际返回列表 |

补充输出 `diagnostics_<task>.pdf/png`：保留完整 A/B/C/文本概率曲线、全样本 KL—Top-32 散点、带 CI 的反转率曲线及路径曲线；`nmse_retention_<task>.pdf` 保留末 block 输出 NMSE—Top-32 散点。NMSE 仅统计三个预测位置，不冒充全 token 训练 MSE。`summary_<task>.json` 保留全部统计，包括熵、合法词表质量、tie-aware 保留率、并列数量、Recall@1/8/32 和配对差值。

正文与图注交代模型、域、样本数、精度、前缀和 CI 单位。热力图仍先按用户求比例再聚合，不混合所有 pair；颜色固定为 0–100%。路径 overlap 不是因果分解，也不等于任务正确率；同子集 Recall 不代替全量主表。低 KL 时的尾部换位不作为单独的主要证据。

## 2. 模型与实验协议

- **模型**：OneRec-1.7B，28 blocks。教师 BF16；学生为已有 per-channel FlatQuant W8A8/W4A8 MSE 基线，复用检查点、不重训。先不跑 8B，不将 RecQuant 提升混入这轮机制诊断。
- **默认路径**：模型 `/root/dataDisk/guowei/models/1.7B`；数据 `/root/dataDisk/guowei/data/onerec_data/benchmark_data`。
- **检查点**：`artifacts/results/flat_quant/official_fake_quant/1p7b_{task}_{w8a8,w4a8}_pc_prefix128_final512_heldout512_epoch15/final_train512_heldout512/1.7B/{task}/flatquant_calibration/`。预检全部 28 个 layer 文件、精度和 per-channel 配置。
- **推荐抽样**：test 固定随机顺序，seed=42；排除整个 calib1024 的用户及相同 messages，每个用户最多一条。条件诊断为 AD 512 / Product 256 / Video 256；beam 诊断取相同顺序的前 AD 256 / Product 128 / Video 128，是条件诊断的子集。不能再把曾用于训练的 calib 后 512 条作为独立 held-out。
- **条件前缀**：A 用 prompt，B/C 追加同一个真实目标 SID 的 A/AB；取 ground truth 中第一个 SID，与现有训练一致。不按教师概率或实验表现筛选样本。
- **中文文本**：同一 OneRec-1.7B BF16，在 [CMRC2018 官方 dev 数据](https://github.com/ymcui/cmrc2018)的中文 `context` 上诊断原文续写。随机取 512 篇不同文章各一个去重段落，每段均匀抽 4 个预测位置；前缀至少 128 tokens，段落最多 1024 tokens。不输入问答，不筛除标点或困难位置；CI 按文章聚合，不把 2048 个位置视为独立样本。
- **推理**：单卡、batch=1。实际 beam 宽度 32、返回 32 条、生成 3 tokens、非采样；完整词表、无 SID mask，与现有评测一致。合法 slot 词表只用于条件诊断，不改变推理。
- **统计**：FP32 softmax/指标，2000 次用户/文章 bootstrap。条件排序并列按 token ID 固定打破，另报 tie-aware 版本；教师 gap=0 单列，(0,0.01] 单独一桶，学生 gap=0 不计严格反转；记录每桶有效用户数、pair 数及学生并列数。真实 beam 沿用 HF 的 tie-breaking。

**结论边界**：同一推荐模型的中文文本对照只能支持“本模型在两类上下文中的分布差异”，不能独自证明“通用大模型普遍更低熵”或“通用 PTQ 在通用任务上普遍更好”。如保留这些强论断，另补原始 Qwen3-1.7B 文本控制及原生通用任务量化评测；当前本地未发现原始 Qwen 检查点，本轮不自动下载模型。也不预设 A→B→C 条件熵单调增加。

## 3. 三个启动任务

统一入口 `scripts/run_task.sh`。每个任务只占一张卡，内部串行加载模型；`GPU` 为物理卡号，默认 6，`TASKS` 默认 `ad`。脚本采用 benchmark 环境的 Python。

```bash
# 先预检；不加载 GPU 模型
TASKS="ad product video" bash paper/motivation/scripts/run_task.sh preflight

# E1：中文文本 + BF16 SID 分布，服务图 (a)
GPU=2 TASKS="ad product video" bash paper/motivation/scripts/run_task.sh e1

# E2：BF16/W8A8/W4A8 相同前缀诊断，服务图 (b) 和补充图，自动复用已完成的 BF16
GPU=3 TASKS="ad product video" bash paper/motivation/scripts/run_task.sh e2

# E3：真实 beam 路径与小样本任务指标，服务图 (c)
GPU=4 TASKS="ad product video" bash paper/motivation/scripts/run_task.sh e3

# 或一次串行完成所有 probe；下面默认只做 AD
GPU=6 bash paper/motivation/scripts/run_task.sh all

# 三项完成后绘图，不需要 GPU
TASKS="ad product video" bash paper/motivation/scripts/run_task.sh plot
```

E3 可与 E1/E2 并行；**不要同时对同一个域运行 E1/E2**，两者共享 BF16 条件缓存，文件锁会拒绝重复写入。也可优先 E2/E3、之后补 E1。当前 benchmark 环境已可绘图，AD 可直接执行 `TASKS=ad bash paper/motivation/scripts/run_task.sh plot`，无需重跑 probe。

首次 E1 下载官方 CMRC dev JSON 至 `cache/cmrc2018_dev.json`。离线时自行放置 SQuAD 格式文件，或设置 `TEXT_JSON=/path/to/file.json`；不自动下载模型。可用 `PYTHON_BIN`、`MODEL_PATH`、`DATA_DIR` 覆盖环境路径；原始 Qwen 的可选文本控制只运行 Python 入口的 `text` 任务并使用单独结果目录，不混入当前主图。

```bash
# 只打印命令，不启动模型
DRY_RUN=1 GPU=2 bash paper/motivation/scripts/run_task.sh all
# 小样本检查完整流程；与正式结果隔离
RUN_TAG=smoke N=2 BEAM_N=2 TEXT_N=2 GPU=2 bash paper/motivation/scripts/run_task.sh all
```

正式实验保持默认样本数；改变 seed/样本数/采集或模型源码/检查点时换 `RUN_TAG`，仅调整绘图样式不需要重跑。脚本有进度条、逐样本原子保存，原命令可续跑；配置不一致会拒绝覆盖。beam hook 针对 transformers 4.57.6 编写，CPU 微型模型测试已检查记录前后输出一致；不支持的 API 显式报错。

## 4. 保存、验收与写作

所有新增产物均在本目录，外部原始数据和检查点只读：

```text
paper/motivation/
  MOTIVATION_PLAN.md               # 本地可选写作草稿，不通过 Git 迁移
  EXPERIMENT_GUIDE.md              # 当前执行方案与启动命令
  scripts/                         # 启动、采集、指标、beam 记录、绘图
  tests/                           # CPU 单测
  results/<RUN_TAG>/
    text/bf16/                      # 中文文本分布
    <task>/{conditional,beam}/{bf16,w8a8,w4a8}/
      manifest.json                # 抽样身份、协议、源码哈希、版本、输入文件信息
      records/<sample_id>.json      # 样本级指标；beam 任务含路径/分数
      cache/                       # 仅 conditional BF16：合法 slot logits 和末块隐藏状态
      COMPLETE.json                # 完成标记
  figures/<RUN_TAG>/                # 三联主图、完整诊断图、NMSE 图、summary_<task>.json
  logs/<RUN_TAG>/                   # 按任务/域/精度记录日志
  cache/                           # 文本数据、运行缓存、CPU 测试临时产物
```

不保存全词表逐 token logits、KV cache 或模型副本；产物已加入局部 `.gitignore`。绘图只接受完整任务，保留全部抽样点，汇总附 CI 和有效样本数。

验收：①确认样本独立、前缀/精度正确；②检查合法概率质量、并列和异常提前停止；③查看三联主图、完整诊断图及 NMSE 补图；④结合全量主表解读小样本 Recall；⑤按证据写三个观察，不支持的强断言删去或收窄。

**本轮不追加训练**：严格同初始化、同 1024 条 GT-prefix、同联合参数、同 15 epochs 的 MSE vs RecQuant 对照放消融实验。已有冻结矩阵或仅 prompt 的 MSE-1024 不自动视为严格对照。本轮不宣布两个 loss 不可替代，也不把局部 token cutoff 等同于全局 beam cutoff。

## 5. WikiText-2 主参照：下一版实验方案

### 5.1 目标、主图与复用范围

问题限定为：**同一个模型在 WikiText-2 自然文本续写和 AD 首个 SID 决策上的概率集中度有何差异？** 这是预测分布诊断，不是 WikiText-2 标准全量 PPL 评测，也不单凭它证明“所有通用任务都低熵”。

- 主模型：现有 OneRec-1.7B BF16；文本和 SID 使用相同权重、tokenizer、完整模型词表。原始 Qwen 只作独立补充控制，不混入同模型主曲线。
- 图 (a)：WikiText-2 与 AD SID-A 两条累计概率曲线，直接标注各自实测 Top-32 质量；图例明确数据来源。
- 图 (b)(c)：直接复用当前 AD 的反转率、beam 保持率和同子集 Recall，不重训、不重跑 AD 模型前向。
- 现有 512 条 AD 用户、256 条 beam 用户不换样本；当前 CMRC2018 的全部记录保留为中文补充对照。

### 5.2 WikiText-2 数据与采样

使用 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) 的 `wikitext-2-raw-v1`、`test` split，固定 dataset revision 和文件 SHA256。采用 raw 版本，避免词表外单词被预先替换成 `<unk>`；不另行改写原文来提高可预测性。[数据说明](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md)

**独立单位是文章，不是 Hugging Face 的文本行。** 原论文报告 test 有 60 篇文章，数据卡的 4,358 行不是 4,358 篇独立文档。之前通用建议中的“512 个独立文档”不能直接套到 WikiText-2 test。[原论文 Table 1](https://arxiv.org/html/1609.07843)

预先固定以下协议，不能看过熵或 Top-32 质量后再改抽样：

1. 按原始文章顶层标题重建文章边界，保存文章 ID、标题、原始行范围和文本哈希；不把不同文章拼成一个预测上下文。标题需为单层等号且前后有空行，避免把正文中形如 `= 1 henry and capacitance C =` 的公式行当标题。真实数据预检已重建出 60 篇。
2. 使用全部合格测试文章；每篇以 seed=42 均匀、无放回抽取至多 32 个 next-token 预测位置，目标约 1,920 个位置。预测位置必须有至少 128 个前文 tokens 和一个实际下一 token；不足 32 个时全部保留，无法满足最短上下文的文章单列原因，不从 train/validation 补齐。
3. 每个位置仅输入其之前至多 1,024 个 tokens，不跨文章边界，不输入待预测 token；不加 chat template、问答提示或 SID-begin token。保留原数据中的标点、空白和标题等，不按目标词、置信度或预测正确性筛样本。
4. 分开记录文章数、预测位置数、上下文长度及截断情况。正文按实际数量报告，不把 1,920 个位置写成 1,920 个独立样本。
5. 这是子集预测分布诊断；原始数据的预训练暴露未知，不称其为模型从未见过的文本。

### 5.3 计算口径与稳健性检查

- BF16 模型输出转 FP32 后计算完整词表 softmax，temperature=1；不对文本词表或合法 SID 子词表重新归一化主图概率。
- 统计 Top-k 质量（k=1/2/4/8/16/32/64/128）、熵；先在文章内平均，再对文章等权平均。95% CI 使用 2,000 次文章级 bootstrap；AD 则按用户。补充表同时报告文章/用户层面的中位数，不能把不同聚合口径混比。
- 保存每个位置的 Top-128 token IDs/概率、完整词表 log-normalizer、熵、目标 token NLL、上下文长度及文章/位置 ID；保留字符偏移以便核对不同 tokenizer 的目标边界。只保留这些轻量统计，不保存全词表 logits 或完整激活。
- NLL、预测 top-1 是否命中真实下一 token、误分到 SID token 的质量用于检查 OneRec 在英文上是否异常；不据此删除样本。抽样 NLL 不能改名为标准 WikiText-2 全量 PPL。
- 上下文长度检查：当前 AD prompt 中位数 721.5 tokens，范围 214–1,158；不能假定新文本与推荐输入等长。补充报告 `[128,256)`、`[256,512)`、`[512,1024]`、`>1024` 各区间的数量与集中度，仅比较有共同支持的区间；最后一档文本无覆盖时标明缺失，不补造结果。
- 长度敏感性检查以文章为聚合单位，对相同位置构造 256/512/1,024-token 上下文（仅使用三种长度都满足的固定位置），最多每篇 8 个位置；报告结果是否随长度显著变化，不拿新增前向重新挑主图样本。
- 验收前用少量位置比对“完整因果 forward 的指定位置”与“仅前缀 forward 的最后位置”，确认 next-token 索引、原文截断、dtype 和概率计算一致；BF16 的微小数值差异记录容差，不承诺逐位完全相同。

### 5.4 补充材料与优先级

| 优先级 | 实验/材料 | 配置与用途 | 新增工作 |
|---|---|---|---|
| P0，已完成 | SID-A 数值审计 | 原始 JSON、512 份缓存 logits、历史 5 样本及真实 beam 交叉检查，见第 6 节 | CPU 读取，不重跑模型 |
| P1，必做 | WikiText-2 主参照 | OneRec BF16；60 篇测试文章、每篇至多 32 位置；主图 (a) | 新文本前向；先 smoke 再正式采集 |
| P1，必保留 | 中文与完整分布补图 | 现有 CMRC2018、SID-A/B/C 曲线与汇总表，检查英文主参照是否改变解释 | 复用现有结果 |
| P1，验收项 | 长度与预测位置检查 | 上述共同长度区间统计、至多 60×8 个位置的长度控制 | 部分 CPU 汇总，少量额外 BF16 前向 |
| P2，增强稳健性 | [mC4 中文网页对照](https://huggingface.co/datasets/allenai/c4) | 固定中文子集及 revision，去重后随机 512 文档、每篇 4 位置；完整词表统计，语言和文体单独标明 | 新文本前向；不因结果不同而丢弃该来源 |
| P2，模型对照 | 原始 Qwen3-1.7B × WikiText-2 | 复用文章与字符级目标位置；检查 tokenizer 后建立对应，不把 OneRec token IDs 直接传入不同 tokenizer | 需要另备原始模型；当前仅查到 OneRec 1.7B/8B，不自动下载 |
| P2，跨域复核 | Product/Video 的 SID probe | 沿用既定 E1–E3 用户和 checkpoint 配置，分别报告域结果 | 独立于 WikiText-2，待有卡后执行 |

补充图/表按证据类型拆开：S1 文本来源与完整 SID 分布（每来源分开，不混成一条“general”曲线）；S2 上下文长度控制；S3 原始 Qwen 对照；保留既有 KL/NMSE、并列和 gap 桶 CI。未运行的项目标为 pending，不在论文中写成已验证结论。

WikiText-2 为英文、推荐 prompt 主要为中文且含 SID，不能仅凭主图把全部差异归因于推荐任务。已有中文对照是必要补充。原始 Qwen 与 OneRec 词表/训练状态不同，模型对照也不是严格单因素实验。若要声称通用 PTQ 的任务损失显著小于推荐，另按总计划 E5 做实际量化性能评测，本轮不扩展到该任务。

### 5.5 实施顺序、路径与切图条件

已实现独立入口 `scripts/probe_wikitext2.py` 和启动器 `scripts/run_wikitext2_reference.sh`，不修改旧采集源码及其 manifest。固定数据 revision 为 `b08601e04326c79dfdd32d625aee71d232d685c3`。预检确认 60 篇均合格，主采样 1,920 个位置；其中 454 个位置满足长度控制，共 1,362 条配对记录，1024-token 结果复用主采样，实际新增前向 908 次。

```bash
# 一条命令：预检 → WikiText-2 BF16 采集与长度控制 → 自动生成新主图和补图
GPU=5 bash paper/motivation/scripts/run_wikitext2_reference.sh

# 可选：先用 2 篇文章检查 GPU 全流程；结果与正式采集隔离
GPU=5 bash paper/motivation/scripts/run_wikitext2_reference.sh smoke

# 只预检，不加载模型；或采集完成后仅重新绘图
bash paper/motivation/scripts/run_wikitext2_reference.sh preflight
bash paper/motivation/scripts/run_wikitext2_reference.sh plot
```

默认采用 benchmark 环境的 Python 和已有 OneRec-1.7B，`GPU` 选物理卡号。数据文件已缓存；无缓存时仅下载固定版本的 raw test parquet，离线可设 `WIKITEXT_FILE=/path/to/test.parquet`，不自动下载模型。日志实时输出，采集有文章/位置进度条，每个位置原子保存；中断后原命令续跑。配置、模型或采集源码改变时须另设 `RUN_TAG` 和 `FIGURE_TAG`，不能修改旧 manifest 绕过检查。

该命令只执行 P1：复用 AD 与 CMRC，不重训、不重跑 AD、不自动启动 mC4、Qwen 或 Product/Video。绘图从 manifest 读取数据来源并检查模型一致，主图使用真实 WikiText-2 记录；中文和 A/B/C 完整曲线、长度控制与 NLL 等检查保留在补充输出中。

新增结果布局：

```text
paper/motivation/
  results/v1_seed42/                         # 原 AD 与 CMRC；只读复用
  results/v2_text_controls_seed42/
    wikitext2/onerec_bf16/
      manifest.json                         # 数据 revision/SHA、文章身份和采样位置
      records/<article_id>.json              # positions 与 length_control 分别保存
      position_sanity.json                   # 因果预测位置检查
      COMPLETE.json
  figures/v2_wikitext2_seed42/
    motivation_ad.pdf/png                   # 整张三联图，便于预览
    motivation_ad_a/b/c.pdf/png             # 三个独立面板，供 LaTeX subfigure 排版
    text_controls_ad.pdf/png/json           # 中文/英文、完整 SID、长度与健康检查
    diagnostics_ad.pdf/png                  # 完整诊断
    nmse_retention_ad.pdf
    summary_ad.json
  logs/v2_text_controls_seed42/              # preflight、wikitext2、plot 日志
```

验收：正式采集已完成，60 篇文章 / 1,920 个主位置，长度控制覆盖 58 篇 / 454 个配对位置；完成标记与原始记录一致。WikiText-2 Top-32 质量为 92.28%，熵 1.94 nats；AD SID-A 对应为 37.80%、6.13 nats。23 项 CPU 测试通过。正文与附录的数值已核对，旧 v1 结果保留。

Overleaf：将本目录 `figures/v2_wikitext2_seed42/` 中的 `motivation_ad_a.pdf`、`motivation_ad_b.pdf`、`motivation_ad_c.pdf` 上传到 `Figures/`，使用正文内的三个 `subfigure` 和独立标签；导言区需 `subcaption`。精简附录只保留两张单栏表，不再引用 `text_controls_ad.pdf`、`diagnostics_ad.pdf`、`nmse_retention_ad.pdf`；这些图仍作为本地诊断材料保留。若 Overleaf 主文件已有旧 motivation 图环境，应替换而非重复插入。WikiText-2/CMRC 引用仍待统一补 Bib。

## 6. 当前 SID-A Top-32 概率的真实数据核查（2026-10-16）

**当前图来自模型 probe 原始记录，不是示意数据。** 处理顺序为：每个用户的完整词表 FP32 softmax → 排序 → 前 32 个概率求和 → 对 512 用户取均值。没有把前 32 个候选重新归一化，也没有先平均 logits 再 softmax。

| 样本与统计口径 | Top-32 累计概率 |
|---|---:|
| 当前 AD 512 用户，均值（主图） | 37.8045% |
| 当前 AD 512 用户，中位数 | 37.2845% |
| 当前均值的 95% 用户 bootstrap CI | [37.2842%, 38.3561%] |
| 早期 5 条多正例样本，均值（此次重算） | 36.5000% |
| 早期 5 条多正例样本，中位数（旧报告） | 32.7456% |
| 早期 sample 14，单样本 SID-A | 30.1476% |

旧 5 样本分别为 14/70/113/263/268，历史采样包含多正例/指定 ID，并非当前随机 512 用户。旧“32.75%”是中位数；同样比较均值时，旧 5 条与当前 512 条只差 1.3045 个百分点，不应将两种统计直接视为模型变化。旧 sample 14 的 AB Top-32 **联合**质量为 9.8868%，不是 SID-A 的条件概率。

交叉核验：

- `COMPLETE.json` 中的 manifest 哈希、样本数和原始记录 ID 一致；主图汇总与逐条原始 JSON 重算均值完全一致。
- 对全部 512 份教师缓存 SID-A logits 重新 softmax、取 Top-32，再乘该样本的合法 SID-A 总质量，复原完整词表质量；与原始记录最大绝对差为 `4.70e-7`（概率单位，浮点误差量级）。这项核验使用完整词表统计记录和合法子词表缓存，不冒充保存了完整词表 logits。
- SID-A 合法词表归一化后的均值是 37.8159%，完整词表版本是 37.8045%；图采用后者，差别不是子词表归一化抬高导致的。
- 独立的 256 用户真实 BF16 beam 首步记录给出 Top-32 绝对质量均值 37.5112%；相同 256 用户的条件前向均值为 37.5327%。两种前向并非逐位一致（输入形状/数值路径不同，原因未逐项隔离），但均值接近；图仍使用既定 512 用户统计。
- 当前曲线在 **k=128** 处为 61.2609%，不能把右端点当成 k=32。SID/beam 保持率是集合重合比例，与累计概率也不是同一指标。
- SID-A 取 prompt 末尾、预测第一个 SID token 的 logits。虽然条件探针一次输入 prompt+A+B 采集三个位置，但因果掩码使 A 位置不能读取未来的 A/B；并非给定答案之后再预测 A。

可追溯入口：

- 当前记录与缓存：`results/v1_seed42/ad/conditional/bf16/{records,cache}/`；图汇总：`figures/v1_seed42/summary_ad.json`。
- 指标与聚合：`scripts/metrics.py::concentration`、`scripts/plot_motivation.py`。
- 旧报告：[GENERIC_PROMPT_OUTPUT_DISTRIBUTION_PROBE.md](../../docs/GENERIC_PROMPT_OUTPUT_DISTRIBUTION_PROBE.md)。
- 旧样本：[sample 14 SID-A](../../docs/figures/bf16_ad_sid_a_top50.json)、[sample 14 AB](../../docs/figures/bf16_ad_sid_ab_beam_top50_sample_14.json)，其余 `docs/figures/bf16_ad_sid_a_top50_sample_*.json`。
