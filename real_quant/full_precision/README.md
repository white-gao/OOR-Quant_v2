# HF Full-Precision Baseline

This directory contains the standalone HuggingFace full-precision baseline for
real FP8 runtime comparisons. It is intentionally separated from
`fake_quant` so BF16 latency, real naive W8A8, real GPTQ, and real GPTAQ can
share one runner without mixing in fake-quant code.

## OpenOneRec Alignment

The implementation follows the public OpenOneRec benchmark structure:

- data loading uses `benchmarks/benchmark/tasks/v1_0/registry.py::get_loader`;
- metric evaluation uses `benchmark.Benchmark.evaluate_dev`;
- outputs are saved as `<output>/<model>/<task>/<split>_generated.json`;
- recommendation prompts append `<|sid_begin|>` before SID generation;
- default recommendation generation uses `num_beams=32`,
  `num_return_sequences=32`, and `max_new_tokens=3`, matching the retained
  recommendation task configuration.

The intentional difference is the backend: this runner uses
`transformers.AutoModelForCausalLM.generate` instead of vLLM/Ray. That keeps the
baseline comparable to upcoming real FP8 PyTorch module replacements.

## Usage

AD full precision, one GPU:

```bash
PYTHONPATH=. python3 -m real_quant.full_precision.run_hf_baseline \
  --model_path /root/dataDisk/guowei/models/1.7B \
  --data_dir /root/dataDisk/guowei/data/onerec_data/benchmark_data \
  --output_dir artifacts/results/real_quant/recommender/bf16/ad_1p7b_bf16 \
  --task ad \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --max_new_tokens 3 \
  --overwrite \
  --evaluate
```

`--batch_size` defaults to `1`. This is the safest setting for accuracy
alignment because it avoids any batch-coupled effects from padding, batched
kernel choices, or future activation quantization scale implementations.

For throughput experiments, pass `--batch_size auto` explicitly. The runner then
checks the selected CUDA device total memory, infers the model size from the
model path, and chooses a conservative throughput-oriented batch size. On the
current 140GB-class cards, the auto values are:

- OneRec-1.7B: `ad/product/label_cond=8`, `video/interactive=4`;
- OneRec-8B: `ad/product/label_cond=4`, `video/interactive=2`.

Override it explicitly, for example `--batch_size 2`, if the card is already
heavily occupied or if a task OOMs. With beam search, the effective active
sequences are roughly `batch_size * num_beams`, so video and 8B runs need lower
batches than ad/product.

## Latency Fields

The generated JSON keeps the OpenOneRec-compatible fields and adds:

- top-level `latency`: aggregate tokenize/generate/decode/end-to-end stats;
- per-sample `latency`: token counts and timing;
- per-sample `input_tokens`, `output_tokens`, and `times`, compatible with the
  benchmark MFU-style schema.

For `batch_size > 1`, batch wall time is distributed across samples, so the
reported per-sample values are batch-amortized latency. With `batch_size=1`,
they are single-request latency.

For BF16 vs real FP8 comparisons, use `generate_time_*` as the primary compute
latency and keep `end_to_end_time_*` as the user-visible total.

## Generic perplexity controls

`run_ppl_benchmark.py` evaluates WikiText-2 or a cached C4 English-validation
subset with full precision, real FP8 RTN W8A8, or fake QDQ. C4 is fixed to 256
non-overlapping 2048-token windows by default. Build the shared cache once,
then launch the paired real-FP8 1.7B/8B experiment suite with:

```bash
bash scripts/real_quant/run_1p7b_8b_c4_ppl_cuda7.sh
```

For an INT4-W / FP8-A quality experiment, use the same cache and add the fake
QDQ format pair explicitly:

```bash
python -m real_quant.full_precision.run_ppl_benchmark \
  --model_path /root/dataDisk/guowei/models/1.7B \
  --dataset c4 \
  --device cuda:7 \
  --quantization fake_qdq \
  --weight_quant_format int4 \
  --activation_quant_format fp8_e4m3fn \
  --output_dir artifacts/results/fake_quant/generic/c4_1p7b_int4w_fp8a
```

## RecIF auxiliary-task baselines

`run_recif_baseline.py` adds **BF16 only** support for three tasks; this is not
yet a FlatQuant/RecQuant checkpoint runner. These are recommendation-related
capabilities inside RecIF, not a standalone general-language benchmark.

| Task | Generation / prediction | Main metrics |
| --- | --- | --- |
| `interactive` (交互推荐) | non-thinking, 32 beams, 32 candidates, 3 SID tokens | SID Pass/Recall@1,4,8,16,32 |
| `label_pred` (行为预测) | next-token probabilities of `是` / `否`, normalized over this pair | ROC-AUC |
| `item_understand` (物品理解) | non-thinking, greedy, up to 128 tokens | macro WIP double-weighted F1 |

All use the retained official `qwen3_soft_switch.jinja2` prompt template.
Text/classification prompts do **not** append `<|sid_begin|>`. Interactive
generation appends it once, without a SID-vocabulary mask. Smaller cutoffs
use the same ordered beam list. AUC uses probabilities, not hard labels.
Full evaluation reads every test-parquet row; a numeric sample size uses the
first N rows, with original row IDs. No sampled files are written into the data
directory, and malformed/skipped inputs cause an error.

Reference: [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec/tree/a969edcadd579a06c1966ae1db5984e02f48beff/benchmarks),
commit `a969edcadd579a06c1966ae1db5984e02f48beff`. We reuse its loaders, SID/AUC
evaluators, WIP prompts, matching and metric functions. The HF backend replaces
official Ray/vLLM. Interactive deliberately uses this repository's **32/32**
beam protocol, not the upstream config's 16 beams / 128 returned sequences;
only SID metrics are reported, not optional PID-mapping metrics. Therefore this
is a controlled in-repository baseline, **not a claim of exact reproduction of
the official report's scores**. Keep these settings fixed for later quantized
comparisons.

### Data and launch

If the three test datasets are missing, explicitly download just these files
(does not overwrite existing files or download AD/Product/Video):

The [official dataset](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)
is gated: request/accept access on Hugging Face and run `hf auth login` using
the authorized account first. An unauthenticated download returns HTTP 401.

```bash
python -m real_quant.full_precision.download_recif_tasks
```

The default destination is `/root/dataDisk/guowei/data/onerec_data/benchmark_data`.
An alternative must end in `benchmark_data`; pass it with `--data_dir` to the
downloader and `DATA_DIR` to the launcher. Dataset downloads pin Hugging Face
`OpenOneRec/OpenOneRec-RecIF` revision
`8f7cf2ee0b949e955a87a708d02024687be232c8`.

```bash
# Read-only validation; does not load weights, occupy GPU memory, or call a judge.
GPU=5 CHECK_ONLY=1 bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh

# Small end-to-end smoke test, with a separate output directory.
GPU=5 SAMPLE_SIZE=8 bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh

# Full serial run: interactive + label_pred scoring, item_understand generation.
GPU=5 bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh
```

By default, **item_understand scoring is pending**, not successful: it requires
external judge access. This is printed explicitly. Official task sizes are
1,000 / 346,190 / 500 respectively, so label prediction dominates sample count.
`BATCH_SIZE=1` is the conservative default; use a larger value only after a
smoke test. `MODEL_PATH`, `OUTPUT_DIR`, `TASKS`, `STAGE`, `CHUNK_SIZE`, and
`PYTHON_BIN` are optional overrides. For 8B set `MODEL_PATH` to the 8B directory.

Output layout (smoke tests use `bf16_smoke_N`):

```text
artifacts/results/recif/bf16/1.7B/<task>/
  run_config.json           # input/model/config fingerprint
  parts/                   # atomic generation checkpoints, default every 128 rows
  test_generated.json      # complete predictions + references, no long prompts
  eval_results.json        # written only after successful scoring
  per_sample_metrics.json
  wip_cache/               # item_understand only, resumable judge extraction/matching
```

Rerunning the same command resumes completed chunks. An interrupted chunk is
recomputed; no partial JSON is treated as finished. Changed data, model file
identity, batch size, seed, or generation settings require a new `OUTPUT_DIR`.
Weights use path/size/mtime identity, not a full weight-content hash. A file
lock prevents two writers using the same task output. Do not delete `parts/`
until no more generation resumption is needed. `STAGE=evaluate` only reads
completed predictions; no OneRec model weights are loaded. These runs are
quality measurements, not deployment latency benchmarks.

### Recorded full-test baselines (2026-10-22)

Both runs below completed with OneRec-1.7B **BF16**, batch size 1, seed 42,
SDPA attention, and non-thinking prompts. Sample counts match the full test
splits; prediction-file SHA-256 checksums match the scoring records.

**Interactive recommendation (`interactive`, 1,000 samples).** Metrics below
are percentages (original JSON values multiplied by 100), using SID matching
and the first K candidates from a single 32-beam / 32-return sequence list.

| Metric (%) ↑ | K=1 | K=4 | K=8 | K=16 | K=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pass@K | 7.1000 | 16.0000 | 21.4000 | 28.2000 | 33.0000 |
| Recall@K | 3.8097 | 9.1527 | 12.2138 | 16.3635 | 20.4597 |

Pass@K measures whether at least one target SID is retrieved; Recall@K is
the per-sample fraction of target SIDs retrieved, averaged across samples.
Source: [interactive/eval_results.json](../../artifacts/results/recif/bf16/1.7B/interactive/eval_results.json)
and [run configuration](../../artifacts/results/recif/bf16/1.7B/interactive/run_config.json).

**Behavior prediction (`label_pred`).** AUC uses the normalized next-token
probabilities of `是` / `否`, not generated hard labels.

| Model | Precision | Test samples | Valid samples | Invalid samples | ROC-AUC ↑ (0–1) |
| --- | --- | ---: | ---: | ---: | ---: |
| OneRec-1.7B | BF16 | 346,190 | 346,190 | 0 | 0.618539 |

Source: [label_pred/eval_results.json](../../artifacts/results/recif/bf16/1.7B/label_pred/eval_results.json)
and [run configuration](../../artifacts/results/recif/bf16/1.7B/label_pred/run_config.json).

These are in-repository full-precision reference scores for subsequent
quantized comparisons, not measurements of quantization-induced degradation.
Item-understanding WIP scores are not included in this record.

### Item-understanding WIP scoring (explicit external API opt-in)

The official metric extracts and matches weighted information points with a
judge, then uses BERTScore (Chinese BERT, layer 9) as match quality. It cannot
be replaced by ROUGE or plain BERTScore while retaining the same metric name.

#### DeepSeek (private configuration file or environment variable)

DeepSeek uses the same WIP prompts and scoring functions. The adapter calls the
[official Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
with [JSON mode](https://api-docs.deepseek.com/guides/json_mode/), reads only final
`message.content`, and defaults to non-thinking mode. The example uses the
currently documented `deepseek-flash` model ID; adjust `model_name` if your
account/provider requires a different supported model. Keep the same judge
model/settings for BF16 and every quantized comparison; scores are not directly
interchangeable with Gemini-judged scores.

Install **only this provider's** optional dependencies in the benchmark environment:

```bash
python -m pip install -r real_quant/full_precision/requirements-recif-deepseek.txt
```

For file-based authentication, edit the local, Git-ignored file
`artifacts/config/recif_judge.deepseek.local.json`: replace
`"api_key": "YOUR_DEEPSEEK_API_KEY"` with your actual key. No `export` is
needed. This file is local-only and is not distributed with Git; on a fresh
checkout, copy `real_quant/full_precision/recif_judge.deepseek.example.json`
to that private path, then add the `api_key` field. Keep file permissions at
`600` and never commit or share the filled configuration.

An explicit `api_key` takes precedence over `api_key_env`; an empty or
placeholder `api_key` is an error, not a fallback to another credential.
The runner strips `api_key` from result metadata and cache identities, so
rotating a key does not invalidate existing results or WIP caches.

Alternatively, use the public environment-based configuration
`real_quant/full_precision/recif_judge.deepseek.example.json`. Its
`"api_key_env": "DEEPSEEK_API_KEY"` names the environment variable, not the
key itself. Set it in the launching terminal (hidden input, no shell history):

```bash
read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
echo
```

With environment-based authentication, a new terminal requires setting the
key again. Neither approach writes the key to results or application logs.
URLs with embedded credentials are rejected. Never send the key in chat.

Read-only preflight (no OneRec weights, no API request, no charge):

```bash
GPU=5 TASKS=item_understand CHECK_ONLY=1 ALLOW_JUDGE=1 \
  JUDGE_CONFIG=artifacts/config/recif_judge.deepseek.local.json \
  bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh
```

Start with eight samples in the separate smoke-test output directory:

```bash
GPU=5 TASKS=item_understand STAGE=all SAMPLE_SIZE=8 ALLOW_JUDGE=1 \
  JUDGE_CONFIG=artifacts/config/recif_judge.deepseek.local.json \
  bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh
```

**This command sends reference texts and generated captions to the configured
DeepSeek endpoint and may incur charges.** It does not send recommendation
histories. Change `SAMPLE_SIZE=8` to `SAMPLE_SIZE=full` for all 500 test samples;
the default model is 1.7B BF16. Choose an available GPU instead of 5 as needed.
If generation is already complete for the selected sample size/output directory,
use `STAGE=evaluate` to run only WIP/BERTScore, without loading OneRec weights.

Changing `provider`, endpoint, judge model, thinking mode, or other configuration
isolates judge caches. Existing final metrics with another score configuration
are refused, not overwritten. To change a judge after scoring, archive
`eval_results.json` and `per_sample_metrics.json` first; the generated captions
can be reused. Timeout/connection failures, HTTP 408/409/429 and 5xx, and empty
or malformed JSON are retried with bounded backoff. Authentication, balance,
and other non-retryable HTTP errors stop immediately. `finish_reason=length`
stops with an instruction to increase `max_new_tokens`, rather than accepting
truncated output. `max_retries` is the total attempt limit (SDK retries disabled).
The API does not need to be configured to run `interactive` or `label_pred`.

#### Gemini / Vertex AI (existing configuration remains supported)

The example uses the upstream configured `gemini-2.5-flash-lite`, temperature
0.01, max output 10,000; actual availability depends on your Vertex AI account.
The upstream `vertexai.generative_models` SDK was
[retired in June 2026](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk).
Our adapter preserves its judge prompts/settings but uses the replacement
[Google Gen AI SDK](https://github.com/googleapis/python-genai/tree/v1.55.0)
over Vertex AI. If the configured model is unavailable, scoring remains pending;
generation is still reusable. Record any judge-model change for every comparison arm.

```bash
python -m pip install -r real_quant/full_precision/requirements-recif-judge.txt
```

Create a private config, e.g. `artifacts/config/recif_judge.json`, based on
`recif_judge.example.json`. Set `project`, `location`, and `model_name`, and use
Google Application Default Credentials or add `credentials_path` to your local
service-account JSON. Do not commit credentials. BERTScore may download
`bert-base-chinese`; `BERTSCORE_MODEL` may point to a local copy instead.

The following **sends generated captions and reference texts to Google Vertex
AI and may incur charges**. It does not send the recommendation histories.

```bash
GPU=5 TASKS=item_understand STAGE=evaluate ALLOW_JUDGE=1 \
  JUDGE_CONFIG=artifacts/config/recif_judge.json \
  bash scripts/real_quant/run_1p7b_recif_aux_baselines.sh
```

To perform all three complete evaluations in one run, omit `TASKS` and `STAGE`
from that command.

#### Shared scoring details

BERTScore may download `bert-base-chinese` for either judge provider;
`BERTSCORE_MODEL` can point to a local copy. Successful judge substeps are cached; API failures, invalid
JSON, or missing BERTScore abort scoring instead of producing partial/zero
metrics. As in upstream, valid empty WIP extractions are excluded from the
macro average; `wip_num_samples`, `wip_excluded_empty_samples`, and
`wip_coverage` expose the denominator. Always inspect coverage when comparing
models. Do not interpret “generation complete” as “all metrics complete”.
Reference WIPs are shared across comparison arms under
`artifacts/results/recif/wip_reference_cache`, keyed by reference text and judge
configuration, to avoid independently re-judging the same reference for each
model. Run scoring arms serially if they use this shared cache.

Tests (no GPU, model downloads, or judge calls):

```bash
python -m pytest -q tests/test_recif_baseline.py tests/test_recif_deepseek.py
```
