# Benchmark support

`benchmarks/benchmark/` is the local OpenOneRec task dependency used by the
HuggingFace BF16 and real-FP8 runners. It provides:

- message-format data loading and the Qwen3 soft-switch chat template;
- recommendation SID/PID metrics for `ad`, `product`, and `video`;
- the non-recommendation task configs and evaluators retained for future work;
- the generic `Benchmark.evaluate_dev` compatibility interface.

The active runners are under `../real_quant/` and use this package through
`get_loader`, `get_task_config`, and `Benchmark.evaluate_dev`; do not add a
second generation runner here.

The package initializers and evaluator registry are lazy: importing a current
recommendation runner does not eagerly import every task evaluator. Older
checkpoint-QDQ, llmcompressor/vLLM, beam-shift, and calibration-export scripts
were removed because they target deleted experiment artifacts and are not part
of the real-FP8 PTQ workflow.
