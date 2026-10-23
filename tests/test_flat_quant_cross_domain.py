"""Cross-domain evaluation is load-only and never reads target calibration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flat_quant import cross_domain_eval as cross


SCRIPT = cross.repo_root() / "scripts/flat_quant/run_1p7b_w8a8_pc_ad_to_product_video_cuda23.sh"
MATRIX_SCRIPT = cross.repo_root() / "scripts/flat_quant/run_1p7b_w8a8_pc_cross_domain_matrix_cuda67.sh"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_evaluation(plan, task, *, transferred=False, value=0.1, source="ad"):
    output = plan.transferred_eval(task, source) if transferred else plan.reference_eval(task)
    cfg = {**cross.COMMON, "task": task, "weight_quant_format": "fp8_e4m3fn",
           "eval_sample_size": "full", "eval_offset": 0, "split": "test",
           "eval_merged": True, "calibration_only": False, "eval_data_dir": str(plan.data),
           "flat_load_checkpoint_dir": str(plan.checkpoint(source if transferred else task))}
    if transferred:
        cfg.update(calibration_data_loaded=False, calib_split=None,
                   flat_finetune_checkpoint_dir=None, flat_prefix_checkpoint_dir=None)
    write_json(cross.config_path(output, task), cfg)
    write_json(output / "eval_results.json", {"1.7B": {task: {"test": {
        "total_samples": cross.SAMPLE_COUNTS[task], **{m: value for m in cross.METRICS},
    }}}})


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for key in ("PAIRS", "AD_RESULTS_ROOT"):
        monkeypatch.delenv(key, raising=False)
    for key, val in {
        "OOR_QUANT_ARTIFACTS": str(tmp_path / "artifacts"), "RUN_ROOT": str(tmp_path / "output"),
        "MODEL_PATH": str(tmp_path / "1.7B"), "DATA_DIR": str(tmp_path / "data"),
        "PYTHON_BIN": sys.executable, "GPUS": "2,3", "DRY_RUN": "1",
        "EVAL_OVERWRITE": "0", "TARGETS": "product video",
    }.items():
        monkeypatch.setenv(key, val)
    p = cross.Plan.from_env()
    write_json(p.model / "config.json", {"num_hidden_layers": 28, "hidden_size": 2048})
    p.data.mkdir()
    for task in cross.DOMAINS:
        base = p.artifacts / "results/flat_quant/official_fake_quant" / f"1p7b_{task}_w8a8_pc_prefix128_final512_heldout512_epoch15/final_train512_heldout512/1.7B/{task}/flatquant_calibration"
        cfg = {**cross.COMMON, **cross.ALIGNMENT, "task": task,
               "weight_quant_format": "fp8_e4m3fn", "flat_lfq_loss_weight": 1.0,
               "flat_lfq_boundary_loss_weight": 0.3, "calib_data_dir": str(p.data),
               "flat_finetune_checkpoint_dir": str(base), "baseline_summaries": {"27": {
                   "best_epoch": None, "trainable_transform_parameters": 91120,
                   "trainable_lwc_parameters": 40960, "trainable_lac_parameters": 8}}}
        write_json(cross.config_path(p.reference_training(task), task), cfg)
        write_evaluation(p, task)
        p.checkpoint(task).mkdir()
        for layer in range(28):
            (p.checkpoint(task) / f"layer_{layer:02d}.pt").write_bytes(b"fixture-only")
    return p


def launch(*args, **env):
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cross.repo_root(),
                          env={**os.environ, **env}, capture_output=True, text=True, timeout=60)


def commands(output):
    result = []
    for line in output.splitlines():
        if " -m " in line and "flat_quant.run_m1_onerec_ad" in line:
            words = shlex.split(line)
            result.append(words[words.index("flat_quant.run_m1_onerec_ad") + 1:])
    return result


def value(args, flag):
    return args[args.index(flag) + 1]


@pytest.mark.parametrize("gpus,expected", [(None, "6,7"), ("2,3", "2,3")])
def test_launcher_gpu_default_and_override(plan, monkeypatch, gpus, expected):
    if gpus is None:
        monkeypatch.delenv("GPUS", raising=False)
    else:
        monkeypatch.setenv("GPUS", gpus)
    result = launch()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"GPUs={expected}" in result.stdout
    assert len(commands(result.stdout)) == 4
    for gpu in expected.split(","):
        assert f"CUDA_VISIBLE_DEVICES={gpu} " in result.stdout
    assert not plan.output.exists()


def test_exactly_two_serial_evaluations_with_ad_weights(plan, monkeypatch):
    result = launch(SOURCE_CHECKPOINT_DIR="/wrong", RUN_EVAL="0", TASK="ad")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = commands(result.stdout)
    assert len(calls) == 4
    assert [value(a, "--task") for a in calls] == ["product", "product", "video", "video"]
    assert not plan.output.exists()
    from flat_quant.run_m1_onerec_ad import parse_args, build_flatquant_config
    for args in calls:
        assert value(args, "--flat_load_checkpoint_dir") == str(plan.checkpoint())
        assert value(args, "--weight_quant_format") == "fp8_e4m3fn"
        assert value(args, "--activation_quant_format") == "fp8_e4m3fn"
        assert value(args, "--weight_group_size") == "0"
        assert value(args, "--eval_sample_size") == "full"
        for flag in ("--calibration_only", "--flat_finetune_checkpoint_dir", "--flat_prefix_checkpoint_dir",
                     "--omni_final_objective", "--act_quant_mode", "--flat_epochs"):
            assert flag not in args
        monkeypatch.setattr(sys, "argv", ["cross-eval", *args])
        parsed = parse_args()
        build_flatquant_config(parsed).validate()
        assert parsed.seed == 42
        assert parsed.num_beams == parsed.num_return_sequences == 32
        assert parsed.max_new_tokens == 3


def test_summary_includes_ad_reuse_and_signed_percentage_point_deltas(plan):
    write_evaluation(plan, "product", transferred=True, value=0.09)
    plan.dry_run = False
    rows = cross.collect(plan)
    assert [r["test_domain"] for r in rows] == list(cross.DOMAINS)
    assert rows[0]["status"] == "reused" and rows[0]["delta_recall@32_pp"] == 0
    assert rows[1]["ad_calibrated_recall@32"] == pytest.approx(9)
    assert rows[1]["delta_recall@32_pp"] == pytest.approx(-1)
    assert rows[2]["ad_calibrated_recall@32"] is None
    assert "-1.000" in (plan.output / "cross_domain.csv").read_text()
    payload = cross.read_json(plan.output / "cross_domain.json")
    assert payload["source_checkpoint"] == str(plan.checkpoint())


def test_complete_results_skip_all_generation(plan):
    for task in plan.targets:
        write_evaluation(plan, task, transferred=True)
    result = launch()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[skip]") == 2
    assert not commands(result.stdout)
    assert not (plan.output / "cross_domain.csv").exists()


def test_partial_evaluation_requires_explicit_rerun(plan):
    plan.transferred_eval("video").mkdir(parents=True)
    result = launch()
    assert result.returncode != 0 and "EVAL_OVERWRITE=1" in result.stderr
    assert not commands(result.stdout)
    result = launch(EVAL_OVERWRITE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert all("--overwrite" in a for a in commands(result.stdout))
    assert not (plan.output / "cross_domain.csv").exists()
    result = launch("--collect-only")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key,wrong", [("flat_load_checkpoint_dir", "/wrong"),
    ("calibration_data_loaded", True), ("weight_group_size", 128)])
def test_wrong_cached_protocol_rejected(plan, key, wrong):
    write_evaluation(plan, "product", transferred=True)
    path = cross.config_path(plan.transferred_eval("product"), "product")
    cfg = cross.read_json(path)
    cfg[key] = wrong
    write_json(path, cfg)
    with pytest.raises(ValueError):
        cross.preflight(plan)


@pytest.mark.parametrize("env", [{"GPUS": "2,2"}, {"GPUS": "2,3,"},
    {"TARGETS": "ad"}, {"TARGETS": "product product"}])
def test_invalid_settings_fail_before_work(plan, env):
    result = launch(**env)
    assert result.returncode != 0
    assert not plan.output.exists()


@pytest.mark.parametrize("task", ("ad", "product", "video"))
def test_load_only_main_never_reads_calibration_or_trains(tmp_path, monkeypatch, task):
    """Exercise the real main control flow with mocked model I/O, not a CUDA run."""
    import torch
    from flat_quant import run_m1_onerec_ad as runner

    checkpoint = tmp_path / "ad_checkpoint"
    output = tmp_path / "output"
    argv = ["runner", "--task", task, "--mode", "flatquant_core", "--model_path", str(tmp_path / "1.7B"),
            "--output_dir", str(output), "--data_dir", str(tmp_path / "test_only_data"),
            "--device", "cpu", "--flat_load_checkpoint_dir", str(checkpoint),
            "--weight_quant_format", "fp8_e4m3fn", "--activation_quant_format", "fp8_e4m3fn",
            "--weight_quant_scheme", "symmetric", "--weight_group_size", "0", "--omni_lwc"]
    monkeypatch.setattr(sys, "argv", argv)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token="eos")
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(runner, "load_hf_model", lambda args: object())
    monkeypatch.setattr(runner, "resolve_input_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(runner, "get_transformer_layers", lambda model: [object()])
    monkeypatch.setattr(runner, "get_task_config", lambda task: {"generation_config": {}})

    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration or training was reached during load-only evaluation")

    for name in ("default_calib_split", "build_model_batches", "build_lfq_sid_slot_batches", "apply_flatquant_core_layers"):
        monkeypatch.setattr(runner, name, forbidden)
    restored = []

    def restore(**kwargs):
        restored.append(kwargs["checkpoint_dir"])
        return {}

    monkeypatch.setattr(runner, "restore_flatquant_core_layers_from_checkpoints", restore)
    reads = []

    def load_data(**kwargs):
        reads.append((kwargs["task_name"], kwargs["split"]))
        assert kwargs["split"] == "test"
        return {"sample_0": {"prompt": "test prompt", "ground_truth": "sid"}}

    monkeypatch.setattr(runner, "load_task_data", load_data)
    monkeypatch.setattr(runner, "generate_one", lambda **kwargs: ["sid"])
    runner.main()
    assert reads == [(task, "test")]
    assert restored == [checkpoint]
    cfg = cross.read_json(cross.config_path(output, task))
    assert cfg["calibration_data_loaded"] is False and cfg["calib_split"] is None
    assert cfg["flat_load_checkpoint_dir"] == str(checkpoint)


@pytest.fixture
def matrix_plan(plan, monkeypatch):
    for task in ("product", "video"):
        write_evaluation(plan, task, transferred=True, value=0.09)
    monkeypatch.setenv("AD_RESULTS_ROOT", str(plan.output))
    monkeypatch.setenv("RUN_ROOT", str(plan.output.with_name("matrix")))
    monkeypatch.delenv("GPUS", raising=False)
    return cross.Plan.from_env(matrix=True)


def launch_matrix(*args, **env):
    return subprocess.run(["bash", str(MATRIX_SCRIPT), *args], cwd=cross.repo_root(),
                          env={**os.environ, **env}, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("gpus", [None, "0,1"])
def test_matrix_launches_exactly_four_transfers_with_correct_checkpoints(matrix_plan, monkeypatch, gpus):
    env = {} if gpus is None else {"GPUS": gpus}
    result = launch_matrix(SOURCE_CHECKPOINT_DIR="/wrong", TASK="product", TARGETS="video", RUN_EVAL="0", **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"GPUs={gpus or '4,5'}" in result.stdout
    calls = commands(result.stdout)
    pairs = matrix_plan.requested_pairs()
    assert pairs == (("product", "ad"), ("product", "video"), ("video", "ad"), ("video", "product"))
    assert len(calls) == 8
    from flat_quant.run_m1_onerec_ad import parse_args, build_flatquant_config
    for i, args in enumerate(calls):
        source, task = pairs[i // 2]
        assert value(args, "--task") == task
        assert value(args, "--flat_load_checkpoint_dir") == str(matrix_plan.checkpoint(source))
        assert value(args, "--output_dir") == str(matrix_plan.transferred_eval(task, source))
        assert value(args, "--eval_shard_id") == str(i % 2)
        assert value(args, "--eval_num_shards") == "2"
        assert value(args, "--eval_sample_size") == "full"
        assert "--overwrite" not in args
        for flag in ("--calibration_only", "--flat_finetune_checkpoint_dir", "--flat_prefix_checkpoint_dir",
                     "--omni_final_objective", "--act_quant_mode", "--flat_epochs"):
            assert flag not in args
        monkeypatch.setattr(sys, "argv", ["matrix-eval", *args])
        parsed = parse_args()
        build_flatquant_config(parsed).validate()
        assert parsed.weight_quant_format == parsed.activation_quant_format == "fp8_e4m3fn"
        assert parsed.weight_group_size == 0 and parsed.seed == 42
        assert parsed.num_beams == parsed.num_return_sequences == 32 and parsed.max_new_tokens == 3
    assert not matrix_plan.output.exists()


def test_matrix_collection_uses_target_diagonal_and_preserves_references(matrix_plan):
    write_evaluation(matrix_plan, "product", value=0.2)
    write_evaluation(matrix_plan, "ad", source="product", transferred=True, value=0.08)
    write_evaluation(matrix_plan, "product", source="video", transferred=True, value=0.18)
    before = {p: p.read_bytes() for root in (matrix_plan.ad_output, matrix_plan.artifacts)
              for p in root.rglob("*.json")}
    matrix_plan.dry_run = False
    rows = cross.collect(matrix_plan)
    assert len(rows) == 9
    lookup = {(r["calibration_domain"], r["test_domain"]): r for r in rows}
    assert [r["status"] for r in rows].count("reused") == 5
    assert lookup["ad", "product"]["retention_recall@32_pct"] == pytest.approx(45)
    assert lookup["product", "ad"]["retention_recall@32_pct"] == pytest.approx(80)
    assert lookup["video", "product"]["retention_recall@32_pct"] == pytest.approx(90)
    assert lookup["product", "ad"]["delta_recall@32_pp"] == pytest.approx(-2)
    assert lookup["product", "video"]["recall@32"] is None
    for task in cross.DOMAINS:
        assert lookup[task, task]["retention_recall@32_pct"] == pytest.approx(100)
    payload = cross.read_json(matrix_plan.output / "cross_domain_matrix.json")
    assert payload["matrices"]["recall@32"] == [[10, 9, 9], [8, 20, None], [None, 18, 10]]
    assert payload["retention_matrices_pct"]["recall@32"][2][1] == pytest.approx(90)
    assert len((matrix_plan.output / "cross_domain_matrix.csv").read_text().splitlines()) == 10
    assert all(p.read_bytes() == data for p, data in before.items())


def test_matrix_all_cached_results_skip_without_writing(matrix_plan):
    for source, task in matrix_plan.requested_pairs():
        write_evaluation(matrix_plan, task, source=source, transferred=True)
    before = {p: p.read_bytes() for root in (matrix_plan.output, matrix_plan.ad_output, matrix_plan.artifacts)
              for p in root.rglob("*.json")}
    result = launch_matrix()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[skip]") == 4 and not commands(result.stdout)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not (matrix_plan.output / "cross_domain_matrix.json").exists()


def test_matrix_partial_run_and_overwrite_do_not_touch_five_references(matrix_plan):
    matrix_plan.transferred_eval("ad", "product").mkdir(parents=True)
    result = launch_matrix()
    assert result.returncode != 0 and "EVAL_OVERWRITE=1" in result.stderr
    assert not commands(result.stdout)
    before = {p: p.read_bytes() for root in (matrix_plan.ad_output, matrix_plan.artifacts)
              for p in root.rglob("*.json")}
    result = launch_matrix(EVAL_OVERWRITE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = commands(result.stdout)
    assert len(calls) == 8 and all("--overwrite" in a for a in calls)
    for args in calls:
        assert value(args, "--flat_load_checkpoint_dir") != str(matrix_plan.checkpoint("ad"))
        assert Path(value(args, "--output_dir")).parent == matrix_plan.output
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("key,wrong", [("flat_load_checkpoint_dir", "/wrong"),
    ("calibration_data_loaded", True), ("weight_group_size", 128)])
def test_matrix_wrong_new_result_protocol_rejected(matrix_plan, key, wrong):
    write_evaluation(matrix_plan, "ad", source="product", transferred=True)
    path = cross.config_path(matrix_plan.transferred_eval("ad", "product"), "ad")
    cfg = cross.read_json(path)
    cfg[key] = wrong
    write_json(path, cfg)
    result = launch_matrix()
    assert result.returncode != 0 and not commands(result.stdout)


def test_matrix_reused_ad_results_validated_even_when_overwriting(matrix_plan):
    path = cross.config_path(matrix_plan.transferred_eval("video", "ad"), "video")
    cfg = cross.read_json(path)
    cfg["flat_load_checkpoint_dir"] = str(matrix_plan.checkpoint("video"))
    write_json(path, cfg)
    result = launch_matrix(EVAL_OVERWRITE="1")
    assert result.returncode != 0 and not commands(result.stdout)


@pytest.mark.parametrize("source", ("product", "video"))
def test_matrix_missing_source_checkpoint_fails_before_work(matrix_plan, source):
    (matrix_plan.checkpoint(source) / "layer_27.pt").unlink()
    result = launch_matrix()
    assert result.returncode != 0 and f"Incomplete {source} checkpoint" in result.stderr
    assert not commands(result.stdout) and not matrix_plan.output.exists()


@pytest.mark.parametrize("target", ("ad_root", "inside_ad_root", "ancestor", "reference", "model", "data"))
def test_matrix_output_cannot_overlap_protected_roots(matrix_plan, target):
    paths = {"ad_root": matrix_plan.ad_output, "inside_ad_root": matrix_plan.ad_output / "nested",
             "ancestor": matrix_plan.ad_output.parent, "reference": matrix_plan.reference_training("video"),
             "model": matrix_plan.model, "data": matrix_plan.data}
    result = launch_matrix(RUN_ROOT=str(paths[target]), EVAL_OVERWRITE="1")
    assert result.returncode != 0 and "overlaps a protected reference" in result.stderr
    assert not commands(result.stdout)


@pytest.mark.parametrize("pairs", ("ad:video", "product:product", "product:ad product:ad", " "))
def test_matrix_invalid_pair_selection_fails_before_work(matrix_plan, pairs):
    result = launch_matrix(PAIRS=pairs)
    assert result.returncode != 0 and not commands(result.stdout)
    assert not matrix_plan.output.exists()


def test_matrix_subset_and_collect_only(matrix_plan):
    result = launch_matrix(PAIRS="video:product", GPUS="4,5")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPUs=4,5" in result.stdout
    calls = commands(result.stdout)
    assert len(calls) == 2
    assert all(value(a, "--task") == "product" for a in calls)
    assert all(value(a, "--flat_load_checkpoint_dir") == str(matrix_plan.checkpoint("video")) for a in calls)
    matrix_plan.transferred_eval("video", "product").mkdir(parents=True)
    result = launch_matrix("--collect-only", DRY_RUN="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not commands(result.stdout)
    payload = cross.read_json(matrix_plan.output / "cross_domain_matrix.json")
    assert len(payload["rows"]) == 9
    assert sum(r["status"] == "pending" for r in payload["rows"]) == 4
