"""Loss-ablation protocol, dry-run CLI, cache safety, and result collection."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from flat_quant import loss_ablation as ab


SCRIPT = ab.repo_root() / "scripts/flat_quant/run_1p7b_ad_w8a8_w4a8_pc_loss_ablation_cuda23.sh"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def training_config(plan, precision, variant):
    cfg = {**ab.COMMON, "weight_quant_format": ab.FORMATS[precision],
           "calib_data_dir": str(plan.data)}
    if variant == "mse_baseline":
        cfg.update(omni_final_objective="mse", flatquant_variant="official_svd_cayley_diag_lwc_lac_mse",
                   flat_train_sample_size=512, flat_validation_sample_size=512, flat_epochs=15)
    else:
        ce, boundary = ab.WEIGHTS[variant]
        cfg.update(ab.ALIGNMENT)
        cfg.update(flat_lfq_loss_weight=ce, flat_lfq_boundary_loss_weight=boundary,
                   flat_finetune_checkpoint_dir=str(plan.source(precision)),
                   baseline_summaries={"27": {"best_epoch": None,
                       "trainable_transform_parameters": 91120,
                       "trainable_lwc_parameters": 40960, "trainable_lac_parameters": 8}})
    return cfg


def write_evaluation(plan, precision, variant, sample_count=27677):
    directory = plan.evaluation(precision, variant)
    config = {**ab.COMMON, "weight_quant_format": ab.FORMATS[precision],
              "eval_sample_size": "full", "eval_offset": 0, "split": "test",
              "eval_merged": True, "eval_data_dir": str(plan.data),
              "flat_load_checkpoint_dir": str(plan.checkpoint(precision, variant))}
    write_json(ab.config_path(directory), config)
    write_json(directory / "eval_results.json", {"1.7B": {"ad": {"test": {
        "total_samples": sample_count, **dict(zip(ab.METRICS, (0.14, 0.21, 0.045, 0.07))),
    }}}})


@pytest.fixture
def plan(tmp_path, monkeypatch):
    for key, value in {
        "OOR_QUANT_ARTIFACTS": str(tmp_path / "artifacts"),
        "RUN_ROOT": str(tmp_path / "outputs"), "MODEL_PATH": str(tmp_path / "1.7B"),
        "DATA_DIR": str(tmp_path / "data"), "PYTHON_BIN": sys.executable,
        "GPUS": "2,3", "DRY_RUN": "1", "RUN_EVAL": "1", "OVERWRITE": "0",
        "EVAL_OVERWRITE": "0", "PRECISIONS": "w8a8 w4a8",
        "VARIANTS": "wo_boundary wo_distribution",
    }.items():
        monkeypatch.setenv(key, value)
    plan = ab.Plan.from_env()
    write_json(plan.model / "config.json", {"num_hidden_layers": 28, "hidden_size": 2048})
    plan.data.mkdir()
    for precision in plan.precisions:
        plan.source(precision).mkdir(parents=True)
        for layer in range(28):
            (plan.source(precision) / f"layer_{layer:02d}.pt").write_bytes(b"fixture-only")
        for variant in ("mse_baseline", "recquant"):
            write_json(ab.config_path(plan.training(precision, variant)), training_config(plan, precision, variant))
            write_evaluation(plan, precision, variant)
    return plan


def launch(**overrides):
    return subprocess.run(["bash", str(SCRIPT)], cwd=ab.repo_root(),
                          env={**os.environ, **overrides}, text=True, capture_output=True, timeout=60)


def commands(stdout):
    result = []
    for line in stdout.splitlines():
        if " -m " in line and "flat_quant.run_m1_onerec_ad" in line:
            words = shlex.split(line)
            result.append(words[words.index("flat_quant.run_m1_onerec_ad") + 1:])
    return result


def value(args, flag):
    return args[args.index(flag) + 1]


def test_default_gpu_assignment(plan, monkeypatch):
    monkeypatch.delenv("GPUS")
    default_plan = ab.Plan.from_env()
    assert default_plan.gpus == "2,3"
    env = ab.arm_environment(default_plan, "w8a8", "wo_boundary")
    assert env["GPUS"] == "2,3" and env["TRAIN_GPU"] == "2"


def test_four_arms_cli_matches_final_recquant(plan, monkeypatch):
    result = launch(LEARN_TRANSFORM="0", EPOCHS="99", SOURCE_CHECKPOINT_DIR="/wrong")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not plan.output.exists()
    all_commands = commands(result.stdout)
    train = [a for a in all_commands if "--calibration_only" in a]
    evaluate = [a for a in all_commands if "--eval_num_shards" in a]
    assert len(train) == 4 and len(evaluate) == 8
    from flat_quant.run_m1_onerec_ad import parse_args, build_flatquant_config
    for args in all_commands:
        assert "--act_quant_mode" not in args
        assert value(args, "--weight_group_size") == "0"
        monkeypatch.setattr(sys, "argv", ["flatquant", *args])
        parsed = parse_args()
        build_flatquant_config(parsed).validate()
        assert parsed.seed == 42 and parsed.num_beams == 32
    for args, (precision, variant) in zip(train, [(p,v) for p in plan.precisions for v in plan.variants]):
        assert value(args, "--flat_finetune_checkpoint_dir") == str(plan.source(precision))
        assert value(args, "--weight_quant_format") == ab.FORMATS[precision]
        assert value(args, "--flat_epochs") == "15"
        assert value(args, "--flat_train_sample_size") == "1024"
        assert value(args, "--flat_validation_sample_size") == "0"
        assert value(args, "--flat_epoch_eval_interval") == "0"
        assert "--flat_learn_transform" in args and "--flat_learn_lac" in args
        assert (float(value(args, "--omni_lfq_loss_weight")),
                float(value(args, "--omni_lfq_boundary_loss_weight"))) == ab.WEIGHTS[variant]
    for args in evaluate:
        assert "--omni_final_objective" not in args
        assert "--omni_lfq_loss_weight" not in args
        assert "--flat_load_checkpoint_dir" in args
        assert value(args, "--eval_sample_size") == "full"


@pytest.mark.parametrize("field,wrong", [("flat_learn_transform", False), ("flat_epochs", 20),
    ("flat_train_sample_size", 512), ("weight_group_size", 128), ("flat_lfq_loss_weight", 0.0)])
def test_reference_drift_rejected(plan, field, wrong):
    path = ab.config_path(plan.training("w4a8", "recquant"))
    cfg = ab.read_json(path)
    cfg[field] = wrong
    write_json(path, cfg)
    with pytest.raises(ValueError, match=field):
        ab.preflight(plan)
    assert not plan.output.exists()


def test_wrong_initialization_rejected(plan):
    path = ab.config_path(plan.training("w8a8", "recquant"))
    cfg = ab.read_json(path)
    cfg["flat_finetune_checkpoint_dir"] = str(plan.output / "mse1024")
    write_json(path, cfg)
    with pytest.raises(ValueError, match="expected"):
        ab.preflight(plan)


def test_collection_keeps_missing_values_and_provenance(plan):
    plan.dry_run = False
    rows = ab.collect(plan)
    assert len(rows) == 8
    assert sum(r["status"] == "pending" for r in rows) == 4
    assert rows[0]["pass@16"] == pytest.approx(14)
    assert rows[1]["pass@16"] is None
    payload = ab.read_json(plan.output / "loss_ablation.json")
    assert payload["metric_units"] == "percent"
    assert "not a matched" in payload["protocol_note"]
    assert payload["rows"][0]["result_path"].endswith("eval_results.json")
    assert "14.000" in (plan.output / "loss_ablation.csv").read_text()


def test_bad_sample_count_or_empty_metric_rejected(plan):
    write_evaluation(plan, "w8a8", "recquant", sample_count=10)
    with pytest.raises(ValueError, match="27677"):
        ab.collect(plan)


def test_completed_result_skips_even_without_arm_checkpoints(plan):
    for precision in plan.precisions:
        for variant in plan.variants:
            write_json(ab.config_path(plan.training(precision, variant)), training_config(plan, precision, variant))
            write_evaluation(plan, precision, variant)
    result = launch()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[skip]") == 4
    assert not commands(result.stdout)
    assert not (plan.output / "loss_ablation.json").exists()  # Dry run stays read-only.


def test_incomplete_training_is_not_silently_reused(plan):
    cp = plan.checkpoint("w8a8", "wo_boundary")
    cp.mkdir(parents=True)
    (cp / "layer_00.pt").write_bytes(b"partial")
    result = launch()
    assert result.returncode != 0
    assert "partial checkpoints" in result.stderr
    assert not commands(result.stdout)


def test_eval_only_overwrite_never_retrains(plan):
    for precision in plan.precisions:
        for variant in plan.variants:
            write_json(ab.config_path(plan.training(precision, variant)), training_config(plan, precision, variant))
            cp = plan.checkpoint(precision, variant)
            cp.mkdir()
            for i in range(28):
                (cp / f"layer_{i:02d}.pt").write_bytes(b"complete")
    result = launch(EVAL_OVERWRITE="1", GPUS="2,3")
    assert result.returncode == 0, result.stdout + result.stderr
    args = commands(result.stdout)
    assert len(args) == 8
    assert all("--overwrite" in a and "--calibration_only" not in a for a in args)
    assert "physical_gpu=6" not in result.stdout


@pytest.mark.parametrize("settings", [{"GPUS": "6,6"}, {"GPUS": "6,7,"},
    {"PRECISIONS": "w4a8 w4a8"}, {"VARIANTS": "recquant"},
    {"OVERWRITE": "1", "EVAL_OVERWRITE": "0"},
    {"OVERWRITE": "1", "EVAL_OVERWRITE": "1", "RUN_EVAL": "0"}])
def test_invalid_launch_settings(plan, settings):
    result = launch(**settings)
    assert result.returncode != 0
    assert not plan.output.exists()


@pytest.mark.parametrize("precision", ("w8a8", "w4a8"))
@pytest.mark.parametrize("variant", ("wo_boundary", "wo_distribution"))
def test_tiny_model_single_loss_continuation(tmp_path, precision, variant):
    """Exercise actual joint optimization from a saved MSE model, on CPU."""
    import copy
    import torch
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from flat_quant.flatquant import FlatQuantCoreConfig, apply_flatquant_core_layers
    from flat_quant.run_m1_onerec_ad import capture_layer_input_batches

    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=4, max_position_embeddings=32)
    original = Qwen3ForCausalLM(config).eval()
    batches = [{"input_ids": torch.randint(0, 32, (1, 6)),
                "attention_mask": torch.ones(1, 6, dtype=torch.long)}]
    common = dict(weight_quant_format=ab.FORMATS[precision],
                  activation_quant_format="fp8_e4m3fn", weight_group_size=0,
                  epochs=1, train_sample_size=1, validation_sample_size=0,
                  epoch_eval_interval=0, learn_transform=True, learn_lac=True,
                  transform_lr=1e-3, lwc_lr=1e-3, lac_lr=1e-3)
    source = tmp_path / "mse"
    apply_flatquant_core_layers(
        model=copy.deepcopy(original), model_batches=batches, layer_indices=[0, 1],
        config=FlatQuantCoreConfig(**common),
        capture_layer_input_batches=capture_layer_input_batches, checkpoint_dir=source,
    )
    ce, boundary = ab.WEIGHTS[variant]
    student = copy.deepcopy(original)
    destination = tmp_path / variant
    summaries = apply_flatquant_core_layers(
        model=student, model_batches=batches, layer_indices=[0, 1],
        config=FlatQuantCoreConfig(**common, final_objective="lfq_ce",
            lfq_loss_weight=ce, lfq_boundary_loss_weight=boundary,
            lfq_boundary_topk=2, lfq_boundary_negative_count=2),
        capture_layer_input_batches=capture_layer_input_batches,
        checkpoint_dir=destination, finetune_checkpoint_dir=source,
        lfq_token_ids={"a": tuple(range(8)), "b": tuple(range(8, 16)), "c": tuple(range(16, 24))},
    )
    final = summaries[1]
    assert final.objective == "lfq_ce" and final.best_epoch is None
    assert final.trainable_transform_parameters > 0
    assert final.trainable_lwc_parameters > 0 and final.trainable_lac_parameters > 0
    assert math.isfinite(final.final_loss)
    if variant == "wo_distribution":
        assert final.final_loss == pytest.approx(0.3 * final.final_lfq_boundary_loss, rel=1e-5)
    else:
        assert final.final_lfq_boundary_loss is None
    base_prefix = torch.load(source / "layer_00.pt", map_location="cpu", weights_only=True)
    prefix = torch.load(destination / "layer_00.pt", map_location="cpu", weights_only=True)
    saved_final = torch.load(destination / "layer_01.pt", map_location="cpu", weights_only=True)
    assert Path(prefix["source_checkpoint"]).resolve() == (source / "layer_00.pt").resolve()
    assert Path(saved_final["initialization_checkpoint"]).resolve() == (source / "layer_01.pt").resolve()
    for name, tensor in base_prefix["transform_state_dict"].items():
        torch.testing.assert_close(prefix["transform_state_dict"][name], tensor, rtol=0, atol=0)
    with torch.no_grad():
        assert torch.isfinite(student(**batches[0]).logits).all()
