from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from fake_quant.evaluate_lfq_boundary_diagnostics import (
    compute_slot_diagnostics,
    paired_bootstrap_difference,
    verify_single_checkpoint_set,
    verify_single_run_protocol,
    verify_run_protocols,
)


class LFQBoundaryDiagnosticsTest(unittest.TestCase):
    def test_identical_logits_are_perfectly_retained(self) -> None:
        teacher = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        metrics = compute_slot_diagnostics(
            teacher,
            teacher.clone(),
            topk=2,
            negative_count=2,
            tie_threshold=1e-2,
            gap_scale=1.0,
        )
        self.assertAlmostEqual(metrics["kl"], 0.0, places=7)
        self.assertEqual(metrics["top1_agreement_rate"], 1.0)
        self.assertEqual(metrics["teacher_top1_in_student_top5_rate"], 1.0)
        self.assertEqual(metrics["teacher_top1_in_student_top10_rate"], 1.0)
        self.assertEqual(metrics["teacher_top1_in_student_topk_rate"], 1.0)
        self.assertEqual(metrics["teacher_top1_student_rank"], 1.0)
        self.assertEqual(metrics["student_top1_teacher_rank"], 1.0)
        self.assertEqual(metrics["top5_retention"], 1.0)
        self.assertEqual(metrics["top10_retention"], 1.0)
        self.assertEqual(metrics["topk_retention"], 1.0)
        self.assertEqual(metrics["topk_jaccard"], 1.0)
        self.assertEqual(metrics["intruder_rank_k1_to_kplusn_rate"], 0.0)
        self.assertEqual(metrics["intruder_below_rank_kplusn_rate"], 0.0)
        self.assertEqual(metrics["boundary_pair_violation_rate"], 0.0)
        self.assertEqual(metrics["boundary_gap_mae"], 0.0)

    def test_topk_intruders_are_split_by_teacher_rank(self) -> None:
        teacher = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        student = torch.tensor([0.0, 0.0, 10.0, 0.0, 9.0, -1.0])
        metrics = compute_slot_diagnostics(
            teacher,
            student,
            topk=2,
            negative_count=2,
            tie_threshold=1e-2,
            gap_scale=1.0,
        )
        self.assertEqual(metrics["topk_retention"], 0.0)
        self.assertEqual(metrics["intruder_rank_k1_to_kplusn_rate"], 0.5)
        self.assertEqual(metrics["intruder_below_rank_kplusn_rate"], 0.5)
        self.assertEqual(metrics["boundary_pair_violation_rate"], 1.0)

    def test_head_metrics_distinguish_set_retention_from_top1_order(self) -> None:
        teacher = torch.tensor(
            [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        )
        student = torch.tensor(
            [8.0, 11.0, 10.0, 9.0, 12.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        )
        metrics = compute_slot_diagnostics(
            teacher,
            student,
            topk=5,
            negative_count=5,
            tie_threshold=1e-2,
            gap_scale=1.0,
        )
        self.assertEqual(metrics["top5_retention"], 1.0)
        self.assertEqual(metrics["top10_retention"], 1.0)
        self.assertEqual(metrics["topk_retention"], 1.0)
        self.assertEqual(metrics["top1_agreement_rate"], 0.0)
        self.assertEqual(metrics["teacher_top1_in_student_top5_rate"], 1.0)
        self.assertEqual(metrics["teacher_top1_student_rank"], 5.0)
        self.assertEqual(metrics["student_top1_teacher_rank"], 5.0)

    def test_weighted_rates_normalize_small_nonzero_weight(self) -> None:
        metrics = compute_slot_diagnostics(
            torch.tensor([0.1, 0.0]),
            torch.tensor([0.0, 0.1]),
            topk=1,
            negative_count=1,
            tie_threshold=0.0,
            gap_scale=1.0,
        )
        self.assertAlmostEqual(
            metrics["boundary_pair_weighted_violation_rate"],
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            metrics["boundary_gap_weighted_mae"],
            0.2,
            places=6,
        )

    def test_teacher_ties_are_excluded_from_boundary_pairs(self) -> None:
        teacher = torch.tensor([4.0, 3.0, 2.99, 1.0])
        metrics = compute_slot_diagnostics(
            teacher,
            teacher.clone(),
            topk=2,
            negative_count=2,
            tie_threshold=0.02,
            gap_scale=1.0,
        )
        self.assertAlmostEqual(
            metrics["eligible_boundary_pair_fraction"],
            0.75,
            places=7,
        )
        self.assertEqual(metrics["boundary_pair_violation_rate"], 0.0)

    def test_run_protocol_verifier_rejects_seed_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / "benchmark_data"
            data_dir.mkdir()
            prefix_dir = root / "shared_prefix" / "omniquant_calibration"
            prefix_dir.mkdir(parents=True)
            common = {
                "task": "ad",
                "split": "test",
                "calib_split": "ad_calib",
                "calib_offset": 0,
                "calibration_only": True,
                "dtype": "bfloat16",
                "seed": 42,
                "layers": [0, 1],
                "weight_quant_format": "fp4_e2m1",
                "activation_quant_format": "fp8_e4m3fn",
                "weight_quant_scheme": "symmetric",
                "weight_group_size": 0,
                "act_quant_mode": "shared_input",
                "omni_lwc": True,
                "omni_let_mode": "none",
                "omni_let_init": "smoothquant",
                "omni_epochs": 20,
                "omni_epoch_eval_interval": 0,
                "omni_lwc_lr": 1e-2,
                "omni_let_lr": 5e-3,
                "omni_weight_decay": 0.0,
                "omni_init_lwc_logit": 4.0,
                "omni_max_grad_norm": None,
                "omni_prefix_checkpoint_dir": str(prefix_dir),
                "data_dir": str(data_dir),
                "omni_lfq_slot_weights": [1.0, 1.0, 1.0],
                "omni_lfq_loss_weight": 1.0,
                "omni_lfq_boundary_topk": 32,
                "omni_lfq_boundary_negative_count": 32,
                "omni_lfq_boundary_tie_threshold": 0.01,
                "omni_lfq_boundary_gap_scale": 1.0,
            }
            method_values = {
                "mse": (512, 0, 0, "mse"),
                "abc": (1024, 512, 512, "lfq_ce"),
                "boundary": (1024, 512, 512, "lfq_ce"),
            }
            checkpoint_dirs: dict[str, Path] = {}
            config_paths: dict[str, Path] = {}
            run_configs: dict[str, dict[str, object]] = {}
            for method, values in method_values.items():
                checkpoint_dir = root / method / "omniquant_calibration"
                checkpoint_dir.mkdir(parents=True)
                checkpoint_dirs[method] = checkpoint_dir
                config_paths[method] = checkpoint_dir.parent / "omniquant_config.json"
                config = {
                    **common,
                    "calib_sample_size": str(values[0]),
                    "omni_train_sample_size": values[1],
                    "omni_validation_sample_size": values[2],
                    "omni_final_objective": values[3],
                }
                run_configs[method] = config
                config_paths[method].write_text(json.dumps(config), encoding="utf-8")

            prefix_config = {
                **common,
                "layers": [0],
                "calib_sample_size": "128",
                "omni_train_sample_size": 0,
                "omni_validation_sample_size": 0,
                "omni_final_objective": "mse",
            }
            (prefix_dir.parent / "omniquant_config.json").write_text(
                json.dumps(prefix_config), encoding="utf-8"
            )

            verified = verify_run_protocols(
                checkpoint_dirs,
                layer_count=2,
                task="ad",
                data_dir=str(data_dir),
                calib_split="ad_calib",
                calib_sample_size=1024,
                prefix_calib_sample_size=128,
                train_sample_size=512,
                heldout_sample_size=512,
                seed=42,
                topk=32,
                negative_count=32,
                tie_threshold=0.01,
                gap_scale=1.0,
            )
            self.assertEqual(verified["common"]["seed"], 42)

            for method, config in run_configs.items():
                config["omni_let_mode"] = "learned"
                config["omni_let_init"] = "ones"
                config_paths[method].write_text(json.dumps(config), encoding="utf-8")
            prefix_config["omni_let_mode"] = "learned"
            prefix_config["omni_let_init"] = "ones"
            (prefix_dir.parent / "omniquant_config.json").write_text(
                json.dumps(prefix_config), encoding="utf-8"
            )
            learned_let = verify_run_protocols(
                checkpoint_dirs,
                layer_count=2,
                task="ad",
                data_dir=str(data_dir),
                calib_split="ad_calib",
                calib_sample_size=1024,
                prefix_calib_sample_size=128,
                train_sample_size=512,
                heldout_sample_size=512,
                seed=42,
                topk=32,
                negative_count=32,
                tie_threshold=0.01,
                gap_scale=1.0,
                expected_omni_let_mode="learned",
                expected_omni_let_init="ones",
            )
            self.assertEqual(learned_let["common"]["omni_let_mode"], "learned")
            self.assertEqual(learned_let["common"]["omni_let_init"], "ones")

            for method, config in run_configs.items():
                config["omni_let_mode"] = "none"
                config["omni_let_init"] = "smoothquant"
                config_paths[method].write_text(json.dumps(config), encoding="utf-8")
            prefix_config["omni_let_mode"] = "none"
            prefix_config["omni_let_init"] = "smoothquant"
            (prefix_dir.parent / "omniquant_config.json").write_text(
                json.dumps(prefix_config), encoding="utf-8"
            )

            run_configs["boundary"]["omni_lfq_loss_weight"] = 0.0
            config_paths["boundary"].write_text(
                json.dumps(run_configs["boundary"]),
                encoding="utf-8",
            )
            boundary_only = verify_run_protocols(
                checkpoint_dirs,
                layer_count=2,
                task="ad",
                data_dir=str(data_dir),
                calib_split="ad_calib",
                calib_sample_size=1024,
                prefix_calib_sample_size=128,
                train_sample_size=512,
                heldout_sample_size=512,
                seed=42,
                topk=32,
                negative_count=32,
                tie_threshold=0.01,
                gap_scale=1.0,
                expected_boundary_lfq_loss_weight=0.0,
            )
            self.assertEqual(
                boundary_only["per_method_split"]["boundary"]["lfq_loss_weight"],
                0.0,
            )
            run_configs["boundary"]["omni_lfq_loss_weight"] = 1.0
            run_configs["boundary"]["seed"] = 99
            config_paths["boundary"].write_text(
                json.dumps(run_configs["boundary"]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "seed"):
                verify_run_protocols(
                    checkpoint_dirs,
                    layer_count=2,
                    task="ad",
                    data_dir=str(data_dir),
                    calib_split="ad_calib",
                    calib_sample_size=1024,
                    prefix_calib_sample_size=128,
                    train_sample_size=512,
                    heldout_sample_size=512,
                    seed=42,
                    topk=32,
                    negative_count=32,
                    tie_threshold=0.01,
                    gap_scale=1.0,
                )

    def test_single_groupwise_checkpoint_protocol_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / "benchmark_data"
            data_dir.mkdir()
            prefix_dir = root / "shared_prefix" / "omniquant_calibration"
            checkpoint_dir = root / "boundary" / "omniquant_calibration"
            prefix_dir.mkdir(parents=True)
            checkpoint_dir.mkdir(parents=True)

            prefix_checkpoint = prefix_dir / "layer_00.pt"
            torch.save({"layer_idx": 0}, prefix_checkpoint)
            torch.save(
                {
                    "layer_idx": 0,
                    "source_checkpoint": str(prefix_checkpoint),
                },
                checkpoint_dir / "layer_00.pt",
            )
            torch.save(
                {
                    "layer_idx": 1,
                    "config": {
                        "weight_quant_format": "fp4_e2m1",
                        "activation_quant_format": "fp8_e4m3fn",
                        "weight_quant_scheme": "symmetric",
                        "weight_group_size": 128,
                        "use_lwc": True,
                        "use_let": False,
                        "learn_let": False,
                        "final_objective": "lfq_ce",
                        "lfq_loss_weight": 1.0,
                        "lfq_boundary_loss_weight": 0.3,
                        "lfq_boundary_topk": 32,
                        "lfq_boundary_negative_count": 32,
                        "lfq_boundary_tie_threshold": 0.01,
                        "lfq_boundary_gap_scale": 1.0,
                    },
                },
                checkpoint_dir / "layer_01.pt",
            )

            common = {
                "task": "ad",
                "split": "test",
                "calib_split": "ad_calib",
                "calib_offset": 0,
                "calibration_only": True,
                "dtype": "bfloat16",
                "seed": 42,
                "weight_quant_format": "fp4_e2m1",
                "activation_quant_format": "fp8_e4m3fn",
                "weight_quant_scheme": "symmetric",
                "weight_group_size": 128,
                "act_quant_mode": "shared_input",
                "omni_lwc": True,
                "omni_let_mode": "none",
                "omni_let_init": "smoothquant",
                "omni_lwc_lr": 1e-2,
                "omni_let_lr": 5e-3,
                "omni_weight_decay": 0.0,
                "omni_init_lwc_logit": 4.0,
                "omni_max_grad_norm": None,
            }
            prefix_config = {
                **common,
                "layers": [0],
                "calib_sample_size": "128",
                "omni_train_sample_size": 0,
                "omni_validation_sample_size": 0,
                "omni_final_objective": "mse",
            }
            (prefix_dir.parent / "omniquant_config.json").write_text(
                json.dumps(prefix_config), encoding="utf-8"
            )
            run_config = {
                **common,
                "layers": [0, 1],
                "data_dir": str(data_dir),
                "calib_sample_size": "1024",
                "omni_train_sample_size": 512,
                "omni_validation_sample_size": 512,
                "omni_prefix_checkpoint_dir": str(prefix_dir),
                "omni_final_objective": "lfq_ce",
                "omni_lfq_loss_weight": 1.0,
                "omni_lfq_boundary_loss_weight": 0.3,
                "omni_lfq_boundary_topk": 32,
                "omni_lfq_boundary_negative_count": 32,
                "omni_lfq_boundary_tie_threshold": 0.01,
                "omni_lfq_boundary_gap_scale": 1.0,
            }
            config_path = checkpoint_dir.parent / "omniquant_config.json"
            config_path.write_text(json.dumps(run_config), encoding="utf-8")

            checkpoint_config, recovered_prefix = verify_single_checkpoint_set(
                checkpoint_dir,
                layer_count=2,
                expected_lfq_loss_weight=1.0,
                expected_boundary_loss_weight=0.3,
            )
            protocol_kwargs = {
                "prefix_checkpoint_dir": recovered_prefix,
                "checkpoint_config": checkpoint_config,
                "layer_count": 2,
                "task": "ad",
                "data_dir": str(data_dir),
                "calib_split": "ad_calib",
                "calib_sample_size": 1024,
                "prefix_calib_sample_size": 128,
                "train_sample_size": 512,
                "heldout_sample_size": 512,
                "seed": 42,
                "topk": 32,
                "negative_count": 32,
                "tie_threshold": 0.01,
                "gap_scale": 1.0,
                "expected_lfq_loss_weight": 1.0,
                "expected_boundary_loss_weight": 0.3,
                "expected_omni_let_mode": "none",
                "expected_omni_let_init": "smoothquant",
            }
            verified = verify_single_run_protocol(
                checkpoint_dir,
                **protocol_kwargs,
            )
            self.assertEqual(verified["common"]["weight_group_size"], 128)
            self.assertEqual(
                verified["single_method_split"]["boundary_loss_weight"],
                0.3,
            )

            run_config["weight_group_size"] = 0
            config_path.write_text(json.dumps(run_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weight_group_size"):
                verify_single_run_protocol(
                    checkpoint_dir,
                    **protocol_kwargs,
                )

    def test_paired_bootstrap_preserves_constant_difference(self) -> None:
        result = paired_bootstrap_difference(
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            bootstrap_samples=100,
            seed=42,
        )
        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["ci95_low"], 1.0)
        self.assertEqual(result["ci95_high"], 1.0)


if __name__ == "__main__":
    unittest.main()
