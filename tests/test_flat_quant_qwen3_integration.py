from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from flat_quant.flatquant import (
    FlatQuantCoreConfig,
    apply_flatquant_core_layers,
    restore_flatquant_core_layers_from_checkpoints,
)
from flat_quant.run_m1_onerec_ad import capture_layer_input_batches


class FlatQuantQwen3IntegrationTest(unittest.TestCase):
    def test_shared_input_training_checkpoint_and_restore(self) -> None:
        torch.manual_seed(29)
        hf_config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
        )
        model = Qwen3ForCausalLM(hf_config).eval()
        original_state = copy.deepcopy(model.state_dict())
        batch = {
            "input_ids": torch.randint(0, hf_config.vocab_size, (1, 5)),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        heldout_batch = {
            "input_ids": torch.randint(0, hf_config.vocab_size, (1, 5)),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        config = FlatQuantCoreConfig(
            weight_quant_format="int4",
            activation_quant_format="int8",
            weight_quant_scheme="symmetric",
            weight_group_size=0,
            epochs=1,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=1,
            transform_lr=1e-3,
            lwc_lr=1e-3,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            summaries = apply_flatquant_core_layers(
                model=model,
                model_batches=[batch, heldout_batch],
                layer_indices=[0],
                config=config,
                capture_layer_input_batches=capture_layer_input_batches,
                act_quant_mode="shared_input",
                checkpoint_dir=checkpoint_dir,
            )
            checkpoint_path = checkpoint_dir / "layer_00.pt"
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["checkpoint_schema"], 2)
            self.assertTrue(checkpoint["activation_quantizer_state_dict"])
            self.assertEqual(summaries[0].shared_attention_modules, 1)
            self.assertEqual(summaries[0].shared_mlp_modules, 1)
            self.assertIsNotNone(summaries[0].initial_validation_mse_loss)
            self.assertIsNotNone(summaries[0].final_validation_mse_loss)
            self.assertGreater(summaries[0].trainable_lac_parameters, 0)
            self.assertIn(summaries[0].best_epoch, (0, 1))
            with torch.no_grad():
                expected = model(**batch).logits

            restored = Qwen3ForCausalLM(hf_config).eval()
            restored.load_state_dict(original_state)
            restore_flatquant_core_layers_from_checkpoints(
                model=restored,
                layer_indices=[0],
                config=config,
                checkpoint_dir=checkpoint_dir,
                act_quant_mode="shared_input",
            )
            with torch.no_grad():
                actual = restored(**batch).logits

        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_prefix_restore_then_train_final_block(self) -> None:
        torch.manual_seed(37)
        hf_config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
        )
        original = Qwen3ForCausalLM(hf_config).eval()
        original_state = copy.deepcopy(original.state_dict())
        batches = [
            {
                "input_ids": torch.randint(0, hf_config.vocab_size, (1, 5)),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }
            for _ in range(2)
        ]
        prefix_config = FlatQuantCoreConfig(
            weight_quant_format="int4",
            activation_quant_format="int8",
            weight_group_size=0,
            epochs=1,
            transform_lr=1e-3,
            lwc_lr=1e-3,
            lac_lr=1e-3,
        )
        final_config = FlatQuantCoreConfig(
            weight_quant_format="int4",
            activation_quant_format="int8",
            weight_group_size=0,
            epochs=1,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=0,
            transform_lr=1e-3,
            lwc_lr=1e-3,
            lac_lr=1e-3,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix_dir = root / "prefix"
            final_dir = root / "final"
            prefix_model = copy.deepcopy(original)
            apply_flatquant_core_layers(
                model=prefix_model,
                model_batches=batches[:1],
                layer_indices=[0],
                config=prefix_config,
                capture_layer_input_batches=capture_layer_input_batches,
                checkpoint_dir=prefix_dir,
            )

            final_model = Qwen3ForCausalLM(hf_config).eval()
            final_model.load_state_dict(original_state)
            summaries = apply_flatquant_core_layers(
                model=final_model,
                model_batches=batches,
                layer_indices=[0, 1],
                config=final_config,
                capture_layer_input_batches=capture_layer_input_batches,
                checkpoint_dir=final_dir,
                prefix_checkpoint_dir=prefix_dir,
            )
            prefix_state = torch.load(
                prefix_dir / "layer_00.pt",
                map_location="cpu",
                weights_only=True,
            )
            copied_state = torch.load(
                final_dir / "layer_00.pt",
                map_location="cpu",
                weights_only=True,
            )
            final_state = torch.load(
                final_dir / "layer_01.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("source_checkpoint", copied_state)
            self.assertNotIn("source_checkpoint", final_state)
            for name, value in prefix_state["transform_state_dict"].items():
                torch.testing.assert_close(
                    copied_state["transform_state_dict"][name],
                    value,
                    rtol=0.0,
                    atol=0.0,
                )
            self.assertIsNone(summaries[1].best_epoch)
            self.assertIsNotNone(summaries[1].final_validation_mse_loss)
            with torch.no_grad():
                expected = final_model(**batches[0]).logits

            restored = Qwen3ForCausalLM(hf_config).eval()
            restored.load_state_dict(original_state)
            restore_flatquant_core_layers_from_checkpoints(
                model=restored,
                layer_indices=[0, 1],
                config=final_config,
                checkpoint_dir=final_dir,
            )
            with torch.no_grad():
                actual = restored(**batches[0]).logits

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_full_checkpoint_forks_into_matched_final_block_arms(self) -> None:
        torch.manual_seed(41)
        hf_config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
        )
        original = Qwen3ForCausalLM(hf_config).eval()
        original_state = copy.deepcopy(original.state_dict())
        batches = [
            {
                "input_ids": torch.randint(0, hf_config.vocab_size, (1, 6)),
                "attention_mask": torch.ones(1, 6, dtype=torch.long),
            }
            for _ in range(2)
        ]
        base_config = FlatQuantCoreConfig(
            weight_quant_format="int4",
            activation_quant_format="int8",
            weight_group_size=0,
            epochs=1,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=0,
            transform_lr=1e-3,
            lwc_lr=1e-3,
            lac_lr=1e-3,
        )
        lfq_config = FlatQuantCoreConfig(
            weight_quant_format="int4",
            activation_quant_format="int8",
            weight_group_size=0,
            learn_transform=False,
            final_objective="lfq_ce",
            lfq_loss_weight=1.0,
            lfq_boundary_loss_weight=0.3,
            lfq_boundary_topk=2,
            lfq_boundary_negative_count=2,
            epochs=1,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=0,
            transform_lr=1e-3,
            lwc_lr=1e-3,
            lac_lr=1e-3,
        )
        slot_token_ids = {
            "a": tuple(range(0, 8)),
            "b": tuple(range(8, 16)),
            "c": tuple(range(16, 24)),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base"
            frozen_dir = root / "frozen"
            base_model = copy.deepcopy(original)
            apply_flatquant_core_layers(
                model=base_model,
                model_batches=batches,
                layer_indices=[0, 1],
                config=base_config,
                capture_layer_input_batches=capture_layer_input_batches,
                checkpoint_dir=base_dir,
            )

            frozen_model = Qwen3ForCausalLM(hf_config).eval()
            frozen_model.load_state_dict(original_state)
            summaries = apply_flatquant_core_layers(
                model=frozen_model,
                model_batches=batches,
                layer_indices=[0, 1],
                config=lfq_config,
                capture_layer_input_batches=capture_layer_input_batches,
                checkpoint_dir=frozen_dir,
                finetune_checkpoint_dir=base_dir,
                lfq_token_ids=slot_token_ids,
            )
            base_prefix = torch.load(
                base_dir / "layer_00.pt", map_location="cpu", weights_only=True
            )
            copied_prefix = torch.load(
                frozen_dir / "layer_00.pt", map_location="cpu", weights_only=True
            )
            base_final = torch.load(
                base_dir / "layer_01.pt", map_location="cpu", weights_only=True
            )
            frozen_final = torch.load(
                frozen_dir / "layer_01.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(
                Path(copied_prefix["source_checkpoint"]).resolve(),
                (base_dir / "layer_00.pt").resolve(),
            )
            self.assertEqual(
                Path(frozen_final["initialization_checkpoint"]).resolve(),
                (base_dir / "layer_01.pt").resolve(),
            )
            self.assertEqual(frozen_final["objective"], "lfq_ce")
            self.assertEqual(summaries[1].objective, "lfq_ce")
            self.assertEqual(summaries[1].trainable_transform_parameters, 0)
            self.assertIsNotNone(summaries[1].final_validation_loss)
            for name, value in base_prefix["transform_state_dict"].items():
                torch.testing.assert_close(
                    copied_prefix["transform_state_dict"][name],
                    value,
                    rtol=0.0,
                    atol=0.0,
                )
            for name, value in base_final["transform_state_dict"].items():
                torch.testing.assert_close(
                    frozen_final["transform_state_dict"][name],
                    value,
                    rtol=0.0,
                    atol=0.0,
                )
            with torch.no_grad():
                expected = frozen_model(**batches[0]).logits

            restored = Qwen3ForCausalLM(hf_config).eval()
            restored.load_state_dict(original_state)
            restore_flatquant_core_layers_from_checkpoints(
                model=restored,
                layer_indices=[0, 1],
                config=lfq_config,
                checkpoint_dir=frozen_dir,
            )
            with torch.no_grad():
                actual = restored(**batches[0]).logits

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_fixed_smoothquant_control_checkpoint_and_restore(self) -> None:
        torch.manual_seed(31)
        hf_config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
        )
        model = Qwen3ForCausalLM(hf_config).eval()
        original_state = copy.deepcopy(model.state_dict())
        batch = {
            "input_ids": torch.randint(0, hf_config.vocab_size, (1, 5)),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        heldout_batch = {
            "input_ids": torch.randint(0, hf_config.vocab_size, (1, 5)),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        config = FlatQuantCoreConfig(
            weight_quant_format="int4",
            activation_quant_format="int8",
            weight_quant_scheme="symmetric",
            weight_group_size=0,
            use_lwc=True,
            learn_transform=False,
            transform_kind="smoothquant",
            smoothquant_alpha=0.4,
            epochs=1,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=1,
            lwc_lr=1e-3,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            summaries = apply_flatquant_core_layers(
                model=model,
                model_batches=[batch, heldout_batch],
                layer_indices=[0],
                config=config,
                capture_layer_input_batches=capture_layer_input_batches,
                act_quant_mode="shared_input",
                checkpoint_dir=checkpoint_dir,
            )
            checkpoint = torch.load(
                checkpoint_dir / "layer_00.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["config"]["transform_kind"], "smoothquant")
            self.assertTrue(
                all(key.endswith(".scale") for key in checkpoint["transform_state_dict"])
            )
            self.assertEqual(summaries[0].trainable_transform_parameters, 0)
            self.assertIsNotNone(summaries[0].final_validation_mse_loss)
            with torch.no_grad():
                expected = model(**batch).logits

            restored = Qwen3ForCausalLM(hf_config).eval()
            restored.load_state_dict(original_state)
            restore_flatquant_core_layers_from_checkpoints(
                model=restored,
                layer_indices=[0],
                config=config,
                checkpoint_dir=checkpoint_dir,
                act_quant_mode="shared_input",
            )
            with torch.no_grad():
                actual = restored(**batch).logits

        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
