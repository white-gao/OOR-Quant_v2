from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from flat_quant.flatquant.runtime import (
    FlatQuantCoreConfig,
    FlatQuantCoreLinear,
    _TrainableFlatQuantBlock,
    _collect_fixed_smoothquant_transform_scales,
    _finalize_flat_block,
    _first_tensor,
    _official_transform_activation,
    _official_transformed_bias,
    _official_transformed_weight,
    _train_flat_block,
)
from flat_quant.flatquant.transforms import (
    FixedSmoothQuantTransform,
    KroneckerSVDTransform,
    SingleSVDTransform,
    closest_factor_pair,
    kronecker_matmul,
)
from flat_quant.support.smoothquant_runtime import collect_smoothquant_scales


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.head_dim = 4
        self.config = SimpleNamespace(num_attention_heads=hidden_size // self.head_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states) + (q + k) * 0.0
        return self.o_proj(value), None


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int = 12, intermediate_size: int = 24) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = _TinyAttention(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = _TinyMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        residual = hidden_states
        attention_input = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(attention_input)[0]
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        return (residual + self.mlp(mlp_input),)


def _config(
    *,
    epochs: int = 1,
    use_lwc: bool = True,
    use_lac: bool = True,
    learn_lac: bool = True,
    learn_transform: bool = True,
    train_sample_size: int = 0,
    validation_sample_size: int = 0,
    epoch_eval_interval: int = 0,
    transform_kind: str = "kronecker",
) -> FlatQuantCoreConfig:
    return FlatQuantCoreConfig(
        weight_quant_format="int4",
        activation_quant_format="int8",
        weight_quant_scheme="symmetric",
        weight_group_size=0,
        use_lwc=use_lwc,
        use_lac=use_lac,
        learn_lac=learn_lac,
        learn_transform=learn_transform,
        transform_kind=transform_kind,
        transform_init="identity",
        epochs=epochs,
        train_sample_size=train_sample_size,
        validation_sample_size=validation_sample_size,
        epoch_eval_interval=epoch_eval_interval,
        transform_lr=1e-3,
        lwc_lr=1e-3,
        lac_lr=1e-3,
        normalize_mse_gradient=True,
    )


class FlatQuantTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_closest_factor_pair_matches_onerec_dimensions(self) -> None:
        self.assertEqual(closest_factor_pair(2048), (32, 64))
        self.assertEqual(closest_factor_pair(6144), (64, 96))
        self.assertEqual(closest_factor_pair(12), (2, 6))
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            closest_factor_pair(6)

    def test_fast_kronecker_matmul_matches_explicit_matrix(self) -> None:
        x = torch.randn(5, 12)
        left = torch.randn(3, 3)
        right = torch.randn(4, 4)
        expected = x @ torch.kron(left, right)
        actual = kronecker_matmul(x, left, right)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_transform_and_inverse_weight_preserve_unquantized_linear(self) -> None:
        transform = KroneckerSVDTransform(
            12,
            init="random_orthogonal",
            add_diag=True,
            diag_init=torch.exp(torch.randn(12)),
        )
        with torch.no_grad():
            transform.left.singular_values.copy_(
                torch.linspace(0.75, 1.2, transform.left_size)
            )
            transform.right.singular_values.copy_(
                torch.linspace(1.15, 0.7, transform.right_size)
            )
        x = torch.randn(7, 12)
        weight = torch.randn(9, 12)
        expected = F.linear(x, weight)
        actual = F.linear(transform(x), transform.transform_weight(weight))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_official_attention_and_mlp_topology_preserves_full_precision(self) -> None:
        transforms = nn.ModuleDict(
            {
                "attn_in": KroneckerSVDTransform(
                    12,
                    add_diag=True,
                    diag_init=torch.exp(torch.randn(12) * 0.2),
                ),
                "head": SingleSVDTransform(3),
                "head_dim": SingleSVDTransform(4),
                "mlp_in": KroneckerSVDTransform(
                    12,
                    add_diag=True,
                    diag_init=torch.exp(torch.randn(12) * 0.2),
                ),
                "down": KroneckerSVDTransform(
                    24,
                    add_diag=True,
                    diag_init=torch.exp(torch.randn(24) * 0.2),
                ),
            }
        )
        x = torch.randn(2, 5, 12)

        value_weight = torch.randn(12, 12)
        value_bias = torch.randn(12)
        output_weight = torch.randn(12, 12)
        output_bias = torch.randn(12)
        expected_attention = F.linear(
            F.linear(x, value_weight, value_bias),
            output_weight,
            output_bias,
        )
        attention_input = x * transforms["attn_in"].diag_scale
        attention_input = _official_transform_activation(
            attention_input,
            role="v",
            transform_bank=transforms,
            head_dim=4,
        )
        transformed_value = F.linear(
            attention_input,
            _official_transformed_weight(
                value_weight,
                role="v",
                transform_bank=transforms,
                head_dim=4,
            ),
            _official_transformed_bias(
                value_bias,
                role="v",
                transform_bank=transforms,
            ),
        )
        output_input = _official_transform_activation(
            transformed_value,
            role="o",
            transform_bank=transforms,
            head_dim=4,
        )
        actual_attention = F.linear(
            output_input,
            _official_transformed_weight(
                output_weight,
                role="o",
                transform_bank=transforms,
                head_dim=4,
            ),
            output_bias,
        )
        torch.testing.assert_close(
            actual_attention,
            expected_attention,
            rtol=3e-5,
            atol=3e-5,
        )

        gate_weight = torch.randn(24, 12)
        gate_bias = torch.randn(24)
        up_weight = torch.randn(24, 12)
        up_bias = torch.randn(24)
        down_weight = torch.randn(12, 24)
        down_bias = torch.randn(12)
        expected_mlp = F.linear(
            F.silu(F.linear(x, gate_weight, gate_bias))
            * F.linear(x, up_weight, up_bias),
            down_weight,
            down_bias,
        )
        mlp_input = x * transforms["mlp_in"].diag_scale
        mlp_input = _official_transform_activation(
            mlp_input,
            role="gate",
            transform_bank=transforms,
            head_dim=4,
        )
        transformed_gate = F.linear(
            mlp_input,
            _official_transformed_weight(
                gate_weight,
                role="gate",
                transform_bank=transforms,
                head_dim=4,
            ),
            gate_bias,
        )
        transformed_up = F.linear(
            mlp_input,
            _official_transformed_weight(
                up_weight,
                role="up",
                transform_bank=transforms,
                head_dim=4,
            ),
            _official_transformed_bias(
                up_bias,
                role="up",
                transform_bank=transforms,
            ),
        )
        down_input = F.silu(transformed_gate) * transformed_up
        down_input = _official_transform_activation(
            down_input,
            role="down",
            transform_bank=transforms,
            head_dim=4,
        )
        actual_mlp = F.linear(
            down_input,
            _official_transformed_weight(
                down_weight,
                role="down",
                transform_bank=transforms,
                head_dim=4,
            ),
            down_bias,
        )
        torch.testing.assert_close(actual_mlp, expected_mlp, rtol=3e-5, atol=3e-5)

    def test_near_singular_raw_factor_fails_fast(self) -> None:
        transform = KroneckerSVDTransform(12, init="identity")
        with torch.no_grad():
            transform.left.singular_values[0] = 0.0
        with self.assertRaisesRegex(FloatingPointError, "near-singular"):
            transform.transform_weight(torch.randn(5, 12))

    def test_fixed_smoothquant_pair_preserves_unquantized_linear(self) -> None:
        transform = FixedSmoothQuantTransform(torch.exp(torch.randn(12)))
        x = torch.randn(7, 12)
        weight = torch.randn(9, 12)
        expected = F.linear(x, weight)
        actual = F.linear(transform(x), transform.transform_weight(weight))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class FlatQuantRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_finalize_matches_train_forward_and_retains_online_transforms(self) -> None:
        train_block = _TrainableFlatQuantBlock(
            _TinyBlock(),
            config=_config(),
            act_quant_mode="per_linear",
        )
        with torch.no_grad():
            train_block.transforms["attn_in"].left.singular_values.copy_(
                torch.linspace(
                    0.9,
                    1.15,
                    train_block.transforms["attn_in"].left_size,
                )
            )
            train_block.transforms["down"].right.singular_values.copy_(
                torch.linspace(
                    1.2,
                    0.9,
                    train_block.transforms["down"].right_size,
                )
            )
        x = torch.randn(2, 4, 12)
        train_block.eval()
        with torch.no_grad():
            expected = _first_tensor(train_block(x))
            finalized, replaced, shared_attention, shared_mlp = _finalize_flat_block(
                train_block
            )
            actual = _first_tensor(finalized(x))
        self.assertEqual(replaced, 7)
        self.assertEqual(shared_attention, 0)
        self.assertEqual(shared_mlp, 0)
        self.assertTrue(hasattr(finalized, "flatquant_transforms"))
        self.assertEqual(
            sum(isinstance(module, FlatQuantCoreLinear) for module in finalized.modules()),
            7,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_lac_can_remain_active_while_frozen(self) -> None:
        train_block = _TrainableFlatQuantBlock(
            _TinyBlock(),
            config=_config(
                use_lwc=True,
                use_lac=True,
                learn_lac=False,
                learn_transform=False,
            ),
            act_quant_mode="per_linear",
        )
        quantizer = train_block.activation_quantizers["attn_in"]
        with torch.no_grad():
            quantizer.upper_clip_logit.zero_()
            quantizer.lower_clip_logit.zero_()
        values = torch.tensor([[[-4.0, -1.0, 2.0, 8.0]]])
        expected = torch.tensor([[[-2.0, -1.0, 2.0, 4.0]]])
        torch.testing.assert_close(
            quantizer.clipped(values),
            expected,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in quantizer.clip_parameters()
            )
        )

    def test_one_epoch_mse_training_has_finite_transform_gradients(self) -> None:
        teacher = _TinyBlock()
        train_block = _TrainableFlatQuantBlock(
            copy.deepcopy(teacher),
            config=_config(epochs=1),
            act_quant_mode="per_linear",
        )
        initial_lac = {
            name: parameter.detach().clone()
            for name, parameter in train_block.activation_quantizers.named_parameters()
        }
        batches = [torch.randn(1, 3, 12), torch.randn(1, 2, 12)]
        (
            initial,
            final,
            initial_validation,
            final_validation,
            best_epoch,
            metrics,
        ) = _train_flat_block(
            teacher_block=teacher,
            train_block=train_block,
            fp_inputs=batches,
            quant_inputs=batches,
            config=train_block.config,
            layer_idx=0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(initial)))
        self.assertTrue(torch.isfinite(torch.tensor(final)))
        self.assertIsNone(initial_validation)
        self.assertIsNone(final_validation)
        self.assertIsNone(best_epoch)
        self.assertEqual([metric.epoch for metric in metrics], [0, 1])
        self.assertIsNotNone(metrics[-1].mean_grad_norm)
        self.assertGreater(metrics[-1].mean_grad_norm or 0.0, 0.0)
        self.assertIsNotNone(metrics[-1].lac_lr)
        self.assertTrue(
            any(
                not torch.equal(initial_lac[name], parameter.detach())
                for name, parameter in train_block.activation_quantizers.named_parameters()
            )
        )

    def test_rtn_arm_skips_training_and_reports_heldout_mse(self) -> None:
        teacher = _TinyBlock()
        config = _config(
            epochs=2,
            use_lwc=False,
            use_lac=False,
            learn_transform=False,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=1,
        )
        train_block = _TrainableFlatQuantBlock(
            copy.deepcopy(teacher),
            config=config,
            act_quant_mode="per_linear",
        )
        batches = [torch.randn(1, 3, 12), torch.randn(1, 2, 12)]
        (
            initial,
            final,
            initial_validation,
            final_validation,
            best_epoch,
            metrics,
        ) = _train_flat_block(
            teacher_block=teacher,
            train_block=train_block,
            fp_inputs=batches,
            quant_inputs=batches,
            config=config,
            layer_idx=0,
        )
        self.assertEqual(initial, final)
        self.assertEqual(initial_validation, final_validation)
        self.assertEqual(best_epoch, 0)
        self.assertEqual(len(metrics), 1)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in train_block.transforms.parameters()
            )
        )

    def test_lwc_only_tracks_heldout_and_selects_training_best_epoch(self) -> None:
        teacher = _TinyBlock()
        config = _config(
            epochs=2,
            use_lwc=True,
            use_lac=False,
            learn_transform=False,
            train_sample_size=2,
            validation_sample_size=2,
            epoch_eval_interval=1,
        )
        train_block = _TrainableFlatQuantBlock(
            copy.deepcopy(teacher),
            config=config,
            act_quant_mode="per_linear",
        )
        batches = [torch.randn(1, 3, 12) for _ in range(4)]
        (
            _initial,
            _final,
            initial_validation,
            final_validation,
            best_epoch,
            metrics,
        ) = _train_flat_block(
            teacher_block=teacher,
            train_block=train_block,
            fp_inputs=batches,
            quant_inputs=batches,
            config=config,
            layer_idx=0,
        )
        self.assertIsNotNone(initial_validation)
        self.assertIsNotNone(final_validation)
        self.assertIn(best_epoch, (0, 1, 2))
        self.assertEqual([metric.epoch for metric in metrics], [0, 1, 2])
        self.assertTrue(
            all(metric.validation_mse is not None for metric in metrics)
        )

    def test_smoothquant_statistics_use_training_prefix_only(self) -> None:
        teacher = _TinyBlock()
        train_batch = torch.linspace(-1.0, 1.0, 12).reshape(1, 1, 12).repeat(1, 3, 1)
        heldout_outlier = torch.zeros(1, 3, 12)
        heldout_outlier[..., 0] = 1000.0
        config = _config(
            use_lwc=False,
            use_lac=False,
            learn_transform=False,
            train_sample_size=1,
            validation_sample_size=1,
            epoch_eval_interval=1,
            transform_kind="smoothquant",
        )
        scales = _collect_fixed_smoothquant_transform_scales(
            teacher,
            [train_batch, heldout_outlier],
            config=config,
        )
        expected = collect_smoothquant_scales(
            teacher,
            [train_batch],
            alpha=config.smoothquant_alpha,
        )
        leaked = collect_smoothquant_scales(
            teacher,
            [train_batch, heldout_outlier],
            alpha=config.smoothquant_alpha,
        )
        torch.testing.assert_close(scales["qkv"], expected["self_attn.q_proj"])
        self.assertFalse(
            torch.allclose(scales["qkv"], leaked["self_attn.q_proj"])
        )


if __name__ == "__main__":
    unittest.main()
