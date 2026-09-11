from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fake_quant.apply import apply_baseline_qdq
from fake_quant.modules import (
    BaselineFakeQuantLinear,
    SmoothQuantFakeQuantLinear,
)
from fake_quant.omniquant.runtime import OmniQuantConfig, _TrainableSymmetricLinear
from fake_quant.quant import (
    normalize_weight_group_size,
    weight_per_output_channel_qdq_forward,
)
from fake_quant.support.smoothquant_runtime import (
    smoothquant_quantized_module_from_scales,
)


class GroupwiseWeightQuantTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_zero_group_size_preserves_per_channel_path(self) -> None:
        weight = torch.randn(5, 11, dtype=torch.float32)
        expected = weight_per_output_channel_qdq_forward(
            weight,
            quant_format="int4",
            quant_scheme="asymmetric",
        )
        actual = weight_per_output_channel_qdq_forward(
            weight,
            quant_format="int4",
            quant_scheme="asymmetric",
            group_size=0,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_groupwise_matches_independent_input_slices_with_tail(self) -> None:
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0],
                [-1.0, -2.0, -3.0, -10.0, -20.0, -30.0, -100.0, -200.0],
            ]
        )
        actual = weight_per_output_channel_qdq_forward(
            weight,
            quant_format="int4",
            quant_scheme="asymmetric",
            group_size=3,
        )
        expected = torch.cat(
            [
                weight_per_output_channel_qdq_forward(
                    weight[:, start : start + 3],
                    quant_format="int4",
                    quant_scheme="asymmetric",
                )
                for start in range(0, weight.shape[1], 3)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_finer_groups_reduce_weight_reconstruction_error(self) -> None:
        weight = torch.tensor(
            [
                [-1.0, -0.5, 0.5, 1.0, -100.0, -50.0, 50.0, 100.0],
                [-2.0, -1.0, 1.0, 2.0, -80.0, -40.0, 40.0, 80.0],
            ]
        )
        per_channel = weight_per_output_channel_qdq_forward(
            weight,
            quant_format="int4",
            quant_scheme="symmetric",
        )
        groupwise = weight_per_output_channel_qdq_forward(
            weight,
            quant_format="int4",
            quant_scheme="symmetric",
            group_size=4,
        )
        per_channel_mse = (per_channel - weight).float().square().mean()
        groupwise_mse = (groupwise - weight).float().square().mean()
        self.assertLess(groupwise_mse.item(), per_channel_mse.item())

    def test_rtn_replacement_propagates_group_size(self) -> None:
        model = nn.Sequential(nn.Linear(8, 6), nn.GELU(), nn.Linear(6, 4))
        summary = apply_baseline_qdq(
            model,
            weight_quant_format="int4",
            weight_quant_scheme="asymmetric",
            weight_group_size=4,
            activation_quant_format="none",
        )
        self.assertEqual(summary.replaced_linears, 2)
        wrappers = [
            module
            for module in model.modules()
            if isinstance(module, BaselineFakeQuantLinear)
        ]
        self.assertEqual(len(wrappers), 2)
        self.assertTrue(all(module.weight_group_size == 4 for module in wrappers))
        output = model(torch.randn(3, 8))
        self.assertEqual(tuple(output.shape), (3, 4))

    def test_smoothquant_replacement_propagates_group_size(self) -> None:
        linear = nn.Linear(8, 5)
        wrapped, replaced = smoothquant_quantized_module_from_scales(
            linear,
            {"": torch.ones(8)},
            act_quant="none",
            weight_quant_format="int4",
            weight_quant_scheme="asymmetric",
            weight_group_size=4,
            activation_quant_format="none",
        )
        self.assertEqual(replaced, 1)
        self.assertIsInstance(wrapped, SmoothQuantFakeQuantLinear)
        self.assertEqual(wrapped.weight_group_size, 4)
        output = wrapped(torch.randn(2, 8))
        self.assertEqual(tuple(output.shape), (2, 5))

    def _omniquant_groupwise_matches_independent_slices(
        self,
        *,
        weight_quant_format: str,
        weight_quant_scheme: str,
        symmetric_lwc_mode: str = "absmax",
    ) -> _TrainableSymmetricLinear:
        linear = nn.Linear(10, 3, bias=False)
        config = OmniQuantConfig(
            weight_quant_format=weight_quant_format,
            activation_quant_format="none",
            weight_quant_scheme=weight_quant_scheme,
            symmetric_lwc_mode=symmetric_lwc_mode,
            weight_group_size=4,
            use_lwc=True,
            use_let=False,
            learn_let=False,
            init_lwc_logit=0.75,
            epochs=1,
        )
        config.validate()
        wrapped = _TrainableSymmetricLinear(
            linear,
            config=config,
            let_parameters=nn.ParameterDict(),
        )
        self.assertEqual(wrapped.weight_group_size, 4)
        self.assertEqual(wrapped.num_weight_groups, 3)

        expected_parts = []
        for group_idx, start in enumerate(range(0, linear.in_features, 4)):
            weight_slice = linear.weight[:, start : start + 4]
            slice_linear = nn.Linear(
                weight_slice.shape[1],
                linear.out_features,
                bias=False,
            )
            with torch.no_grad():
                slice_linear.weight.copy_(weight_slice)
            slice_config = OmniQuantConfig(
                weight_quant_format=weight_quant_format,
                activation_quant_format="none",
                weight_quant_scheme=weight_quant_scheme,
                symmetric_lwc_mode=symmetric_lwc_mode,
                weight_group_size=0,
                use_lwc=True,
                use_let=False,
                learn_let=False,
                init_lwc_logit=0.75,
                epochs=1,
            )
            slice_wrapped = _TrainableSymmetricLinear(
                slice_linear,
                config=slice_config,
                let_parameters=nn.ParameterDict(),
            )
            with torch.no_grad():
                for parameter_name in wrapped.lwc_state():
                    grouped_parameter = getattr(wrapped, parameter_name)
                    slice_parameter = getattr(slice_wrapped, parameter_name)
                    slice_parameter.copy_(grouped_parameter[:, group_idx])
            expected_parts.append(
                slice_wrapped.finalize_qdq_weight(slice_linear.weight)
            )

        actual = wrapped.finalize_qdq_weight(linear.weight)
        expected = torch.cat(expected_parts, dim=1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            wrapped.qdq_weight(),
            actual,
            rtol=0.0,
            atol=0.0,
        )
        return wrapped

    def test_omniquant_symmetric_groupwise_lwc_with_tail(self) -> None:
        wrapped = self._omniquant_groupwise_matches_independent_slices(
            weight_quant_format="fp4_e2m1",
            weight_quant_scheme="symmetric",
        )
        self.assertEqual(tuple(wrapped.clip_logits.shape), (3, 3, 1))
        output = wrapped(torch.randn(2, 10))
        output.float().square().mean().backward()
        self.assertIsNotNone(wrapped.clip_logits.grad)
        self.assertTrue(torch.isfinite(wrapped.clip_logits.grad).all())
        self.assertGreater(torch.count_nonzero(wrapped.clip_logits.grad).item(), 0)

    def test_omniquant_symmetric_two_sided_groupwise_lwc_with_tail(self) -> None:
        wrapped = self._omniquant_groupwise_matches_independent_slices(
            weight_quant_format="fp4_e2m1",
            weight_quant_scheme="symmetric",
            symmetric_lwc_mode="two_sided",
        )
        self.assertIsNone(wrapped.clip_logits)
        self.assertEqual(tuple(wrapped.upper_clip_logits.shape), (3, 3, 1))
        self.assertEqual(tuple(wrapped.lower_clip_logits.shape), (3, 3, 1))
        output = wrapped(torch.randn(2, 10))
        output.float().square().mean().backward()
        for parameter in wrapped.lwc_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_omniquant_asymmetric_groupwise_lwc_with_tail(self) -> None:
        wrapped = self._omniquant_groupwise_matches_independent_slices(
            weight_quant_format="int4",
            weight_quant_scheme="asymmetric",
        )
        self.assertEqual(tuple(wrapped.upper_clip_logits.shape), (3, 3, 1))
        self.assertEqual(tuple(wrapped.lower_clip_logits.shape), (3, 3, 1))

    def test_omniquant_group_size_validation(self) -> None:
        invalid = OmniQuantConfig(weight_group_size=-1)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            invalid.validate()

    def test_invalid_group_size_is_rejected(self) -> None:
        self.assertIsNone(normalize_weight_group_size(None))
        self.assertIsNone(normalize_weight_group_size(0))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            normalize_weight_group_size(-1)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            normalize_weight_group_size(True)


if __name__ == "__main__":
    unittest.main()
