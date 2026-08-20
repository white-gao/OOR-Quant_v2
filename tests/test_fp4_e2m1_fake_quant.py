from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fake_quant.modules import BaselineFakeQuantLinear, SmoothQuantFakeQuantLinear
from fake_quant.omniquant.runtime import OmniQuantConfig, _TrainableSymmetricLinear
from fake_quant.quant import (
    FP4_E2M1_MAX,
    QUANT_FORMAT_CHOICES,
    activation_per_token_qdq_by_format,
    fp4_e2m1_qdq_forward,
    fp4_e2m1_quantize,
    quant_format_bits,
    quant_format_qmax,
    resolve_weight_quant_scheme,
    weight_per_output_channel_qdq_forward,
)
from fake_quant.support.smoothquant_runtime import smoothquant_quantized_module_from_scales


FP4_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


class FP4E2M1FakeQuantTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_format_metadata_and_weight_scheme(self) -> None:
        self.assertIn("fp4_e2m1", QUANT_FORMAT_CHOICES)
        self.assertEqual(quant_format_bits("fp4_e2m1"), 4)
        self.assertEqual(quant_format_qmax("fp4_e2m1"), FP4_E2M1_MAX)
        self.assertEqual(resolve_weight_quant_scheme("fp4_e2m1"), "symmetric")
        with self.assertRaisesRegex(ValueError, "no affine integer zero point"):
            resolve_weight_quant_scheme("fp4_e2m1", "asymmetric")

    def test_exact_codebook_and_saturation(self) -> None:
        torch.testing.assert_close(
            fp4_e2m1_quantize(FP4_VALUES),
            FP4_VALUES,
            rtol=0.0,
            atol=0.0,
        )
        x = torch.tensor([-100.0, -6.1, 6.1, 100.0])
        expected = torch.tensor([-6.0, -6.0, 6.0, 6.0])
        torch.testing.assert_close(
            fp4_e2m1_qdq_forward(x, torch.tensor(1.0)),
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_round_to_nearest_ties_to_even(self) -> None:
        positive = torch.tensor([0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
        torch.testing.assert_close(
            fp4_e2m1_quantize(positive),
            expected,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            fp4_e2m1_quantize(-positive),
            -expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_current_weight_and_activation_granularity(self) -> None:
        first = FP4_VALUES
        second = FP4_VALUES * 2.0
        weight = torch.stack((first, second), dim=0)
        weight_qdq = weight_per_output_channel_qdq_forward(
            weight,
            quant_format="fp4_e2m1",
        )
        torch.testing.assert_close(weight_qdq, weight, rtol=0.0, atol=0.0)

        activation = torch.stack((first, second), dim=0).unsqueeze(0)
        activation_qdq = activation_per_token_qdq_by_format(
            activation,
            quant_format="fp4_e2m1",
        )
        torch.testing.assert_close(activation_qdq, activation, rtol=0.0, atol=0.0)

    def test_baseline_and_smoothquant_wrappers_support_fp4(self) -> None:
        linear = nn.Linear(8, 5, bias=True).to(dtype=torch.bfloat16)
        baseline = BaselineFakeQuantLinear(
            linear,
            act_quant="per_token",
            weight_quant_format="fp4_e2m1",
            activation_quant_format="fp4_e2m1",
        )
        self.assertEqual(baseline.weight_quant_scheme, "symmetric")
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        self.assertEqual(baseline(x).dtype, torch.bfloat16)

        smooth, replaced = smoothquant_quantized_module_from_scales(
            linear,
            {"": torch.ones(linear.in_features)},
            act_quant="per_token",
            weight_quant_format="fp4_e2m1",
            activation_quant_format="fp4_e2m1",
        )
        self.assertEqual(replaced, 1)
        self.assertIsInstance(smooth, SmoothQuantFakeQuantLinear)
        self.assertEqual(smooth.weight_quant_format, "fp4_e2m1")
        self.assertEqual(smooth.activation_quant_format, "fp4_e2m1")
        self.assertEqual(smooth(x).dtype, torch.bfloat16)

    def test_omniquant_fp4_ste_matches_finalization(self) -> None:
        config = OmniQuantConfig(
            weight_quant_format="fp4_e2m1",
            activation_quant_format="fp4_e2m1",
            weight_quant_scheme="symmetric",
            use_lwc=True,
            use_let=False,
            learn_let=False,
            epochs=1,
        )
        config.validate()
        linear = nn.Linear(8, 5, bias=True).to(dtype=torch.bfloat16)
        wrapped = _TrainableSymmetricLinear(
            linear,
            config=config,
            let_parameters=nn.ParameterDict(),
        )
        train_weight = wrapped.qdq_weight()
        final_weight = wrapped.finalize_qdq_weight(wrapped.transformed_weight())
        torch.testing.assert_close(train_weight, final_weight, rtol=0.0, atol=0.0)

        output = wrapped(torch.randn(2, 3, 8, dtype=torch.bfloat16))
        loss = output.float().square().mean()
        loss.backward()
        self.assertIsNotNone(wrapped.clip_logits.grad)
        self.assertTrue(torch.isfinite(wrapped.clip_logits.grad).all())
        self.assertGreater(torch.count_nonzero(wrapped.clip_logits.grad).item(), 0)

        invalid = OmniQuantConfig(
            weight_quant_format="fp4_e2m1",
            activation_quant_format="fp4_e2m1",
            weight_quant_scheme="asymmetric",
        )
        with self.assertRaisesRegex(ValueError, "requires weight_quant_scheme='symmetric'"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
