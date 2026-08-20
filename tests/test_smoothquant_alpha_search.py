from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fake_quant.search_smoothquant_alpha_mse import _markdown_report, _parse_alphas
from fake_quant.support.smoothquant_runtime import (
    collect_smoothquant_scales,
    collect_smoothquant_statistics,
    smoothquant_scales_from_statistics,
)


class SmoothQuantAlphaSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    def test_statistics_reuse_matches_direct_scale_collection(self) -> None:
        linear = nn.Linear(6, 4, bias=False).to(dtype=torch.bfloat16)
        batches = [
            (torch.randn(2, 3, 6, dtype=torch.bfloat16),),
            (torch.randn(1, 5, 6, dtype=torch.bfloat16),),
        ]
        statistics = collect_smoothquant_statistics(linear, batches)
        reused = smoothquant_scales_from_statistics(statistics, alpha=0.37)
        direct = collect_smoothquant_scales(linear, batches, alpha=0.37)
        self.assertEqual(set(reused), {""})
        torch.testing.assert_close(reused[""], direct[""], rtol=0.0, atol=0.0)

    def test_alpha_endpoints_follow_smoothquant_definition(self) -> None:
        activation = torch.tensor([2.0, 8.0])
        weight = torch.tensor([0.5, 0.25])
        statistics = {"linear": (activation, weight)}
        alpha_zero = smoothquant_scales_from_statistics(statistics, alpha=0.0)
        alpha_one = smoothquant_scales_from_statistics(statistics, alpha=1.0)
        torch.testing.assert_close(alpha_zero["linear"], weight.reciprocal())
        torch.testing.assert_close(alpha_one["linear"], activation)

    def test_alpha_parser_deduplicates_and_validates(self) -> None:
        self.assertEqual(_parse_alphas("0,0.25,0.25,1"), (0.0, 0.25, 1.0))
        with self.assertRaisesRegex(Exception, "in \[0, 1\]"):
            _parse_alphas("1.1")


    def test_report_contains_explicit_no_sq_control(self) -> None:
        baseline = {
            "global_mse": 2.0,
            "relative_mse": 0.2,
            "mean_layer_mse": 2.0,
            "last_layer_mse": 3.0,
        }
        candidate = {
            "alpha": 0.0,
            "global_mse": 1.5,
            "global_mse_change_percent": -25.0,
            "relative_mse": 0.15,
            "mean_layer_mse": 1.5,
            "last_layer_mse": 2.5,
        }
        report = _markdown_report(
            {
                "no_sq_rtn": baseline,
                "summary": [candidate],
                "best_by_global_mse": candidate,
            }
        )
        self.assertIn("no_sq_rtn", report)
        self.assertIn("Alpha=0 is still a SmoothQuant transform", report)
        self.assertIn("-25.0000% vs no-SQ RTN", report)


if __name__ == "__main__":
    unittest.main()
