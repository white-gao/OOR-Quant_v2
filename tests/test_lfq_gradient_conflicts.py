from __future__ import annotations

import unittest

import torch

from fake_quant.analyze_lfq_gradient_conflicts import (
    GradientConflictAccumulator,
    summarize_component_alignment,
)


class LFQGradientConflictTest(unittest.TestCase):
    def test_accumulator_separates_mean_and_sample_conflicts(self) -> None:
        accumulator = GradientConflictAccumulator()
        accumulator.update(
            losses={"a": 1.0, "b": 2.0, "c": 3.0},
            gradients={
                "a": torch.tensor([1.0, 0.0]),
                "b": torch.tensor([-1.0, 0.0]),
                "c": torch.tensor([1.0, 1.0]),
            },
        )
        result = accumulator.finalize()
        self.assertEqual(result["sample_count"], 1)
        self.assertAlmostEqual(result["pairwise"]["a_b"]["mean_gradient_cosine"], -1.0)
        self.assertEqual(
            result["pairwise"]["a_b"]["per_sample_cosine"]["negative_fraction"],
            1.0,
        )
        self.assertTrue(
            result["slot_alignment_with_total"]["a"]["first_order_descent_compatible"]
        )
        self.assertFalse(
            result["slot_alignment_with_total"]["b"]["first_order_descent_compatible"]
        )

    def test_zero_norm_cosines_are_counted_not_serialized_as_nan(self) -> None:
        accumulator = GradientConflictAccumulator()
        accumulator.update(
            losses={"a": 0.0, "b": 0.0, "c": 0.0},
            gradients={
                "a": torch.zeros(2),
                "b": torch.ones(2),
                "c": torch.tensor([1.0, -1.0]),
            },
        )
        result = accumulator.finalize()
        pair = result["pairwise"]["a_b"]
        self.assertIsNone(pair["mean_gradient_cosine"])
        self.assertIsNone(pair["per_sample_cosine"]["negative_fraction"])
        self.assertEqual(pair["per_sample_cosine"]["zero_norm_count"], 1)

    def test_component_alignment_reports_same_slot_conflict(self) -> None:
        ce = GradientConflictAccumulator()
        boundary = GradientConflictAccumulator()
        ce_gradients = {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([0.0, 1.0]),
            "c": torch.tensor([1.0, 1.0]),
        }
        boundary_gradients = {
            "a": torch.tensor([-1.0, 0.0]),
            "b": torch.tensor([0.0, 1.0]),
            "c": torch.tensor([1.0, -1.0]),
        }
        losses = {"a": 0.0, "b": 0.0, "c": 0.0}
        ce.update(losses=losses, gradients=ce_gradients)
        boundary.update(losses=losses, gradients=boundary_gradients)
        result = summarize_component_alignment(
            ce,
            boundary,
            {"a": [-1.0], "b": [1.0], "c": [0.0]},
            {"a": 0, "b": 0, "c": 0},
        )
        self.assertTrue(result["a"]["mean_gradient_conflict"])
        self.assertFalse(result["b"]["mean_gradient_conflict"])
        self.assertAlmostEqual(result["c"]["mean_gradient_cosine"], 0.0)


if __name__ == "__main__":
    unittest.main()
