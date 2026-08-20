from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fake_quant.modules import BaselineFakeQuantLinear
from fake_quant.omniquant.runtime import (
    DEFAULT_OMNIQUANT_EPOCHS,
    OMNIQUANT_CALIBRATION_FORWARD_MODE,
    OmniQuantConfig,
    _LFQOutputProjector,
    _TrainableOmniBlock,
    _TrainableSymmetricLinear,
    _finalize_block,
    _first_tensor,
    _lfq_soft_cross_entropy,
    _validate_omniquant_checkpoint,
)
from fake_quant.quant import FAKE_QUANT_FORWARD_MODE
from fake_quant.support.smoothquant_runtime import DEFAULT_SMOOTHQUANT_ALPHA


class _TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(dtype=input_dtype)


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.head_dim = hidden_size // 2
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # Q/K are evaluated to exercise their wrappers. The linear V->O path
        # preserves the exact LET fold used by the real attention value path.
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states) + (q + k) * 0.0
        return (self.o_proj(value),)


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        intermediate_size = hidden_size * 2
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _TinyDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.input_layernorm = _TinyRMSNorm(hidden_size)
        self.self_attn = _TinyAttention(hidden_size)
        self.post_attention_layernorm = _TinyRMSNorm(hidden_size)
        self.mlp = _TinyMLP(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))[0]
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states,)


def _config(*, use_let: bool) -> OmniQuantConfig:
    return OmniQuantConfig(
        weight_quant_format="int4",
        activation_quant_format="int8",
        weight_quant_scheme="asymmetric",
        use_lwc=True,
        use_let=use_let,
        learn_let=use_let,
        let_init="ones",
        epochs=1,
    )


class DeploymentMatchedFakeQuantTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_default_omniquant_protocol(self) -> None:
        config = OmniQuantConfig()
        self.assertEqual(DEFAULT_OMNIQUANT_EPOCHS, 20)
        self.assertEqual(config.epochs, 20)
        self.assertEqual(DEFAULT_SMOOTHQUANT_ALPHA, 0.4)
        self.assertEqual(config.smoothquant_alpha, 0.4)
        self.assertEqual(config.lwc_lr, 1e-2)
        self.assertEqual(config.let_lr, 5e-3)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertIsNone(config.max_grad_norm)

    def test_repository_contract_and_baseline_operator_dtype(self) -> None:
        self.assertEqual(FAKE_QUANT_FORWARD_MODE, "deployment_matched")
        self.assertEqual(OMNIQUANT_CALIBRATION_FORWARD_MODE, FAKE_QUANT_FORWARD_MODE)
        linear = nn.Linear(8, 5, bias=True).to(dtype=torch.bfloat16)
        wrapped = BaselineFakeQuantLinear(
            linear,
            act_quant="per_token",
            weight_quant_format="int8",
            weight_quant_scheme="asymmetric",
            activation_quant_format="int8",
        )
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        self.assertEqual(wrapped.weight_qdq.dtype, torch.bfloat16)
        self.assertEqual(wrapped.quantize_activation(x).dtype, torch.bfloat16)
        self.assertEqual(wrapped(x).dtype, torch.bfloat16)

    def _assert_train_finalize_match(self, *, use_let: bool) -> None:
        block = _TinyDecoderBlock().to(dtype=torch.bfloat16)
        train_block = _TrainableOmniBlock(block, config=_config(use_let=use_let), init_scales={})
        self.assertEqual(train_block.calibration_dtype, torch.bfloat16)
        for child in train_block.block.modules():
            if isinstance(child, _TrainableSymmetricLinear):
                self.assertEqual(child.weight_fp.dtype, torch.float32)
                self.assertEqual(child.execution_dtype, torch.bfloat16)

        if use_let:
            with torch.no_grad():
                train_block.let_parameters["qkv"].copy_(torch.linspace(-0.3, 0.25, 8))
                train_block.let_parameters["mlp"].copy_(torch.linspace(0.2, -0.2, 8))
                train_block.let_parameters["vo"].copy_(torch.linspace(-0.15, 0.3, 8))

        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        train_output = _first_tensor(train_block(x))
        self.assertEqual(train_output.dtype, torch.bfloat16)

        loss = train_output.float().square().mean()
        loss.backward()
        lwc_grads = [
            parameter.grad
            for child in train_block.block.modules()
            if isinstance(child, _TrainableSymmetricLinear)
            for parameter in child.lwc_parameters()
        ]
        self.assertTrue(any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in lwc_grads))
        if use_let:
            self.assertTrue(
                any(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                    for parameter in train_block.let_parameters.values()
                )
            )

        finalized, replaced = _finalize_block(train_block)
        self.assertEqual(replaced, 7)
        with torch.no_grad():
            final_output = _first_tensor(finalized(x))
        self.assertEqual(final_output.dtype, torch.bfloat16)
        torch.testing.assert_close(train_output.detach(), final_output, rtol=0.0, atol=0.0)

    def test_lwc_train_wrapper_matches_finalized_block(self) -> None:
        self._assert_train_finalize_match(use_let=False)

    def test_let_train_wrapper_matches_finalized_block(self) -> None:
        self._assert_train_finalize_match(use_let=True)

    def test_lfq_projector_executes_model_dtype_and_loss_uses_fp32(self) -> None:
        norm = _TinyRMSNorm(8).to(dtype=torch.bfloat16)
        head = nn.Linear(8, 12, bias=False).to(dtype=torch.bfloat16)
        projector = _LFQOutputProjector(
            final_norm=norm,
            output_head=head,
            token_ids={"a": (0, 1, 2), "b": (3, 4, 5), "c": (6, 7, 8)},
        )
        hidden = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
        projected = projector(hidden)
        self.assertTrue(all(logits.dtype == torch.bfloat16 for logits in projected.values()))
        teachers = {
            slot: torch.softmax(torch.randn_like(logits, dtype=torch.float32), dim=-1)
            for slot, logits in projected.items()
        }
        loss, slot_losses = _lfq_soft_cross_entropy(
            hidden,
            teachers,
            projector,
            {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0},
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(all(value.dtype == torch.float32 for value in slot_losses.values()))

    def test_legacy_fp32_surrogate_checkpoint_is_rejected(self) -> None:
        config = _config(use_let=False)
        old_state = {
            "config": {
                "weight_quant_format": config.weight_quant_format,
                "activation_quant_format": config.activation_quant_format,
                "weight_quant_scheme": config.weight_quant_scheme,
                "use_lwc": config.use_lwc,
                "use_let": config.use_let,
                "learn_let": config.learn_let,
                "calibration_compute_dtype": "float32",
            },
            "objective": "mse",
        }
        with self.assertRaisesRegex(ValueError, "calibration_forward_mode"):
            _validate_omniquant_checkpoint(
                old_state,
                checkpoint_path=Path("legacy_layer.pt"),
                config=config,
                expected_objective="mse",
                require_run_objective=False,
            )


if __name__ == "__main__":
    unittest.main()
