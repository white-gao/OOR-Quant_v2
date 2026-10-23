from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from flat_quant.flatquant.runtime import (
    FlatQuantCoreConfig, _TrainableFlatQuantBlock, _finalize_flat_block,
    _restore_flatquant_train_state, _train_flat_block,
)
from flat_quant.support.smoothquant_runtime import _batch_to_args_kwargs
from flat_quant.verify_stage_calibration import (
    balanced_mse, capture_inputs, learned_state, prefill_batch,
)


def test_stage_weights_and_history_length_invariance():
    short = torch.tensor([1., 1., 2., 3., 4.]).reshape(1, -1, 1).requires_grad_()
    long = torch.tensor([1.] * 20 + [2., 3., 4.]).reshape(1, -1, 1)
    expected = torch.tensor((1. + 4. + 9. + 16.) / 4)
    torch.testing.assert_close(balanced_mse(short, torch.zeros_like(short)), expected)
    torch.testing.assert_close(balanced_mse(long, torch.zeros_like(long)), expected)
    balanced_mse(short, torch.zeros_like(short)).backward()
    torch.testing.assert_close(short.grad.flatten(), torch.tensor([.25, .25, 1., 1.5, 2.]))


@pytest.mark.parametrize("shape", [(1, 3, 4), (2, 5, 4), (5, 4)])
def test_invalid_stage_shapes_are_rejected(shape):
    with pytest.raises(ValueError):
        balanced_mse(torch.zeros(shape), torch.zeros(shape))


def test_prefill_retains_a_prediction_position():
    complete = {"input_ids": torch.tensor([[1, 2, 90, 91, 92]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    prompt = prefill_batch(complete)
    assert prompt["input_ids"].tolist() == [[1, 2, 90]]
    assert complete["input_ids"].shape == (1, 5)
    complete["attention_mask"][0, 0] = 0
    with pytest.raises(ValueError):
        prefill_batch(complete)


@pytest.mark.parametrize("loss_fn", [F.mse_loss, balanced_mse])
def test_training_callback_and_finalization_roundtrip(loss_fn):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=96, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=64, attn_implementation="sdpa",
    )).eval().requires_grad_(False)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    batches = [{"input_ids": torch.randint(0, 96, (1, 6)),
                "attention_mask": torch.ones(1, 6, dtype=torch.long)}]
    cached = capture_inputs(model, 1, batches, "unit capture")
    assert not model.model.layers[1]._forward_pre_hooks
    args, kwargs = _batch_to_args_kwargs(cached[0])
    config = FlatQuantCoreConfig(epochs=1, train_sample_size=1, transform_lr=1e-3,
                                 lwc_lr=1e-3, lac_lr=1e-3)
    train = _TrainableFlatQuantBlock(copy.deepcopy(model.model.layers[1]),
                                     config=config, act_quant_mode="shared_input")
    initial_state = learned_state(train)
    with torch.no_grad():
        expected_initial = loss_fn(train(*args, **kwargs).float(),
                                   model.model.layers[1](*args, **kwargs).float())
    result = _train_flat_block(teacher_block=model.model.layers[1], train_block=train,
                               fp_inputs=cached, quant_inputs=cached, config=config,
                               layer_idx=1, reconstruction_loss=loss_fn)
    assert result[0] == pytest.approx(float(expected_initial), rel=1e-6)
    assert result[4] is None
    assert torch.isfinite(torch.tensor(result[1]))
    final_state = learned_state(train)
    assert any(not torch.equal(v, final_state["transform_state_dict"][k])
               for k, v in initial_state["transform_state_dict"].items())
    restored = _TrainableFlatQuantBlock(copy.deepcopy(model.model.layers[1]),
                                        config=config, act_quant_mode="shared_input")
    _restore_flatquant_train_state(restored, final_state, checkpoint_path=Path("unit"))
    final_a, final_b = _finalize_flat_block(train)[0], _finalize_flat_block(restored)[0]
    with torch.no_grad():
        torch.testing.assert_close(final_a(*args, **kwargs), final_b(*args, **kwargs), rtol=0, atol=0)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
