# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    get_forward_backward_func,
    layer_by_layer_forward_backward_no_pipelining,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=4,
            num_attention_heads=1,
            calculate_per_token_loss=True,
            grad_scale_func=lambda loss: 2 * loss,
        )
        self.model_type = ModelType.encoder_or_decoder
        self.embedding = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))
        self.head = nn.Linear(4, 1)
        self.layer_calls = []

    def set_input_tensor(self, inputs):
        assert inputs == [None]


class _Plan:
    def __init__(self, model, microbatch, tokens):
        self.model = model
        self.microbatch = microbatch
        self.tokens = tokens

    def preprocess(self):
        return self.model.embedding(self.tokens)

    def num_layers(self):
        return len(self.model.layers)

    def forward_layer(self, index, hidden_states):
        self.model.layer_calls.append((index, self.microbatch, torch.is_grad_enabled()))
        return self.model.layers[index](hidden_states).tanh()

    def postprocess(self, hidden_states):
        return self.model.head(hidden_states).square()


def _forward_step(data_iterator, model, *, return_schedule_plan=False):
    microbatch, tokens = next(data_iterator)
    plan = _Plan(model, microbatch, tokens)

    def loss_func(output):
        loss = output.sum()
        count = output.new_tensor(tokens.numel(), dtype=torch.int)
        return loss, count, {"loss": loss.detach()}

    if return_schedule_plan:
        return plan, loss_func
    hidden_states = plan.preprocess()
    for layer in range(plan.num_layers()):
        hidden_states = plan.forward_layer(layer, hidden_states)
    return plan.postprocess(hidden_states), loss_func


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("Megatron activation checkpointing requires CUDA RNG state")
    with torch.cuda.device(int(os.environ.get("LOCAL_RANK", "0"))):
        yield torch.device("cuda", torch.cuda.current_device())


def test_layer_by_layer_schedule_matches_no_pipeline_update(cuda_device):
    layer_major = get_forward_backward_func(pp_size=1, layer_by_layer=True)
    microbatch_major = get_forward_backward_func(pp_size=1)
    assert layer_major is layer_by_layer_forward_backward_no_pipelining
    assert microbatch_major is forward_backward_no_pipelining

    model = _Model().to(cuda_device)
    baseline = _Model().to(cuda_device)
    baseline.load_state_dict(model.state_dict())
    batches = [torch.tensor(ids, device=cuda_device) for ids in ((1, 2), (3, 4, 5))]
    single_rank_group = SimpleNamespace(size=lambda: 1)
    pg_collection = SimpleNamespace(tp=None, pp=single_rank_group, cp=single_rank_group)
    finalized = []
    accumulations = []
    for name, param in model.named_parameters():
        param.register_post_accumulate_grad_hook(lambda _, name=name: accumulations.append(name))

    def finalize(models, num_tokens, *, pg_collection, force_all_reduce):
        assert force_all_reduce
        finalized.append((models[0], num_tokens.item(), pg_collection))
        for param in models[0].parameters():
            param.grad.div_(num_tokens)

    def run(schedule, module):
        module.config.finalize_model_grads_func = finalize
        return schedule(
            forward_step_func=_forward_step,
            data_iterator=iter(enumerate(batches)),
            model=[module],
            num_microbatches=len(batches),
            seq_length=3,
            micro_batch_size=1,
            pg_collection=pg_collection,
            force_all_reduce=True,
        )

    losses = run(layer_major, model)
    expected_losses = run(microbatch_major, baseline)
    assert finalized == [(model, 5, pg_collection), (baseline, 5, pg_collection)]
    assert model.layer_calls == [
        (0, 0, False),
        (0, 1, False),
        (1, 0, False),
        (1, 1, False),
        (1, 0, True),
        (1, 1, True),
        (0, 0, True),
        (0, 1, True),
    ]
    assert Counter(accumulations) == {name: 1 for name, _ in model.named_parameters()}
    torch.testing.assert_close(losses, expected_losses)
    for param, ref_param in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(param.grad, ref_param.grad)

    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(baseline.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.state_dict(), baseline.state_dict())
