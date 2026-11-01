# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the forward / backward NVTX ranges of TransformerLayer."""

import types

import pytest
import torch

from megatron.core import utils as mcore_utils
from megatron.core.transformer.transformer_layer import TransformerLayer


@pytest.fixture
def nvtx_events(monkeypatch):
    """Record NVTX pushes / pops with profiling enabled and an empty range stack."""
    events = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda msg: events.append(f"push {msg}"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))
    monkeypatch.setattr(mcore_utils, "_nvtx_enabled", True)
    monkeypatch.setattr(mcore_utils, "_nvtx_range_messages", [])
    return events


def _fake_layer(label, weight, events):
    """A stand-in exposing only what ``TransformerLayer.forward`` reads."""

    def forward_attention(hidden_states, **kwargs):
        return hidden_states * weight, None

    def forward_mlp(hidden_states, *args, **kwargs):
        return _LogBackward.apply(events, f"{label} compute", hidden_states)

    return types.SimpleNamespace(
        _nvtx_fwd_label=f"{label} fwd",
        _nvtx_bwd_label=f"{label} bwd",
        _forward_attention=forward_attention,
        _forward_mlp=forward_mlp,
    )


class _LogBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, events, msg, tensor):
        ctx.events, ctx.msg = events, msg
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.events.append(ctx.msg)
        return None, None, grad


def test_layer_ranges_cover_forward_and_backward_in_order(nvtx_events):
    w0, w1 = torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)
    layers = [_fake_layer("L0", w0, nvtx_events), _fake_layer("L1", w1, nvtx_events)]
    hidden = torch.ones(4, requires_grad=True)
    out, _ = TransformerLayer.forward(layers[0], hidden)
    out, _ = TransformerLayer.forward(layers[1], hidden_states=out)
    assert nvtx_events == ["push L0 fwd", "pop", "push L1 fwd", "pop"]

    nvtx_events.clear()
    out.sum().backward()
    assert nvtx_events == ["push L1 bwd", "L1 compute", "pop", "push L0 bwd", "L0 compute", "pop"]
    assert mcore_utils._nvtx_range_messages == []
    assert torch.equal(hidden.grad, torch.full((4,), 6.0))


def test_layer_skips_backward_range_without_grad(nvtx_events):
    layer = _fake_layer("L0", torch.tensor(2.0), nvtx_events)
    with torch.no_grad():
        TransformerLayer.forward(layer, torch.ones(4, requires_grad=True))
    TransformerLayer.forward(layer, torch.ones(4))
    assert nvtx_events == ["push L0 fwd", "pop", "push L0 fwd", "pop"]


def test_layer_skips_backward_range_when_nvtx_disabled(nvtx_events, monkeypatch):
    monkeypatch.setattr(mcore_utils, "_nvtx_enabled", False)
    layer = _fake_layer("L0", torch.tensor(2.0, requires_grad=True), nvtx_events)
    out, _ = TransformerLayer.forward(layer, torch.ones(4, requires_grad=True))
    out.sum().backward()
    assert nvtx_events == ["L0 compute"]
