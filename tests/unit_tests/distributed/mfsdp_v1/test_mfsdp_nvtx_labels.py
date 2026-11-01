# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the layer labels Megatron-FSDP attaches to its collectives in NVTX ranges."""

import types

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import utils as mfsdp_utils
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    GradReducePipeline,
    _parameter_group_nvtx_label,
)


@pytest.fixture
def fake_buffer():
    """A stand-in exposing only the parameter groups, labelled as ``ParamAndGradBuffer`` does."""
    names = [
        "module.decoder.layers.10.self_attention.linear_qkv.weight",
        "module.decoder.layers.2.mlp.experts.linear_fc1.weight0",
        "module.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight",
        "module.embedding.word_embeddings.weight",
        "module.decoder.layers.2.mlp.router.weight",
    ]
    params = [
        torch.nn.Parameter(torch.empty(n)) for n in (2_000_000, 3_000_000, 1_000_000, 500_000, 100)
    ]
    param_to_name = dict(zip(params, names))
    groups = [types.SimpleNamespace(params=[p]) for p in params[:4]]
    groups.append(types.SimpleNamespace(params=[params[1], params[4]]))
    for group in groups:
        group.nvtx_label = _parameter_group_nvtx_label(group.params, param_to_name)
    return types.SimpleNamespace(parameter_groups=groups)


def test_parameter_group_nvtx_label_names_layer_kind_and_size(fake_buffer):
    labels = [group.nvtx_label for group in fake_buffer.parameter_groups]
    assert labels == [
        "L10 dense 2.0M",
        "L2 expert 3.0M",
        "mtp0 dense 1.0M",
        "embedding dense 0.5M",
        "L2 mixed 3.0M",
    ]


def test_all_gather_nvtx_label_separates_prefetched_buckets(fake_buffer):
    pipeline = types.SimpleNamespace(buffer=fake_buffer, _nvtx_labels={})
    label = AllGatherPipeline.all_gather_nvtx_label(pipeline, [0, 1, 2], {0}, bwd=True)
    assert label == (
        "FSDP AG bwd L10 dense 2.0M | FSDP AG bwd prefetch L2 expert 3.0M + mtp0 dense 1.0M"
    )
    # The joined label is cached per combination; a later lookup does not rebuild it.
    fake_buffer.parameter_groups[0].nvtx_label = "changed"
    assert AllGatherPipeline.all_gather_nvtx_label(pipeline, [0, 1, 2], {0}, bwd=True) == label
    assert AllGatherPipeline.all_gather_nvtx_label(pipeline, [0], {0}, bwd=False) == (
        "FSDP AG fwd changed"
    )


def test_grad_reduce_nvtx_label_joins_buckets(fake_buffer):
    pipeline = types.SimpleNamespace(buffer=fake_buffer, _nvtx_collective="RS", _nvtx_labels={})
    assert GradReducePipeline.grad_reduce_nvtx_label(pipeline, [1, 4]) == (
        "FSDP RS L2 expert 3.0M + L2 mixed 3.0M"
    )
    assert pipeline._nvtx_labels == {(1, 4): "FSDP RS L2 expert 3.0M + L2 mixed 3.0M"}


def test_nvtx_range_only_builds_label_when_enabled(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda msg: events.append(f"push {msg}"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))

    def message():
        events.append("built")
        return "FSDP RS L0 dense 1.0M"

    monkeypatch.setattr(mfsdp_utils, "_nvtx_enabled", False)
    with mfsdp_utils.nvtx_range(message):
        events.append("body")
    assert events == ["body"]

    events.clear()
    monkeypatch.setattr(mfsdp_utils, "_nvtx_enabled", True)
    with mfsdp_utils.nvtx_range(message):
        events.append("body")
    assert events == ["built", "push FSDP RS L0 dense 1.0M", "body", "pop"]


def test_mcore_configure_nvtx_profiling_forwards_to_megatron_fsdp(monkeypatch):
    from megatron.core import utils as mcore_utils

    monkeypatch.setattr(mfsdp_utils, "_nvtx_enabled", False)
    monkeypatch.setattr(mcore_utils, "_nvtx_enabled", False)
    mcore_utils.configure_nvtx_profiling(True)
    assert mfsdp_utils._nvtx_enabled is True
    mcore_utils.configure_nvtx_profiling(False)
    assert mfsdp_utils._nvtx_enabled is False
