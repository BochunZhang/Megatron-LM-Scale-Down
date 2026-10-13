# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.hybrid_ep_sm_sync import HAVE_TRITON, HostCountsReader
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="needs CUDA and Triton")
class TestHostCountsReader:
    def test_reads_values_after_preceding_kernels(self):
        reader = HostCountsReader(capacity=4)
        for num_values in (1, 4, 16, 33):
            src = torch.arange(num_values, device="cuda", dtype=torch.int64)
            # Queue enough work ahead of the copy that a read racing the stream would
            # observe the pre-update values.
            for _ in range(8):
                src = src * 1 + 1
            values = reader.read(src)
            assert values.device.type == "cpu"
            assert values.tolist() == (torch.arange(num_values) + 8).tolist()

    def test_returned_tensor_is_not_overwritten_by_next_read(self):
        reader = HostCountsReader()
        first = reader.read(torch.full((8,), 3, device="cuda", dtype=torch.int64))
        reader.read(torch.full((8,), 7, device="cuda", dtype=torch.int64))
        assert first.tolist() == [3] * 8

    def test_rejects_non_int64_input(self):
        reader = HostCountsReader()
        with pytest.raises(AssertionError):
            reader.read(torch.zeros(4, device="cuda", dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("sync_with_sm", [False, True])
def test_setup_metadata_pads_to_group_max(monkeypatch, sync_with_sm):
    from types import SimpleNamespace

    from megatron.core.transformer.moe import hybrid_ep_sm_sync
    from megatron.core.transformer.moe.token_dispatcher import (
        HYBRIDEP_TOKEN_ALIGNMENT,
        _HybridEPManager,
    )

    group_max = 100

    def fake_all_reduce(tensor, op=None, group=None):
        tensor.fill_(group_max)

    sm_reads = []
    real_read = hybrid_ep_sm_sync.read_device_counts_with_sm

    def spy_read(src):
        sm_reads.append(src.numel())
        return real_read(src)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(hybrid_ep_sm_sync, "read_device_counts_with_sm", spy_read)

    # Bypass __init__, which requires the HybridEP extension.
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=True, moe_hybridep_sync_with_sm=sync_with_sm
    )
    manager.group = None
    manager.num_experts = 8
    manager.moe_expert_rank_capacity_factor = None
    manager.drop_and_pad = False

    num_tokens = 37
    routing_map = torch.zeros(num_tokens, 8, dtype=torch.bool, device="cuda")
    probs = torch.zeros(num_tokens, 8, dtype=torch.float32, device="cuda")
    manager.setup_metadata(routing_map, probs)

    expected = group_max + (-group_max % HYBRIDEP_TOKEN_ALIGNMENT)
    assert manager._padded_num_tokens == expected
    assert manager.routing_map.shape == (expected, 8)
    assert manager.token_probs.shape == (expected, 8)
    assert sm_reads == ([1] if sync_with_sm else [])


def test_sync_with_sm_requires_hybridep_backend():
    with pytest.raises(ValueError, match="moe_hybridep_sync_with_sm"):
        TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_token_dispatcher_type="alltoall",
            moe_hybridep_sync_with_sm=True,
        )
