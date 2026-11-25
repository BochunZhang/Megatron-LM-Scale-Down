# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    BucketStatus,
    PrefetchOrder,
)


@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_nvme_weight_lookahead_and_clean_release(monkeypatch, backward):
    order = PrefetchOrder.BACKWARD_PASS_ORDER if backward else PrefetchOrder.FORWARD_PASS_ORDER
    first, next_bucket = (1, 0) if backward else (0, 1)
    params = [object(), object()]
    accesses = []
    nvme = object()
    groups = []
    for bucket_id in range(2):
        weight = SimpleNamespace(
            bucket_index=SimpleNamespace(size=1),
            cpu_offload=True,
            is_data_distributed=False,
            nvme=nvme,
            prefetch_cpu_data=Mock(),
            offload_to_cpu=Mock(),
            free_bucket_storage=Mock(),
            data_parallel_group=None,
        )
        weight.prefetch_cpu_data.side_effect = lambda b=bucket_id: accesses.append(("prefetch", b))
        groups.append(
            SimpleNamespace(
                fsdp_unit_id=bucket_id, model_weight_buffer=weight, transpose_weight_buffer=None
            )
        )

    pipeline = AllGatherPipeline.__new__(AllGatherPipeline)
    pipeline.ag_stream = Mock()
    pipeline.buffer = SimpleNamespace(
        ddp_config=SimpleNamespace(
            fsdp_nvme_weights_only=True,
            fsdp_double_buffer=True,
            fsdp_cpu_offload_weight=True,
            outer_dp_sharding_strategy="no_shard",
        ),
        param_to_param_group={param: bucket_id for bucket_id, param in enumerate(params)},
        parameter_groups=groups,
        num_buckets=2,
        double_buf_units={0, 1},
        bucket_to_bucket_group={0: [0], 1: [1]},
        nvme=nvme,
    )
    pipeline.bucket_to_bucket_group = {0: 0, 1: 1}
    pipeline.bucket_status = {(i, False): BucketStatus.EMPTY for i in range(2)}
    pipeline.bucket_can_be_released = {(i, False): False for i in range(2)}
    pipeline.keep_persistent_buckets = False
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    def gather(bucket_id, bwd):
        accesses.append(("gather", bucket_id))
        pipeline.bucket_status[pipeline.get_bucket_key(bucket_id, bwd)] = BucketStatus.READY_TO_USE

    pipeline.async_bucket_gather = gather
    pipeline.all_gather_params([params[first]], prefetch=True, prefetch_order=order, bwd=backward)

    assert accesses == [("gather", first), ("prefetch", next_bucket)]
    next_key = pipeline.get_bucket_key(next_bucket, backward)
    assert pipeline.bucket_status[next_key] == BucketStatus.EMPTY

    weight = groups[first].model_weight_buffer
    pipeline.release_bucket(first, backward, lazy=backward)
    pipeline.recycle_unused_buckets()
    weight.offload_to_cpu.assert_not_called()
    weight.free_bucket_storage.assert_called_once_with()
    assert pipeline.bucket_status[pipeline.get_bucket_key(first, backward)] == BucketStatus.EMPTY

    pipeline.all_gather_params(
        [params[next_bucket]], prefetch=True, prefetch_order=order, bwd=backward
    )
    assert accesses == [("gather", first), ("prefetch", next_bucket), ("gather", next_bucket)]
