# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-FSDP weight buffers for the layer-major training schedule."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist

from .fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    DataParallelBuffer,
    MaxPoolAllocator,
    ParameterGroup,
)
from .fsdp.src.megatron_fsdp.utils import GlobalMemoryBuffer

if TYPE_CHECKING:
    from .layer_by_layer import _Unit


class _FSDPWeightSlots:
    """Stage pinned CPU weight shards into Megatron-FSDP's two GPU buffer slots."""

    def __init__(
        self,
        units: Sequence[_Unit],
        device: torch.device,
        group: dist.ProcessGroup,
        *,
        initialized: torch.cuda.Event,
    ) -> None:
        self.device = device
        self.group = group
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_event(initialized)
        self.stream.wait_stream(torch.cuda.current_stream(device))

        groups = []
        sizes = {}
        self.buffers: list[list[DataParallelBuffer]] = []
        for unit_id, unit in enumerate(units):
            buffers = []
            for bucket in unit.buckets:
                buffer = bucket.weight_buffer
                assert buffer is not None and buffer.bucket_id == len(groups)
                groups.append(
                    ParameterGroup(
                        params=bucket.params,
                        dtype=bucket.key[0],
                        fsdp_unit_id=unit_id,
                    )
                )
                sizes[buffer.bucket_id] = buffer.bucket_index.size
                buffers.append(buffer)
            self.buffers.append(buffers)
        self.pool = MaxPoolAllocator(
            name="layer_by_layer_fsdp_weights",
            fsdp_param_groups=groups,
            size=2,
            memory_buffer=GlobalMemoryBuffer(),
            bucket_sizes=sizes,
        )
        for buffers in self.buffers:
            for buffer in buffers:
                buffer.temporary_bucket_allocator = self.pool

        self.placeholders = {
            dtype: torch.empty(0, dtype=dtype, device=device) for dtype in {g.dtype for g in groups}
        }
        self.pending: tuple[int, list[Bucket]] | None = None
        self.ready: dict[int, tuple[torch.cuda.Event, list[Bucket]]] = {}
        self.released: torch.cuda.Event | None = None
        self.reloads = 0

    def prefetch(self, index: int, _unit: _Unit) -> None:
        """Asynchronously reload the next CPU shard into its full GPU bucket."""
        assert self.pending is None
        with (
            torch.cuda.nvtx.range(f"layer_by_layer.parameter_h2d.{index}"),
            torch.cuda.stream(self.stream),
        ):
            if self.released is not None:
                self.stream.wait_event(self.released)
            weights = []
            for buffer in self.buffers[index]:
                full = buffer.fetch_bucket(strict_assignments=False)
                buffer.reload_from_cpu(buffer.get_shard_from_bucket(full), self.stream)
                weights.append(full)
        self.pending = index, weights
        self.reloads += 1

    def stage(self) -> None:
        """The prefetch has already submitted all host-to-device copies."""

    def progress(self) -> None:
        """Enqueue all-gathers independently of H2D to preserve collective order."""
        if self.pending is None:
            return
        index, weights = self.pending
        with (
            torch.cuda.nvtx.range(f"layer_by_layer.parameter_all_gather.{index}"),
            torch.cuda.stream(self.stream),
        ):
            for buffer, full in zip(self.buffers[index], weights, strict=True):
                dist.all_gather_into_tensor(
                    full.data, buffer.get_shard_from_bucket(full), group=self.group
                )
            self.ready[index] = self.stream.record_event(), weights
        self.pending = None

    def bind(self, index: int, unit: _Unit) -> None:
        """Make the gathered parameters available to all micro-batches of a unit."""
        self.progress()
        ready, weights = self.ready.pop(index)
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(ready)
        for bucket, full in zip(unit.buckets, weights, strict=True):
            full.data.record_stream(compute)
            bucket.bind(full.data)

    def release(self, index: int, unit: _Unit) -> None:
        """Return the unit's buffers after its last compute use."""
        self.released = torch.cuda.current_stream(self.device).record_event()
        for bucket, buffer in zip(unit.buckets, self.buffers[index], strict=True):
            bucket.unbind(self.placeholders[bucket.key[0]])
            buffer.free_bucket_storage()
