# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Host staging and native NVMe AIO for Megatron-FSDP offload."""

from __future__ import annotations

import mmap
import tempfile
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import torch

from .utils import nvtx_range

if TYPE_CHECKING:
    from .param_and_grad_buffer import DataParallelBuffer


class NVMeOffload:
    """Store CPU tensor payloads in per-process direct-I/O swap files."""

    def __init__(self, path: str):
        try:
            import megatron_nvme_aio
        except ImportError as error:
            raise RuntimeError(
                "NVMe offload requires the optional megatron_nvme_aio extension. "
                "Build and install Megatron-LM's src/nvme/deepspeed in the training environment."
            ) from error

        self.transfer = megatron_nvme_aio.transfer
        self.worker = megatron_nvme_aio.create_worker()

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(
            dir=root, prefix=f"mfsdp-rank{torch.distributed.get_rank()}-"
        )
        self.lock = Lock()
        self.reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fsdp-nvme")
        self.io_storage = None
        self.io_buffer = torch.empty(0, dtype=torch.uint8, device="cpu")

    def _transfer(self, key: str, tensor: torch.Tensor, *, write: bool) -> None:
        if tensor.numel() == 0:
            return
        filename = str(Path(self.directory.name) / key)
        payload = tensor.view(torch.uint8).flatten()
        # O_DIRECT needs an aligned address and transfer length.
        size = ((payload.numel() + 4095) // 4096) * 4096
        with (
            self.lock,
            nvtx_range(lambda: f"fsdp.nvme.{'write' if write else 'read'} {key} bytes={size}"),
        ):
            if self.io_buffer.numel() < size:
                # Anonymous mmap gives the native worker page-aligned scratch storage.
                self.io_storage = mmap.mmap(-1, size)
                self.io_buffer = torch.frombuffer(self.io_storage, dtype=torch.uint8)
            staging = self.io_buffer[:size]
            if write:
                staging[: payload.numel()].copy_(payload)
                staging[payload.numel() :].zero_()
            self.transfer(self.worker, memoryview(self.io_storage)[:size], filename, write)
            if not write:
                payload.copy_(staging[: payload.numel()])

    def write(self, key: str, tensor: torch.Tensor) -> None:
        """Write a contiguous CPU tensor to its swap file."""
        self._transfer(key, tensor, write=True)

    def read(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Read a swap file into a contiguous CPU tensor."""
        self._transfer(key, tensor, write=False)
        return tensor

    def prefetch(self, key: str, tensor: torch.Tensor) -> Future[torch.Tensor]:
        """Start a CPU-side read while the caller works on the current bucket."""
        return self.reader.submit(self.read, key, tensor)


class CPUGradientAccumulator:
    """Reuse a pinned MaxPool double buffer sized for the largest shard of each dtype."""

    def __init__(self, sizes: dict[torch.dtype, int]):
        self.slots = {
            dtype: [torch.empty(size, dtype=dtype, device="cpu", pin_memory=True) for _ in range(2)]
            for dtype, size in sizes.items()
        }
        self.next_slot = defaultdict(int)
        self.pending = {}
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fsdp-cpu-grad")

    def submit(
        self, grad: torch.Tensor, buffer: DataParallelBuffer, stream: torch.cuda.Stream
    ) -> torch.cuda.Event:
        """Queue a D2H transfer and accumulate after its event signals."""
        dtype = grad.dtype
        slot_id = self.next_slot[dtype]
        self.next_slot[dtype] = 1 - slot_id
        key = (dtype, slot_id)
        if key in self.pending:
            # The slot stays leased until both DMA and CPU accumulation have finished.
            self.pending.pop(key).result()
        slot = self.slots[dtype][slot_id][: grad.numel()]
        with (
            nvtx_range(lambda: f"fsdp.cpu_grad.d2h bucket={buffer.bucket_id}"),
            torch.cuda.stream(stream),
        ):
            slot.copy_(grad, non_blocking=True)
            event = stream.record_event()
        grad.record_stream(stream)
        valid = buffer.cpu_data_valid
        buffer.cpu_data_valid = True
        buffer.cpu_data_reduced = True
        self.pending[key] = self.worker.submit(self._accumulate, slot, buffer, event, valid)
        return event

    @staticmethod
    @torch.no_grad()
    def _accumulate(slot, buffer, event, valid):
        with nvtx_range(lambda: f"fsdp.cpu_grad.wait_d2h bucket={buffer.bucket_id}"):
            event.synchronize()
        with nvtx_range(lambda: f"fsdp.cpu_grad.accumulate bucket={buffer.bucket_id}"):
            if valid:
                main_grad = buffer.read_cpu_data()
                main_grad.add_(slot)
            else:
                main_grad = buffer.new_cpu_data()
                main_grad.copy_(slot)
            buffer.store_cpu_data(main_grad)

    def wait(self) -> None:
        """Drain the worker before finalizing gradients or starting another iteration."""
        for task in self.pending.values():
            task.result()
        self.pending.clear()
