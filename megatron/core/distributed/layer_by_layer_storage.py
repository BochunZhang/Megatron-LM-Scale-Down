# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Tiered tensor storage and AdamW updates for layer-by-layer training."""

import mmap
import os
from pathlib import Path
from threading import local
from typing import Literal

import torch
from torch.optim._functional import adamw

try:
    import megatron_nvme_aio
except ImportError:
    megatron_nvme_aio = None


StorageTier = Literal["gpu", "cpu", "nvme"]


_ALIGNMENT = 4096
_DIRECT_IO = local()


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Layer-by-layer swap requires contiguous CPU tensors.")
    return tensor.detach().view(-1).view(torch.uint8)


def _direct_transfer(path: Path, payload: torch.Tensor, *, write: bool) -> None:
    size = ((payload.numel() + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT
    if not hasattr(_DIRECT_IO, "worker"):
        _DIRECT_IO.worker = megatron_nvme_aio.create_worker()
        _DIRECT_IO.buffer = torch.empty(0, dtype=torch.uint8, device="cpu")
    if _DIRECT_IO.buffer.numel() < size:
        _DIRECT_IO.mapping = mmap.mmap(-1, size)
        _DIRECT_IO.buffer = torch.frombuffer(_DIRECT_IO.mapping, dtype=torch.uint8)
    staging = _DIRECT_IO.buffer[:size]
    if write:
        staging[: payload.numel()].copy_(payload)
        staging[payload.numel() :].zero_()
    with memoryview(_DIRECT_IO.mapping)[:size] as buffer:
        megatron_nvme_aio.transfer(_DIRECT_IO.worker, buffer, str(path), write)
    if not write:
        payload.copy_(staging[: payload.numel()])


def _buffered_transfer(path: Path, payload: torch.Tensor, *, write: bool) -> None:
    # Keep a portable path for environments without the optional native extension.
    mode = "wb" if write else "rb"
    with path.open(mode, buffering=0) as file, memoryview(payload.numpy()) as buffer:
        if not write and os.fstat(file.fileno()).st_size != len(buffer):
            raise OSError(f"Unexpected layer-by-layer swap file size for {path}")
        transfer = file.write if write else file.readinto
        offset = 0
        while offset < len(buffer):
            count = transfer(buffer[offset:])
            if not count:
                raise OSError(f"Short layer-by-layer swap transfer for {path}")
            offset += count
        if write:
            getattr(os, "fdatasync", os.fsync)(file.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def read_tensor(path: Path, out: torch.Tensor) -> None:
    """Load a swap file into a preallocated contiguous CPU tensor."""
    payload = _byte_view(out)
    if megatron_nvme_aio is not None and payload.numel():
        _direct_transfer(path, payload, write=False)
    else:
        _buffered_transfer(path, payload, write=False)


def write_tensor(path: Path, value: torch.Tensor) -> None:
    """Write a contiguous CPU tensor to its swap file."""
    payload = _byte_view(value)
    if megatron_nvme_aio is not None and payload.numel():
        _direct_transfer(path, payload, write=True)
    else:
        _buffered_transfer(path, payload, write=True)


class OffloadedShard:
    """Store one rank's compute and optimizer shards in independent tiers.

    A caller serializes accesses to each shard. It can prefetch the optimizer
    state on one worker, then submit the update to another once the gradient is
    ready. Calls that use different shards can overlap.
    """

    def __init__(
        self,
        initial: torch.Tensor,
        directory: Path,
        *,
        parameter_tier: StorageTier,
        optimizer_tier: StorageTier,
        device: torch.device,
        weight_decay: float,
    ) -> None:
        if parameter_tier not in {"gpu", "cpu", "nvme"}:
            raise ValueError(f"Unknown layer-by-layer parameter tier {parameter_tier!r}")
        if optimizer_tier not in {"gpu", "cpu", "nvme"}:
            raise ValueError(f"Unknown layer-by-layer optimizer tier {optimizer_tier!r}")
        if "gpu" in (parameter_tier, optimizer_tier) and device.type != "cuda":
            raise ValueError("GPU layer-by-layer storage requires a CUDA device.")
        if (
            initial.ndim != 1
            or initial.dtype not in (torch.bfloat16, torch.float32)
            or not initial.is_contiguous()
        ):
            raise ValueError("Layer-by-layer compute shards must be contiguous flat BF16 or FP32 tensors.")
        self.shape = initial.shape
        self.dtype = initial.dtype
        self._weight_decay = weight_decay
        self._parameter_tier = parameter_tier
        self._optimizer_tier = optimizer_tier
        cpu = torch.device("cpu")
        parameter_device = device if parameter_tier == "gpu" else cpu
        optimizer_device = device if optimizer_tier == "gpu" else cpu
        self._state_step = torch.zeros((), dtype=torch.float32, device=optimizer_device)
        self._optimizer_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._parameter: torch.Tensor | None = None
        self._parameter_path = directory / "parameter"
        self._state_paths = (directory / "master", directory / "exp_avg", directory / "exp_avg_sq")
        self._has_moments = False
        if "nvme" in (parameter_tier, optimizer_tier):
            directory.mkdir(parents=True)

        initial = initial.detach()
        parameter = initial.to(device=parameter_device, copy=parameter_tier != "nvme")
        if parameter_tier == "nvme":
            write_tensor(self._parameter_path, parameter)
        else:
            self._parameter = parameter
        if parameter_device == optimizer_device:
            initial = parameter
        master = initial.to(device=optimizer_device, dtype=torch.float32, copy=True)
        if optimizer_tier == "nvme":
            write_tensor(self._state_paths[0], master)
        else:
            self._optimizer_state = master, torch.zeros_like(master), torch.zeros_like(master)

    @property
    def gpu_parameter(self) -> torch.Tensor | None:
        """Return the resident GPU parameter shard for direct all-gather."""
        return self._parameter if self._parameter_tier == "gpu" else None

    def read_parameter(self, out: torch.Tensor) -> None:
        """Stage the current compute-weight shard in caller-provided CPU memory."""
        if out.shape != self.shape or out.dtype != self.dtype:
            raise ValueError("Parameter staging tensor does not match its layer-by-layer shard.")
        _byte_view(out)
        if self._parameter is None:
            read_tensor(self._parameter_path, out)
        else:
            out.copy_(self._parameter)

    def prefetch_optimizer(self) -> None:
        """Stage NVMe optimizer tensors in CPU memory before the update."""
        if self._optimizer_state is not None:
            return
        master = torch.empty(self.shape, dtype=torch.float32, device="cpu")
        read_tensor(self._state_paths[0], master)
        average = torch.empty_like(master)
        square_average = torch.empty_like(master)
        if self._has_moments:
            read_tensor(self._state_paths[1], average)
            read_tensor(self._state_paths[2], square_average)
        else:
            average.zero_()
            square_average.zero_()
        self._optimizer_state = master, average, square_average

    def discard_prefetched_optimizer(self) -> None:
        """Release the staged NVMe state without applying an update."""
        if self._optimizer_tier == "nvme":
            self._optimizer_state = None

    @torch.no_grad()
    def step(
        self,
        gradient: torch.Tensor,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        found_inf: torch.Tensor | None = None,
    ) -> None:
        """Apply AdamW to a prefetched shard and publish its weights.

        A CUDA caller can supply the fused optimizer's nonfinite flag to skip
        the update on-device.
        """
        if gradient.shape != self.shape or not gradient.is_contiguous():
            raise ValueError("Gradient tensor does not match its layer-by-layer shard.")
        if self._optimizer_state is None:
            raise RuntimeError("Layer-by-layer AdamW step requires a prefetched shard.")
        master, average, square_average = self._optimizer_state
        if gradient.device != master.device or gradient.dtype != torch.float32:
            raise ValueError(f"Layer-by-layer AdamW expects an FP32 gradient on {master.device}.")
        adamw(
            [master],
            [gradient],
            [average],
            [square_average],
            [],
            [self._state_step],
            fused=True,
            found_inf=found_inf,
            amsgrad=False,
            beta1=betas[0],
            beta2=betas[1],
            lr=lr,
            weight_decay=self._weight_decay,
            eps=eps,
            maximize=False,
        )
        if self._optimizer_tier == "nvme":
            for path, state in zip(self._state_paths, self._optimizer_state):
                write_tensor(path, state)
            self._has_moments = True
            self._optimizer_state = None
        if self._parameter is None:
            write_tensor(self._parameter_path, master.to(device="cpu", dtype=self.dtype))
        else:
            self._parameter.copy_(master)
