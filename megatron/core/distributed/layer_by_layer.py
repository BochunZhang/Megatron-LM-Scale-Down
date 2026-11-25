# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-major training with sharded weights, gradients, and CPU or GPU AdamW."""

import math
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch import distributed as dist
from torch import nn

from .layer_by_layer_storage import OffloadedShard, read_tensor, write_tensor

_GroupKey = tuple[torch.dtype, bool]


@dataclass
class _Bucket:
    key: _GroupKey
    params: list[nn.Parameter]
    shapes: list[torch.Size]
    sizes: list[int]
    shard_size: int
    shard: OffloadedShard

    @classmethod
    def create(
        cls, key, params, directory, *, rank, world, tier, optimizer_tier, device, weight_decay
    ):
        shapes = [param.shape for param in params]
        sizes = [param.numel() for param in params]
        shard_size = (sum(sizes) + world - 1) // world
        start = rank * shard_size
        end = start + shard_size
        shard_device = device if "gpu" in (tier, optimizer_tier) else "cpu"
        initial = torch.zeros(shard_size, dtype=key[0], device=shard_device)
        offset = 0
        for param, size in zip(params, sizes):
            lo = max(start, offset)
            hi = min(end, offset + size)
            if lo < hi:
                initial[lo - start : hi - start].copy_(
                    param.detach().view(-1)[lo - offset : hi - offset]
                )
            offset += size
        shard = OffloadedShard(
            initial,
            directory,
            parameter_tier=tier,
            optimizer_tier=optimizer_tier,
            device=device,
            weight_decay=weight_decay if key[1] else 0.0,
        )
        return cls(key, params, shapes, sizes, shard_size, shard)

    def bind(self, full: Tensor) -> None:
        for param, flat, shape in zip(
            self.params, full[: sum(self.sizes)].split(self.sizes), self.shapes
        ):
            param.data = flat.view(shape)

    def unbind(self, placeholder: Tensor) -> None:
        for param in self.params:
            param.data = placeholder

    def accumulate(self, main: Tensor, grads: Sequence[Tensor]) -> None:
        for flat, grad in zip(main[: sum(self.sizes)].split(self.sizes), grads):
            flat.add_(grad.reshape(-1))


@dataclass
class _Unit:
    module: nn.Module
    buckets: list[_Bucket]

    @property
    def params(self) -> tuple[nn.Parameter, ...]:
        return tuple(param for bucket in self.buckets for param in bucket.params)


class _WeightSlots:
    def __init__(self, units, device, group, reader, *, parameter_tier, initialized):
        self.group = group
        self.reader = reader
        self.gpu_parameters = parameter_tier == "gpu"
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_event(initialized)
        world = group.size()
        sizes: dict[_GroupKey, int] = {}
        for unit in units:
            for bucket in unit.buckets:
                sizes[bucket.key] = max(sizes.get(bucket.key, 0), bucket.shard_size)
        self.host = []
        self.local = []
        if not self.gpu_parameters:
            for _ in range(2):
                self.host.append(
                    {
                        key: torch.empty(n, dtype=key[0], device="cpu", pin_memory=True)
                        for key, n in sizes.items()
                    }
                )
                self.local.append(
                    {key: torch.empty(n, dtype=key[0], device=device) for key, n in sizes.items()}
                )
        self.full = [
            {key: torch.empty(n * world, dtype=key[0], device=device) for key, n in sizes.items()}
            for _ in range(2)
        ]
        self.placeholders = {key: torch.empty(0, dtype=key[0], device=device) for key in sizes}
        self.copied: list[torch.cuda.Event | None] = [None, None]
        self.released: list[torch.cuda.Event | None] = [None, None]
        self.ready: list[torch.cuda.Event | None] = [None, None]
        self.pending: tuple[int, _Unit, Future | None] | None = None
        self.stream.wait_stream(torch.cuda.current_stream(device))

    def prefetch(self, index: int, unit: _Unit) -> None:
        slot = index % 2
        assert self.pending is None
        if self.gpu_parameters:
            self.pending = (slot, unit, None)
            return
        copied = self.copied[slot]

        def read() -> None:
            if copied is not None:
                copied.synchronize()
            for bucket in unit.buckets:
                bucket.shard.read_parameter(self.host[slot][bucket.key][: bucket.shard_size])

        self.pending = (slot, unit, self.reader.submit(read))

    def stage(self, *, block: bool = False) -> None:
        """Copy the prefetched shard to its GPU slot without issuing a collective."""
        if self.pending is None:
            return
        slot, unit, future = self.pending
        if future is None or (not block and not future.done()):
            return
        future.result()
        with torch.cuda.stream(self.stream):
            for bucket in unit.buckets:
                size = bucket.shard_size
                self.local[slot][bucket.key][:size].copy_(
                    self.host[slot][bucket.key][:size], non_blocking=True
                )
            self.copied[slot] = self.stream.record_event()
        self.pending = (slot, unit, None)

    def progress(self, *, block: bool = False) -> None:
        self.stage(block=block)
        if self.pending is None:
            return
        slot, unit, future = self.pending
        if future is not None:
            return
        with torch.cuda.stream(self.stream):
            if self.released[slot] is not None:
                self.stream.wait_event(self.released[slot])
            for bucket in unit.buckets:
                size = bucket.shard_size
                source = bucket.shard.gpu_parameter
                if source is None:
                    source = self.local[slot][bucket.key][:size]
                dist.all_gather_into_tensor(
                    self.full[slot][bucket.key][: size * self.group.size()],
                    source,
                    group=self.group,
                )
            self.ready[slot] = self.stream.record_event()
        self.pending = None

    def bind(self, index: int, unit: _Unit) -> None:
        self.progress(block=True)
        slot = index % 2
        ready = self.ready[slot]
        assert ready is not None
        torch.cuda.current_stream().wait_event(ready)
        for bucket in unit.buckets:
            bucket.bind(self.full[slot][bucket.key])

    def release(self, index: int, unit: _Unit) -> None:
        for bucket in unit.buckets:
            bucket.unbind(self.placeholders[bucket.key])
        self.released[index % 2] = torch.cuda.current_stream().record_event()


class _Activations:
    def __init__(self, example, directory, reader, writer, *, nvme):
        self.directory = directory
        self.reader = reader
        self.writer = writer
        self.nvme = nvme
        self.h2d = torch.cuda.Stream(device=example.device)
        self.d2h = torch.cuda.Stream(device=example.device)
        self.host_inputs = [self._pinned(example) for _ in range(2)] if nvme else []
        self.host_outputs = [torch.empty_like(x, pin_memory=True) for x in self.host_inputs]
        self.inputs = [torch.empty_like(example) for _ in range(2)]
        self.copied: list[torch.cuda.Event | None] = [None, None]
        self.used: list[torch.cuda.Event | None] = [None, None]
        self.saved: dict[tuple[int, int], tuple[Tensor, torch.cuda.Event]] = {}
        self.writes: dict[Path, Future] = {}
        self.write_slots: list[Future | None] = [None, None]
        self.pending: tuple[int, bool, Future | None] | None = None
        self.offloaded = 0
        self.reloaded = 0

    @staticmethod
    def _pinned(example: Tensor) -> Tensor:
        return torch.empty(example.shape, dtype=example.dtype, device="cpu", pin_memory=True)

    def _path(self, index, microbatch):
        return self.directory / f"input-{index}-{microbatch}.bin"

    def store(self, index: int, microbatch: int, value: Tensor) -> None:
        if value.shape != self.inputs[0].shape or value.dtype != self.inputs[0].dtype:
            raise ValueError("Layer boundaries must share the same activation shape and dtype")
        slot = microbatch % 2
        if self.nvme:
            if self.write_slots[slot] is not None:
                self.write_slots[slot].result()
            host = self.host_outputs[slot]
        else:
            host = self._pinned(value)
        compute = torch.cuda.current_stream(value.device)
        with torch.cuda.stream(self.d2h):
            self.d2h.wait_stream(compute)
            host.copy_(value, non_blocking=True)
            value.record_stream(self.d2h)
            copied = self.d2h.record_event()
        if not self.nvme:
            self.saved[index, microbatch] = (host, copied)
            return
        path = self._path(index, microbatch)

        def write() -> None:
            copied.synchronize()
            write_tensor(path, host)

        self.writes[path] = self.write_slots[slot] = self.writer.submit(write)

    def prefetch(self, index: int, microbatch: int, *, backward: bool) -> None:
        assert self.pending is None
        slot = microbatch % 2
        if not self.nvme:
            key = (index, microbatch)
            host, ready = self.saved.pop(key) if backward else self.saved[key]
            self._reload(slot, host, ready)
            self.pending = (slot, backward, None)
            return
        path = self._path(index, microbatch)
        written = self.writes.pop(path) if backward else self.writes[path]
        copied = self.copied[slot]

        def read() -> None:
            written.result()
            if copied is not None:
                copied.synchronize()
            read_tensor(path, self.host_inputs[slot])
            if backward:
                path.unlink()

        self.pending = (slot, backward, self.reader.submit(read))

    def _reload(self, slot: int, host: Tensor, ready: torch.cuda.Event | None = None) -> None:
        with torch.cuda.stream(self.h2d):
            if self.used[slot] is not None:
                self.h2d.wait_event(self.used[slot])
            if ready is not None:
                self.h2d.wait_event(ready)
            self.inputs[slot].copy_(host, non_blocking=True)
            self.copied[slot] = self.h2d.record_event()

    def acquire(self) -> Tensor:
        assert self.pending is not None
        slot, backward, future = self.pending
        if future is not None:
            future.result()
            self._reload(slot, self.host_inputs[slot])
        torch.cuda.current_stream().wait_event(self.copied[slot])
        self.pending = None
        # The first consumer is forward. Backward reloads the same offload.
        self.offloaded += int(not backward)
        self.reloaded += 1
        return self.inputs[slot]

    def release(self, microbatch: int) -> None:
        self.used[microbatch % 2] = torch.cuda.current_stream().record_event()


class _GpuActivations:
    def __init__(self):
        self.saved: dict[tuple[int, int], Tensor] = {}
        self.pending: Tensor | None = None

    def store(self, index: int, microbatch: int, value: Tensor) -> None:
        self.saved[index, microbatch] = value

    def prefetch(self, index: int, microbatch: int, *, backward: bool) -> None:
        assert self.pending is None
        key = (index, microbatch)
        self.pending = self.saved.pop(key) if backward else self.saved[key]

    def acquire(self) -> Tensor:
        value = self.pending
        assert value is not None
        self.pending = None
        return value

    def release(self, microbatch: int) -> None:
        pass


def _reduce_gradients(
    unit: _Unit, main_grads: list[Tensor], group: dist.ProcessGroup
) -> list[Tensor]:
    stream = torch.cuda.current_stream()
    reduced_grads = []
    for bucket, full in zip(unit.buckets, main_grads):
        reduced = torch.empty(bucket.shard_size, dtype=torch.float32, device=full.device)
        dist.reduce_scatter_tensor(reduced, full, group=group)
        full.record_stream(stream)
        reduced.div_(group.size())
        reduced_grads.append(reduced)
    return reduced_grads


class _CpuUpdates:
    def __init__(self, units, device, group, reader, updater, *, betas, eps):
        self.group = group
        self.device = device
        self.reader = reader
        self.updater = updater
        self.betas = betas
        self.eps = eps
        self.stream = torch.cuda.Stream(device=device)
        size = max(sum(bucket.shard_size for bucket in unit.buckets) for unit in units)
        self.host = [
            torch.empty(size, dtype=torch.float32, device="cpu", pin_memory=True) for _ in range(2)
        ]
        self.updates: list[Future | None] = [None, None]
        self.invalid_unit = -1

    def prefetch(self, index: int, unit: _Unit) -> Future:
        reuse = self.updates[index % 2]

        def read() -> None:
            # Bound prefetched optimizer state without blocking GPU submission.
            if reuse is not None:
                reuse.result()
            for bucket in unit.buckets:
                bucket.shard.prefetch_optimizer()

        return self.reader.submit(read)

    def submit(
        self, index: int, unit: _Unit, main_grads: list[Tensor], prefetched: Future, *, lr: float
    ) -> None:
        slot = index % 2
        n = sum(bucket.shard_size for bucket in unit.buckets)
        host_grads = self.host[slot][:n].split([bucket.shard_size for bucket in unit.buckets])
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(compute)
            reduced_grads = _reduce_gradients(unit, main_grads, self.group)

        # The previous update must finish reading this pinned slot before D2H.
        reuse = self.updates[slot]
        if reuse is not None:
            reuse.result()
        with torch.cuda.stream(self.stream):
            for host, reduced in zip(host_grads, reduced_grads):
                host.copy_(reduced, non_blocking=True)
            copied = self.stream.record_event()

        def update() -> None:
            prefetched.result()
            copied.synchronize()
            for grad in host_grads:
                if not grad.numel():
                    continue
                low, high = grad.aminmax()
                if not (math.isfinite(low.item()) and math.isfinite(high.item())):
                    self.invalid_unit = max(self.invalid_unit, index)
                    for bucket in unit.buckets:
                        bucket.shard.discard_prefetched_optimizer()
                    return
            for bucket, grad in zip(unit.buckets, host_grads):
                bucket.shard.step(grad, lr=lr, betas=self.betas, eps=self.eps)

        self.updates[slot] = self.updater.submit(update)

    def finish(self) -> None:
        for slot, update in enumerate(self.updates):
            if update is not None:
                update.result()
                self.updates[slot] = None
        invalid = torch.tensor(self.invalid_unit, dtype=torch.int64, device=self.device)
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.group)
        self.invalid_unit = invalid.item()
        if self.invalid_unit >= 0:
            raise FloatingPointError(f"Non-finite training gradient at unit {self.invalid_unit}")


class _GpuUpdates:
    def __init__(self, device, group, *, betas, eps, initialized):
        self.group = group
        self.device = device
        self.betas = betas
        self.eps = eps
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_event(initialized)
        with torch.cuda.stream(self.stream):
            self.invalid = torch.full((), -1, dtype=torch.int64, device=device)
        self.invalid_unit = -1

    def prefetch(self, _index: int, _unit: _Unit) -> None:
        pass

    def submit(
        self, index: int, unit: _Unit, main_grads: list[Tensor], _prefetched: None, *, lr: float
    ) -> None:
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(compute)
            reduced_grads = _reduce_gradients(unit, main_grads, self.group)
            finite = [
                torch.isfinite(torch.stack(grad.aminmax())).all()
                for grad in reduced_grads
                if grad.numel()
            ]
            if not finite:
                return
            found_inf = torch.stack(finite).all().logical_not().to(torch.float32)
            torch.maximum(
                self.invalid, found_inf.to(torch.int64) * (index + 1) - 1, out=self.invalid
            )
            for bucket, grad in zip(unit.buckets, reduced_grads):
                bucket.shard.step(grad, lr=lr, betas=self.betas, eps=self.eps, found_inf=found_inf)

    def finish(self) -> None:
        # Do not start the next step's all-gathers before the parameters are published.
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        dist.all_reduce(self.invalid, op=dist.ReduceOp.MAX, group=self.group)
        self.invalid_unit = self.invalid.item()
        if self.invalid_unit >= 0:
            raise FloatingPointError(f"Non-finite training gradient at unit {self.invalid_unit}")


class LayerByLayerTrainer:
    """Train an untied, dropout-free sequence of embedding, decoder and loss units.

    Each rank supplies the same initialized units and its own equally sized
    batches. The last unit returns a scalar loss. All other units return new
    hidden-state tensors of the same shape and dtype.
    The trainer owns its worker threads and uses the caller's scratch directory.
    """

    def __init__(
        self,
        modules: Iterable[nn.Module],
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        directory: Path,
        tier: str,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        activation_tier: str = "nvme",
        optimizer_tier: str = "nvme",
    ):
        tier = tier.removesuffix("_param")
        activation_tier = activation_tier.removesuffix("_act")
        for component, storage in (
            ("parameter", tier),
            ("activation", activation_tier),
            ("optimizer", optimizer_tier),
        ):
            if storage not in {"gpu", "cpu", "nvme"}:
                raise ValueError(f"Unknown layer-by-layer {component} tier {storage!r}")
        self.group = group
        self.device = device
        self.directory = directory
        self.parameter_tier = tier
        self.activation_tier = activation_tier
        self.optimizer_tier = optimizer_tier
        self.param_count = 0
        self.units = []
        self.activations: _Activations | _GpuActivations | None = None
        seen = set()
        for index, module in enumerate(modules):
            groups: dict[_GroupKey, list[nn.Parameter]] = {}
            for param in module.parameters():
                if (
                    param in seen
                    or not param.is_contiguous()
                    or not param.requires_grad
                    or param.device != device
                ):
                    raise ValueError(
                        "Layer-by-layer units require independent, contiguous, trainable CUDA parameters"
                    )
                seen.add(param)
                self.param_count += param.numel()
                groups.setdefault((param.dtype, param.ndim > 1), []).append(param)
            buckets = []
            for number, (key, params) in enumerate(groups.items()):
                bucket = _Bucket.create(
                    key,
                    params,
                    directory / f"unit-{index}-bucket-{number}",
                    rank=group.rank(),
                    world=group.size(),
                    tier=tier,
                    optimizer_tier=optimizer_tier,
                    device=device,
                    weight_decay=weight_decay,
                )
                bucket.unbind(torch.empty(0, dtype=key[0], device=device))
                buckets.append(bucket)
            self.units.append(_Unit(module, buckets))
        if len(self.units) < 3:
            raise ValueError("Layer-by-layer training requires embedding, decoder and loss units")
        self.betas = betas
        self.eps = eps
        self._initialized = torch.cuda.current_stream(device).record_event()

    def __enter__(self):
        with ExitStack() as stack:
            readers = [
                stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix=name))
                for name in (
                    "parameter-read",
                    "activation-read",
                    "activation-write",
                    "adam-read",
                    "adam-update",
                )
            ]
            self.weights = _WeightSlots(
                self.units,
                self.device,
                self.group,
                readers[0],
                parameter_tier=self.parameter_tier,
                initialized=self._initialized,
            )
            self.activation_reader, self.activation_writer = readers[1:3]
            if self.optimizer_tier == "gpu":
                self.updates = _GpuUpdates(
                    self.device,
                    self.group,
                    betas=self.betas,
                    eps=self.eps,
                    initialized=self._initialized,
                )
            else:
                self.updates = _CpuUpdates(
                    self.units,
                    self.device,
                    self.group,
                    *readers[3:],
                    betas=self.betas,
                    eps=self.eps,
                )
            self.workers = stack.pop_all()
        return self

    def __exit__(self, *exc):
        return self.workers.__exit__(*exc)

    @property
    def activation_transfer_counts(self) -> tuple[int, int]:
        """Return cumulative D2H and H2D activation counts after a completed step."""
        if isinstance(self.activations, _Activations):
            return self.activations.offloaded, self.activations.reloaded
        return 0, 0

    def _inputs(self, index, batches, *, backward):
        activations = self.activations
        assert not index or activations is not None
        indices = iter(range(len(batches) - 1, -1, -1) if backward else range(len(batches)))
        current = next(indices)
        if index:
            activations.prefetch(index, current, backward=backward)
        for upcoming in (*indices, None):
            tokens, labels = batches[current]
            value = activations.acquire() if index else tokens
            if index and upcoming is not None:
                activations.prefetch(index, upcoming, backward=backward)
            yield current, value, labels
            if index:
                activations.release(current)
            current = upcoming

    @torch.no_grad()
    def _forward(self, batches):
        self.weights.prefetch(0, self.units[0])
        last = len(self.units) - 1
        losses = []
        for index, unit in enumerate(self.units):
            self.weights.bind(index, unit)
            if index < last:
                self.weights.prefetch(index + 1, self.units[index + 1])
            if index == 0:
                layer_name = "embedding"
            elif index == last:
                layer_name = "output"
            else:
                layer_name = f"transformer.{index - 1}"
            with torch.cuda.nvtx.range(f"forward.{layer_name}"):
                for mb, value, labels in self._inputs(index, batches, backward=False):
                    with torch.cuda.nvtx.range(f"fwd.microbatch.{mb}.{layer_name}"):
                        result = unit.module(value, labels)
                    if index == last:
                        losses.append(result.detach())
                    else:
                        if self.activations is None:
                            if self.activation_tier == "gpu":
                                self.activations = _GpuActivations()
                            else:
                                self.activations = _Activations(
                                    result,
                                    self.directory,
                                    self.activation_reader,
                                    self.activation_writer,
                                    nvme=self.activation_tier == "nvme",
                                )
                        self.activations.store(index + 1, mb, result)
                    self.weights.progress()
            if index < last:
                self.weights.release(index, unit)
        loss = torch.stack(losses).mean()
        dist.all_reduce(loss, group=self.group)
        return loss.item() / self.group.size()

    def _backward(self, batches, *, lr):
        input_grads = [None] * len(batches)
        last = len(self.units) - 1
        for index in range(last, -1, -1):
            unit = self.units[index]
            if index < last:
                self.weights.bind(index, unit)
            if index:
                self.weights.prefetch(index - 1, self.units[index - 1])
            optimizer = self.updates.prefetch(index, unit)
            main_grads = [
                torch.zeros(
                    bucket.shard_size * self.group.size(), dtype=torch.float32, device=self.device
                )
                for bucket in unit.buckets
            ]
            params = unit.params
            with torch.cuda.nvtx.range(f"layer_by_layer.backward.{index}"):
                for mb, value, labels in self._inputs(index, batches, backward=True):
                    inputs = params
                    if index:
                        value = value.detach().requires_grad_(True)
                        inputs = (*params, value)
                    result = unit.module(value, labels)
                    if index == last:
                        grads = torch.autograd.grad(result / len(batches), inputs)
                    else:
                        grads = torch.autograd.grad(result, inputs, grad_outputs=input_grads[mb])
                    input_grads[mb] = grads[-1] if index else None
                    offset = 0
                    for bucket, main in zip(unit.buckets, main_grads):
                        end = offset + len(bucket.params)
                        bucket.accumulate(main, grads[offset:end])
                        offset = end
                    del result, grads
                    self.weights.stage()
            # On the shared communicator, every rank submits this layer's
            # reduce-scatter before the next bind issues its all-gather.
            self.weights.release(index, unit)
            self.updates.submit(index, unit, main_grads, optimizer, lr=lr)

    def step(
        self, batches: Sequence[tuple[Tensor, Tensor]], *, lr: float, iteration: int | None = None
    ) -> float:
        """Return the DP mean loss. Accept an optional zero-based iteration for compatibility."""
        if not batches:
            raise ValueError("Layer-by-layer training requires at least one micro-batch")
        if self.updates.invalid_unit >= 0:
            raise RuntimeError("Discard the trainer after a non-finite gradient")
        if iteration is not None and (type(iteration) is not int or iteration < 0):
            raise ValueError("Layer-by-layer iteration must be a nonnegative integer")
        loss = self._forward(batches)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss}")
        self._backward(batches, lr=lr)
        self.updates.finish()
        return loss
