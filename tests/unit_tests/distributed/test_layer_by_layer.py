# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical regression tests for layer-by-layer training."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch import nn

from megatron.core.distributed import layer_by_layer
from megatron.core.distributed.layer_by_layer import LayerByLayerTrainer
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available() or Utils.world_size < 2,
    reason="requires at least two CUDA ranks with NCCL",
)


class _Embedding(nn.Embedding):
    def forward(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return super().forward(tokens)


class _Block(nn.Linear):
    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return hidden + super().forward(hidden).tanh()


class _Loss(nn.Linear):
    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(super().forward(hidden), labels)


def _reference_step(
    units: nn.ModuleList,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for tokens, labels in batches:
        value = tokens
        for unit in units:
            value = unit(value, labels)
        (value / len(batches)).backward()
        losses.append(value.detach())
    for param in units.parameters():
        assert param.grad is not None
        dist.all_reduce(param.grad)
        param.grad.div_(dist.get_world_size())
    optimizer.step()
    loss = torch.stack(losses).mean()
    dist.all_reduce(loss)
    return loss.item() / dist.get_world_size()


def _assert_weights(trainer: LayerByLayerTrainer, reference: nn.ModuleList) -> None:
    for unit, reference_unit in zip(trainer.units, reference):
        for bucket in unit.buckets:
            local = bucket.shard.gpu_parameter
            if local is None:
                host = torch.empty(bucket.shard_size, dtype=bucket.key[0], device="cpu")
                bucket.shard.read_parameter(host)
                local = host.to(trainer.device)
            full = torch.empty(
                bucket.shard_size * dist.get_world_size(), dtype=local.dtype, device=trainer.device
            )
            dist.all_gather_into_tensor(full, local)
            expected = torch.cat(
                [
                    param.detach().flatten()
                    for param in reference_unit.parameters()
                    if (param.dtype, param.ndim > 1) == bucket.key
                ]
            )
            torch.testing.assert_close(full[: expected.numel()], expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("tier", "activation_tier", "optimizer_tier"),
    [
        ("cpu_param", "gpu_act", "nvme"),
        ("cpu_param", "cpu_act", "cpu"),
        ("cpu_param", "nvme_act", "nvme"),
        ("nvme_param", "gpu_act", "nvme"),
        ("nvme_param", "cpu_act", "nvme"),
        ("nvme_param", "nvme_act", "nvme"),
        ("gpu", "gpu", "gpu"),
        ("gpu", "cpu", "cpu"),
        ("gpu", "nvme", "nvme"),
        ("cpu", "gpu", "gpu"),
        ("nvme", "nvme", "gpu"),
    ],
)
def test_layer_by_layer_matches_unsharded_adamw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: str,
    activation_tier: str,
    optimizer_tier: str,
) -> None:
    """Compare weights, losses, and activation transfers across storage tiers."""
    Utils.initialize_distributed()
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(114)
        modules = nn.ModuleList([_Embedding(11, 7), _Block(7, 7), _Block(7, 7), _Loss(7, 3)]).to(
            device
        )
    reference = deepcopy(modules)
    lr, betas, eps, decay = 3e-3, (0.9, 0.95), 1e-8, 0.1
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for param in reference.parameters() if param.ndim > 1]},
            {
                "params": [param for param in reference.parameters() if param.ndim <= 1],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=decay,
        foreach=False,
        fused=False,
    )
    generator = torch.Generator().manual_seed(314 + dist.get_rank())
    directory = tmp_path / f"rank-{dist.get_rank()}"
    write_tensor = layer_by_layer.write_tensor
    activation_writes = []
    nvtx_range = torch.cuda.nvtx.range
    active_ranges = []
    forward_ranges = []

    def record_activation_write(path: Path, value: torch.Tensor) -> None:
        write_tensor(path, value)
        activation_writes.append(path)

    @contextmanager
    def record_nvtx(message: str) -> Iterator[None]:
        with nvtx_range(message):
            active_ranges.append(message)
            try:
                if message.startswith(("forward.", "fwd.microbatch.", "layer_by_layer.forward.")):
                    forward_ranges.append(tuple(active_ranges))
                yield
            finally:
                active_ranges.pop()

    monkeypatch.setattr(layer_by_layer, "write_tensor", record_activation_write)
    monkeypatch.setattr(torch.cuda.nvtx, "range", record_nvtx)
    with LayerByLayerTrainer(
        modules,
        group=dist.group.WORLD,
        device=device,
        directory=directory,
        tier=tier,
        activation_tier=activation_tier,
        optimizer_tier=optimizer_tier,
        betas=betas,
        eps=eps,
        weight_decay=decay,
    ) as trainer:
        if tier == "gpu":
            assert not trainer.weights.host and not trainer.weights.local
        for step in range(2):
            batches = [
                (
                    torch.randint(11, (2, 5), generator=generator).to(device),
                    torch.randn((2, 5, 3), generator=generator).to(device),
                )
                for _ in range(3)
            ]
            actual_loss = trainer.step(batches, lr=lr, iteration=3 if step == 0 else None)
            expected_loss = _reference_step(reference, batches, optimizer)
            assert actual_loss == pytest.approx(expected_loss, abs=1e-6, rel=1e-5)
            _assert_weights(trainer, reference)
            boundaries = (step + 1) * (len(modules) - 1) * len(batches)
            storage = activation_tier.removesuffix("_act")
            d2h = 0 if storage == "gpu" else boundaries
            assert trainer.activation_transfer_counts == (d2h, 2 * d2h)
            assert len(activation_writes) == (boundaries if storage == "nvme" else 0)
            expected_forward_ranges = []
            for layer_name in ("embedding", "transformer.0", "transformer.1", "output"):
                layer_range = f"forward.{layer_name}"
                expected_forward_ranges.append((layer_range,))
                expected_forward_ranges.extend(
                    (layer_range, f"fwd.microbatch.{mb}.{layer_name}") for mb in range(len(batches))
                )
            assert forward_ranges == expected_forward_ranges
            forward_ranges.clear()
            assert not list(directory.glob("input-*.bin"))


@pytest.mark.parametrize("optimizer_tier", ["nvme", "gpu"])
def test_layer_by_layer_rejects_nonfinite_shard_on_all_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer_tier: str
) -> None:
    """An invalid gradient on one rank must remain visible after the slot is reused."""
    Utils.initialize_distributed()
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(115)
        modules = nn.ModuleList([_Embedding(11, 7), _Block(7, 7), _Block(7, 7), _Loss(7, 3)]).to(
            device
        )
    reduce_scatter = dist.reduce_scatter_tensor
    first_reduction = True

    def inject_nonfinite(output: torch.Tensor, *args, **kwargs) -> None:
        nonlocal first_reduction
        reduce_scatter(output, *args, **kwargs)
        if first_reduction and rank == 0:
            output[0] = float("nan")
        first_reduction = False

    monkeypatch.setattr(dist, "reduce_scatter_tensor", inject_nonfinite)
    batches = [
        (
            torch.zeros((2, 5), device=device, dtype=torch.long),
            torch.zeros((2, 5, 3), device=device),
        )
    ]
    with LayerByLayerTrainer(
        modules,
        group=dist.group.WORLD,
        device=device,
        directory=tmp_path / f"rank-{rank}",
        tier="nvme_param",
        optimizer_tier=optimizer_tier,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    ) as trainer:
        bucket = trainer.units[-1].buckets[0]
        before = torch.empty(bucket.shard_size, dtype=bucket.key[0], device="cpu")
        bucket.shard.read_parameter(before)
        with pytest.raises(FloatingPointError, match="Non-finite training gradient at unit 3"):
            trainer.step(batches, lr=3e-3)
        if rank == 0:
            after = torch.empty_like(before)
            bucket.shard.read_parameter(after)
            torch.testing.assert_close(after, before, atol=0, rtol=0)
        with pytest.raises(RuntimeError, match="Discard the trainer"):
            trainer.step(batches, lr=3e-3)
