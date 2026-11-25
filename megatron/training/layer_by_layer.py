# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run Megatron's experimental Qwen3.5 layer-by-layer mock pretraining."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch import distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed.layer_by_layer import LayerByLayerTrainer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.layer_by_layer_workload import Qwen35DenseWorkload

_LOG = logging.getLogger(__name__)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=_positive, default=64)
    parser.add_argument("--seq-length", type=_positive, default=4096)
    parser.add_argument("--micro-batch-size", type=_positive, default=1)
    parser.add_argument("--microbatches", type=_positive, default=4)
    parser.add_argument("--steps", type=_positive, default=3)
    parser.add_argument(
        "--parameter-tier", choices=("cpu_param", "nvme_param"), default="nvme_param"
    )
    parser.add_argument(
        "--activation-tier", choices=("gpu_act", "cpu_act", "nvme_act"), default="nvme_act"
    )
    parser.add_argument("--nvme-dir", required=True, type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=_positive, default=8)
    parser.add_argument("--seed", type=_positive, default=1234)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--profile-start", type=int, help="first profiled step, starting at zero")
    parser.add_argument("--profile-end", type=int, help="first step after the profiled window")
    args = parser.parse_args()
    if not args.nvme_dir.is_dir():
        parser.error("--nvme-dir must be an existing directory")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if (args.profile_start is None) != (args.profile_end is None):
        parser.error("--profile-start and --profile-end must be supplied together")
    if (
        args.profile_start is not None
        and not 0 <= args.profile_start < args.profile_end <= args.steps
    ):
        parser.error("profiling requires 0 <= --profile-start < --profile-end <= --steps")
    return args


def _train(args: argparse.Namespace, device: torch.device, group: dist.ProcessGroup) -> dict:
    workload = Qwen35DenseWorkload(
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
        device=device,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(["tp", "pp", "cp"]),
        tiny=args.tiny,
    )
    rank = group.rank()
    elapsed = []
    if rank == 0:
        _LOG.info(
            "Initializing %d layers | sharding %d | %s | %s | %d micro-batches per rank",
            args.num_layers,
            group.size(),
            args.parameter_tier,
            args.activation_tier,
            args.microbatches,
        )
    with TemporaryDirectory(prefix=f"rank-{rank}-", dir=args.nvme_dir) as scratch:
        with LayerByLayerTrainer(
            workload.make_units(),
            group=group,
            device=device,
            directory=Path(scratch),
            tier=args.parameter_tier,
            activation_tier=args.activation_tier,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        ) as trainer:
            for step in range(args.steps):
                batches = [workload.make_batch(step, mb, rank) for mb in range(args.microbatches)]
                torch.cuda.synchronize(device)
                dist.barrier(group=group, device_ids=[device.index])
                if step == args.profile_start:
                    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                start = perf_counter()
                loss = trainer.step(batches, lr=args.learning_rate)
                torch.cuda.synchronize(device)
                elapsed.append(perf_counter() - start)
                if step + 1 == args.profile_end:
                    # The first rank to stop the capture must wait for every rank's step.
                    dist.barrier(group=group, device_ids=[device.index])
                    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
                if rank == 0:
                    _LOG.info(
                        "Step %d/%d | loss %.8f | %.3f s", step + 1, args.steps, loss, elapsed[-1]
                    )
            model_parameters = trainer.param_count
            activation_transfers = trainer.activation_transfer_counts

    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.int64,
        device=device,
    )
    times = torch.tensor(elapsed, dtype=torch.float64, device=device)
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(times, op=dist.ReduceOp.MAX, group=group)
    allocated, reserved = peaks.tolist()
    local_transfers = torch.tensor(activation_transfers, dtype=torch.int64, device=device)
    transfers = torch.empty(
        group.size() * local_transfers.numel(), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(transfers, local_transfers, group=group)
    transfers_by_rank = transfers.view(group.size(), 2).tolist()
    expected_d2h = (
        0
        if args.activation_tier == "gpu_act"
        else (args.num_layers + 1) * args.microbatches * args.steps
    )
    expected_transfers = [expected_d2h, 2 * expected_d2h]
    if any(counts != expected_transfers for counts in transfers_by_rank):
        raise RuntimeError(
            f"Activation transfer counts for {args.activation_tier}: "
            f"expected {expected_transfers} D2H/H2D per rank, got {transfers_by_rank}"
        )
    return {
        "layers": args.num_layers,
        "steps": args.steps,
        "parameter_count": model_parameters,
        "parameter_tier": args.parameter_tier,
        "activation_tier": args.activation_tier,
        "activation_d2h_completed_by_rank": [counts[0] for counts in transfers_by_rank],
        "activation_h2d_completed_by_rank": [counts[1] for counts in transfers_by_rank],
        "loss": loss,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "step_seconds": times.tolist(),
    }


def main() -> None:
    """Launch the BF16 capacity workload on the torchrun CUDA ranks."""
    args = _args()
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("Layer-by-layer training requires CUDA and NCCL")
    if (
        os.environ.get("WORLD_SIZE") != os.environ.get("LOCAL_WORLD_SIZE")
        or "LOCAL_RANK" not in os.environ
    ):
        raise RuntimeError("Launch on one node using python -m torch.distributed.run")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        parallel_state.initialize_model_parallel(create_gloo_process_groups=False)
        group = parallel_state.get_data_parallel_group()
        result = _train(args, device, group)
        if dist.get_rank() == 0:
            _LOG.info("LAYER_BY_LAYER_RESULT %s", json.dumps(result, allow_nan=False))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
