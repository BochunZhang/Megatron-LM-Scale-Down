# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sweep transformer depth on a four-GPU layer-by-layer training node."""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

_LOG = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[3]
_WORLD_SIZE = 4
_SOURCE_FILES = (
    "examples/scale-down/06-layer-by-layer/run_sweep.sh",
    "examples/scale-down/06-layer-by-layer/sweep.py",
    "megatron/core/distributed/layer_by_layer.py",
    "megatron/core/distributed/layer_by_layer_storage.py",
    "megatron/training/layer_by_layer.py",
    "megatron/training/layer_by_layer_workload.py",
)
_SINGLE_CASE_SCRIPT = "examples/scale-down/06-layer-by-layer/01-baseline/01_cpu_activation.sh"
_RESULT_PREFIX = "LAYER_BY_LAYER_RESULT "
_RESOURCE_ERRORS = {
    "gpu_oom": re.compile(
        r"cuda out of memory|cuda error: out of memory|cublas_status_alloc_failed", re.I
    ),
    "cpu_oom": re.compile(
        r"defaultcpuallocator: (?:can't|not enough memory)|std::bad_alloc|\bMemoryError\b", re.I
    ),
    "nvme_full": re.compile(r"no space left on device|disk quota exceeded", re.I),
}
_RESULT_COLUMNS = (
    "layers",
    "activation_tier",
    "status",
    "returncode",
    "steps",
    "loss",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "activation_d2h_completed_per_rank",
    "activation_h2d_completed_per_rank",
    "log",
)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvme-dir", type=Path, required=True, help="existing directory on local NVMe"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="new output directory (default: unique directory under ./results)",
    )
    parser.add_argument(
        "--layers", type=_positive, nargs="+", default=[16, 32, 64], help="depths to attempt"
    )
    parser.add_argument("--seq-length", type=_positive, default=4096)
    parser.add_argument("--micro-batch-size", type=_positive, default=1)
    parser.add_argument(
        "--microbatches", type=_positive, default=4, help="micro-batches per rank per step"
    )
    parser.add_argument("--steps", type=_positive, default=3)
    parser.add_argument(
        "--parameter-tier", choices=("cpu_param", "nvme_param"), default="nvme_param"
    )
    parser.add_argument(
        "--activation-tier", choices=("gpu_act", "cpu_act", "nvme_act"), default="nvme_act"
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cpu-threads", type=_positive)
    parser.add_argument("--tiny", action="store_true", help="use the small smoke model")
    parser.add_argument(
        "--single-case-layout", action="store_true", help="use node-level files for one depth"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show commands without writing or launching"
    )
    args = parser.parse_args()
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers must contain distinct values")
    if args.single_case_layout and (len(args.layers) != 1 or args.results_dir is None):
        parser.error("--single-case-layout requires exactly one depth and --results-dir")
    return args


def _command(args: argparse.Namespace, layers: int, scratch: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={_WORLD_SIZE}",
        "-m",
        "megatron.training.layer_by_layer",
        "--num-layers",
        str(layers),
        "--seq-length",
        str(args.seq_length),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--microbatches",
        str(args.microbatches),
        "--steps",
        str(args.steps),
        "--parameter-tier",
        args.parameter_tier,
        "--activation-tier",
        args.activation_tier,
        "--nvme-dir",
        str(scratch),
    ]
    if args.tiny:
        command.append("--tiny")
    if args.learning_rate is not None:
        command.extend(("--learning-rate", str(args.learning_rate)))
    if args.cpu_threads is not None:
        command.extend(("--cpu-threads", str(args.cpu_threads)))
    return command


def _read_result(
    log_path: Path,
    returncode: int,
    layers: int,
    *,
    steps: int,
    microbatches: int,
    activation_tier: str,
) -> dict:
    result = {
        "layers": layers,
        "activation_tier": activation_tier,
        "status": "failed",
        "returncode": returncode,
        "log": log_path.name,
    }
    resource_errors = set()
    records = []
    with log_path.open(encoding="utf-8") as log:
        for line in log:
            if line.startswith(_RESULT_PREFIX):
                try:
                    records.append(json.loads(line[len(_RESULT_PREFIX) :]))
                except ValueError:
                    records.append(None)
            resource_errors.update(
                name for name, pattern in _RESOURCE_ERRORS.items() if pattern.search(line)
            )
    if returncode:
        if len(resource_errors) == 1:
            result["status"] = resource_errors.pop()
        return result
    result["status"] = "invalid_result"
    if len(records) != 1 or not isinstance(records[0], dict):
        return result
    metrics = records[0]
    loss = metrics.get("loss")
    allocated = metrics.get("peak_allocated_bytes")
    reserved = metrics.get("peak_reserved_bytes")
    d2h = metrics.get("activation_d2h_completed_by_rank")
    h2d = metrics.get("activation_h2d_completed_by_rank")
    expected_d2h = 0 if activation_tier == "gpu_act" else (layers + 1) * microbatches * steps
    if (
        metrics.get("layers") == layers
        and metrics.get("steps") == steps
        and metrics.get("activation_tier") == activation_tier
        and d2h == [expected_d2h] * _WORLD_SIZE
        and h2d == [2 * expected_d2h] * _WORLD_SIZE
        and all(type(count) is int for count in (*d2h, *h2d))
        and type(loss) in (float, int)
        and math.isfinite(loss)
        and type(allocated) is int
        and type(reserved) is int
        and 0 < allocated <= reserved
    ):
        result.update(
            status="passed",
            steps=steps,
            loss=loss,
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
            activation_d2h_completed_per_rank=d2h[0],
            activation_h2d_completed_per_rank=h2d[0],
        )
    return result


def _write_launch(args: argparse.Namespace, results: Path, nvme_dir: Path) -> None:
    launch = {
        "model": "Qwen/Qwen3.5-27B",
        "dtype": "bf16",
        "num_layers": args.layers[0],
        "tiny": args.tiny,
        "fully_sharded": True,
        "data_parallel_size": _WORLD_SIZE,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.micro_batch_size * args.microbatches * _WORLD_SIZE,
        "num_microbatches": args.microbatches,
        "train_iters": args.steps,
        "gpu_count": _WORLD_SIZE,
        "nnodes": 1,
        "node_rank": 0,
        "parameter_tier": args.parameter_tier,
        "activation_tier": args.activation_tier,
        "nvme_path": str(nvme_dir),
        "profile": "none",
        "cuda_graph": "off",
    }
    (results / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")


def _run_case(args: argparse.Namespace, layers: int, results: Path, nvme_dir: Path) -> dict:
    log_name, command_name = (
        ("train.log", "command.txt")
        if args.single_case_layout
        else (f"layers-{layers}.log", f"layers-{layers}.command.txt")
    )
    log_path = results / log_name
    with tempfile.TemporaryDirectory(prefix=f"layer-by-layer-{layers}-", dir=nvme_dir) as scratch:
        command = _command(args, layers, Path(scratch))
        (results / command_name).write_text(shlex.join(command) + "\n", encoding="utf-8")
        _LOG.info("Running %d layers. Log at %s", layers, log_path)
        with log_path.open("w", encoding="utf-8") as log:
            returncode = subprocess.run(
                command,
                cwd=_REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                check=False,
            ).returncode
    return _read_result(
        log_path,
        returncode,
        layers,
        steps=args.steps,
        microbatches=args.microbatches,
        activation_tier=args.activation_tier,
    )


def main() -> int:
    """Run selected depths and report the largest completed case."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _arguments()
    nvme_dir = args.nvme_dir.resolve()
    if args.dry_run:
        for layers in args.layers:
            scratch = nvme_dir / f"layer-by-layer-{layers}-<unique>"
            _LOG.info("%s", shlex.join(_command(args, layers, scratch)))
        return 0
    if not nvme_dir.is_dir():
        _LOG.error("NVMe directory does not exist: %s", nvme_dir)
        return 2
    if args.results_dir is None:
        root = Path("results").resolve()
        root.mkdir(parents=True, exist_ok=True)
        results = Path(tempfile.mkdtemp(prefix="06-layer-by-layer-", dir=root))
    else:
        results = args.results_dir.resolve()
        results.mkdir(parents=True, exist_ok=False)

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    (results / "revision.txt").write_text(revision + "\n", encoding="utf-8")
    source_files = _SOURCE_FILES
    if args.single_case_layout:
        _write_launch(args, results, nvme_dir)
        source_files += (_SINGLE_CASE_SCRIPT,)
    source_hashes = {
        path: hashlib.sha256((_REPO / path).read_bytes()).hexdigest() for path in source_files
    }
    (results / "source_sha256.json").write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    largest = None
    had_error = False
    with (results / "summary.csv").open("w", encoding="utf-8", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        for layers in args.layers:
            case = _run_case(args, layers, results, nvme_dir)
            writer.writerow(case)
            summary.flush()
            if args.single_case_layout:
                status = 0 if case["status"] == "passed" else case["returncode"] or 1
                (results / "run_info.txt").write_text(
                    f"case_status={case['status']}\nreturncode={case['returncode']}\nstatus={status}\n",
                    encoding="utf-8",
                )
            _LOG.info("%d layers: %s", layers, case["status"])
            if case["status"] == "passed":
                largest = layers if largest is None else max(largest, layers)
            elif case["status"] not in _RESOURCE_ERRORS:
                had_error = True
    _LOG.info("Largest successful attempted depth: %s. Results at %s", largest, results)
    return int(had_error or largest is None)


if __name__ == "__main__":
    sys.exit(main())
