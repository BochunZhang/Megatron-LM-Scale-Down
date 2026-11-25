# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""CPU AdamW with optional GPU layer-zero updates for Megatron-FSDP shards."""

import platform
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from torch.optim._functional import adamw

from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import nvtx_range

_MOMENTS = ("exp_avg", "exp_avg_sq")


def _create_grace_adam() -> torch.optim.Optimizer:
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("FSDP GraceAdam requires Linux AArch64 with SVE.")
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError as exc:
        raise RuntimeError("FSDP GraceAdam could not check CPU support for SVE.") from exc
    if not any(
        line.startswith("Features") and "sve" in line.split() for line in cpuinfo.splitlines()
    ):
        raise RuntimeError("FSDP GraceAdam requires Linux AArch64 with SVE.")

    requirement = "FSDP GraceAdam requires DeepSpeed v0.20.1 or a compatible SVE build."
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(requirement) from exc

    if not hasattr(DeepSpeedCPUAdam, "copy_supported") or not hasattr(
        DeepSpeedCPUAdam, "grad_scale_supported"
    ):
        raise RuntimeError(requirement)
    try:
        optimizer = DeepSpeedCPUAdam([{"params": []}], adamw_mode=True, fp32_optimizer_states=True)
        # On AArch64, the SVE kernel supports gradient scaling and lacks a fused BF16 store.
        supports_sve = not optimizer.copy_supported() and optimizer.grad_scale_supported()
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(requirement) from exc
    if not supports_sve:
        raise RuntimeError(requirement)
    return optimizer


class FSDPCPUAdam(torch.optim.AdamW):
    """Keep FSDP's logical parameter groups while updating their local shards."""

    def __init__(
        self, params: list[dict], model_chunks: list, *, backend: str = "torch", **kwargs
    ) -> None:
        super().__init__(params, foreach=True, **kwargs)
        self._grace_adam = _create_grace_adam() if backend == "grace_adam" else None
        self._layer_by_layer = any(
            getattr(getattr(chunk, "ddp_config", None), "fsdp_layer_by_layer_cpu_optimizer", False)
            for chunk in model_chunks
        )
        if self._layer_by_layer and backend != "grace_adam":
            raise ValueError("The layer-by-layer hybrid optimizer requires GraceAdam.")
        self.buckets = []
        for chunk in model_chunks:
            buffers = chunk.param_and_grad_buffer
            params_by_bucket = defaultdict(dict)
            for _, param in buffers.optimizer_named_parameters:
                bucket = buffers.param_to_param_group[param.orig_param]
                params_by_bucket[bucket][param] = param.orig_param
            for bucket_id, bucket in enumerate(buffers.parameter_groups):
                if bucket_id in params_by_bucket:
                    self.buckets.append((buffers, bucket, params_by_bucket[bucket_id]))

    def _moments(self, master, *, load):
        if master.nvme is None:
            return None
        data = torch.empty((2, master.data_size), dtype=master.dtype, device="cpu")
        if load:
            return master.nvme.read(f"adam-{master.bucket_id}", data)
        return data.zero_()

    @staticmethod
    def _views(master, moments, item_id, shape):
        return [
            master.get_item(item_id, only_shard=True, data=plane).view(shape) for plane in moments
        ]

    @staticmethod
    def _step_grace_adam(optimizer, updates, *, label=None) -> None:
        idle_groups = optimizer.param_groups
        native_groups = []
        step_tensors = {}
        try:
            # DeepSpeed borrows the CPU shards while FSDP retains the checkpoint state.
            for group, weights, grads, averages, squares, steps in updates:
                native_groups.append(
                    dict(
                        idle_groups[0],
                        params=weights,
                        lr=group["lr"],
                        betas=group["betas"],
                        eps=group["eps"],
                        weight_decay=group["weight_decay"],
                    )
                )
                for weight, grad, average, square, step in zip(
                    weights, grads, averages, squares, steps
                ):
                    step_tensors[weight] = step
                    weight.grad = grad
                    optimizer.state[weight] = {
                        "step": int(step.item()),
                        "exp_avg": average,
                        "exp_avg_sq": square,
                    }

            optimizer.param_groups = native_groups
            if label is None:
                optimizer.step()
            else:
                # The native call waits for its OpenMP team before returning.
                with nvtx_range(label):
                    optimizer.step()
            for weight, step in step_tensors.items():
                step.fill_(optimizer.state[weight]["step"])
        finally:
            for weight in step_tensors:
                weight.grad = None
            optimizer.param_groups = idle_groups
            optimizer.state.clear()

    def _active_bucket_groups(self, groups):
        for buffers, bucket, params in self.buckets:
            by_group = defaultdict(list)
            for param in params:
                if param in groups:
                    if not param._local_tensor.numel():
                        state = self.state[param]
                        state.setdefault("step", torch.zeros((), device="cpu")).add_(1)
                    elif (
                        param.grad is not None or getattr(param, "_fsdp_cpu_grad", None) is not None
                    ):
                        by_group[id(groups[param])].append(param)
            if by_group:
                yield buffers, bucket, params, by_group

    def _update_bucket(self, groups, buffers, bucket, params, by_group, weight_data, cpu_grad):
        master = bucket.main_weight_buffer
        gbuf = bucket.main_grad_buffer
        on_gpu = self._layer_by_layer and not master.cpu_offload
        moments = self._moments(
            master, load=any(p._local_tensor.numel() and p in self.state for p in params)
        )
        grace_updates = []
        with nvtx_range(
            lambda: (
                f"fsdp.layer_by_layer.GPUAdamW bucket={master.bucket_id} {bucket.nvtx_label}"
                if on_gpu
                else f"fsdp.cpu_optimizer.adamw bucket={master.bucket_id}"
            )
        ):
            for active_params in by_group.values():
                group = groups[active_params[0]]
                weights, grads, averages, squares, steps = [], [], [], [], []
                for param in active_params:
                    original = params[param]
                    item = master.param_idx[original]
                    weight = master.get_item(item, only_shard=True, data=weight_data)
                    if on_gpu:
                        grad = param.grad._local_tensor.view_as(weight)
                    elif self._layer_by_layer:
                        grad = param._fsdp_cpu_grad.view_as(weight)
                    else:
                        grad = gbuf.get_item(
                            gbuf.param_idx[original], only_shard=True, data=cpu_grad
                        )
                    state = self.state[param]
                    if "step" not in state:
                        state["step"] = torch.zeros((), device="cpu")
                        if moments is None:
                            state.update({key: torch.zeros_like(weight) for key in _MOMENTS})
                    if moments is None:
                        avg, sq = (state[key] for key in _MOMENTS)
                    else:
                        avg, sq = self._views(master, moments, item, weight.shape)
                    weights.append(weight)
                    grads.append(grad)
                    averages.append(avg)
                    squares.append(sq)
                    steps.append(state["step"])
                if self._grace_adam is not None and not on_gpu:
                    update = (group, weights, grads, averages, squares, steps)
                    if self._layer_by_layer:
                        grace_updates.append(update)
                    else:
                        self._step_grace_adam(self._grace_adam, [update])
                    continue
                beta1, beta2 = group["betas"]
                adamw(
                    weights,
                    grads,
                    averages,
                    squares,
                    [],
                    steps,
                    foreach=True,
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                )
            if grace_updates:
                self._step_grace_adam(
                    self._grace_adam,
                    grace_updates,
                    label=lambda: f"GraceAdam bucket={master.bucket_id} {bucket.nvtx_label}",
                )
        if on_gpu:
            wbuf = bucket.model_weight_buffer
            for param in bucket.params:
                source = master.get_item(master.param_idx[param], only_shard=True)
                target = wbuf.get_item(wbuf.param_idx[param], only_shard=True)
                target.copy_(source)
        else:
            if moments is not None:
                master.nvme.write(f"adam-{master.bucket_id}", moments)
            buffers.commit_cpu_optimizer_bucket(bucket, weight_data)

    def _step_layer_by_layer(self, groups):
        cpu_buckets = []
        gpu_buckets = []
        for active in self._active_bucket_groups(groups):
            _, bucket, _, _ = active
            if bucket.main_weight_buffer.cpu_offload:
                cpu_buckets.append(active)
            else:
                gpu_buckets.append(active)

        for buffers, bucket, params, by_group in gpu_buckets:
            self._update_bucket(
                groups, buffers, bucket, params, by_group, bucket.main_weight_buffer.data, None
            )

        # The CPU grad views already reference the finalized and clipped FP32 shards.
        for active in cpu_buckets:
            _, bucket, _, _ = active
            self._update_bucket(groups, *active, bucket.main_weight_buffer.read_cpu_data(), None)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Apply AdamW to one FSDP bucket at a time and publish its compute weights."""
        if closure is not None:
            raise ValueError("FSDP CPU AdamW does not take a closure.")
        groups = {param: group for group in self.param_groups for param in group["params"]}
        if self._layer_by_layer:
            self._step_layer_by_layer(groups)
            return
        for buffers, bucket, params, by_group in self._active_bucket_groups(groups):
            master = bucket.main_weight_buffer
            gbuf = bucket.main_grad_buffer
            cpu_master = master.read_cpu_data()
            with nvtx_range(lambda: f"fsdp.cpu_optimizer.gradient_d2h bucket={master.bucket_id}"):
                # Finalize, unscale and clip use the usual Megatron collectives.
                # When main grads already match Adam's dtype, the clipped views
                # alias the buffer and one D2H copy can stage the whole shard.
                if gbuf.dtype == master.dtype:
                    cpu_grad = gbuf.new_cpu_data()
                    cpu_grad.copy_(gbuf.data)
                else:
                    # Clipping modified the FP32 casts attached to each DTensor.
                    cpu_grad = torch.empty(
                        gbuf.data_size, dtype=master.dtype, device="cpu", pin_memory=True
                    )
                    for active_params in by_group.values():
                        for param in active_params:
                            item_id = gbuf.param_idx[params[param]]
                            target = gbuf.get_item(item_id, only_shard=True, data=cpu_grad)
                            target.copy_(param.grad._local_tensor.view(-1), non_blocking=True)
                    torch.cuda.current_stream(gbuf.device).synchronize()
            self._update_bucket(groups, buffers, bucket, params, by_group, cpu_master, cpu_grad)

    def checkpoint_state(self, *, is_loading: bool) -> dict:
        """Expose moments as FSDP DTensors for the distributed checkpoint planner."""
        parameters = {p for group in self.param_groups for p in group["params"]}
        result = {}
        for _, bucket, params in self.buckets:
            master = bucket.main_weight_buffer
            if master is None:
                continue
            moments = self._moments(
                master, load=any(p._local_tensor.numel() and p in self.state for p in params)
            )
            for param, original in params.items():
                if param not in parameters:
                    continue
                state = self.state.get(param)
                if state is None and not is_loading:
                    continue
                local = param._local_tensor
                if moments is not None:
                    values = self._views(master, moments, master.param_idx[original], local.shape)
                elif state is not None and local.numel():
                    values = [state[key].view(local.shape) for key in _MOMENTS]
                else:
                    values = [torch.zeros_like(local) for _ in _MOMENTS]
                result[param] = {
                    key: DTensor(
                        value.to(local.device, copy=True), param._spec, requires_grad=False
                    )
                    for key, value in zip(_MOMENTS, values)
                }
                result[param]["step"] = (
                    state["step"].clone() if state else torch.zeros((), device="cpu")
                )
        return result

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore AdamW moments to their local optimizer device."""
        super().load_state_dict(state_dict)
        for _, bucket, params in self.buckets:
            master = bucket.main_weight_buffer
            active = [p for p in params if p in self.state]
            if not active:
                continue
            moments = self._moments(master, load=False)
            for param in active:
                state = self.state[param]
                state["step"] = state["step"].cpu()
                values = [state[key] for key in _MOMENTS]
                values = [v._local_tensor if isinstance(v, DTensor) else v for v in values]
                if moments is None:
                    device = (
                        param._local_tensor.device
                        if self._layer_by_layer and not master.cpu_offload
                        else "cpu"
                    )
                    state.update(
                        {
                            key: value.to(device=device).flatten()
                            for key, value in zip(_MOMENTS, values)
                        }
                    )
                else:
                    item = master.param_idx[params[param]]
                    for key, view, value in zip(
                        _MOMENTS, self._views(master, moments, item, (-1,)), values
                    ):
                        view.copy_(value.flatten())
                        del state[key]
            if moments is not None:
                master.nvme.write(f"adam-{master.bucket_id}", moments)
