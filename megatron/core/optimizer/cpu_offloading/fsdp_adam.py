# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""CPU AdamW updates for Megatron-FSDP master weight shards."""

from collections import defaultdict

import torch
from torch.distributed.tensor import DTensor
from torch.optim._functional import adamw

from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import nvtx_range

_MOMENTS = ("exp_avg", "exp_avg_sq")


class FSDPCPUAdam(torch.optim.AdamW):
    """Keep FSDP's logical parameter groups while updating local CPU shards."""

    def __init__(self, params: list[dict], model_chunks: list, **kwargs):
        super().__init__(params, foreach=True, **kwargs)
        self.buckets = []
        for chunk in model_chunks:
            buffers = chunk.param_and_grad_buffer
            params_by_bucket = defaultdict(dict)
            for _, param in buffers.optimizer_named_parameters:
                bucket = buffers.param_to_param_group[param.orig_param]
                params_by_bucket[bucket][param] = param.orig_param
            for bucket, params_in_bucket in params_by_bucket.items():
                self.buckets.append((buffers, buffers.parameter_groups[bucket], params_in_bucket))

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

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Apply AdamW to one FSDP bucket at a time and publish its compute weights."""
        if closure is not None:
            raise ValueError("FSDP CPU AdamW does not take a closure.")
        groups = {param: group for group in self.param_groups for param in group["params"]}
        for buffers, bucket, params in self.buckets:
            by_group = defaultdict(list)
            for param in params:
                if param in groups:
                    if not param._local_tensor.numel():
                        state = self.state[param]
                        state.setdefault("step", torch.zeros((), device="cpu")).add_(1)
                    elif param.grad is not None:
                        by_group[id(groups[param])].append(param)
            if not by_group:
                continue

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
            moments = self._moments(
                master, load=any(p._local_tensor.numel() and p in self.state for p in params)
            )
            with nvtx_range(lambda: f"fsdp.cpu_optimizer.adamw bucket={master.bucket_id}"):
                for active_params in by_group.values():
                    group = groups[active_params[0]]
                    weights, grads, averages, squares, steps = [], [], [], [], []
                    for param in active_params:
                        original = params[param]
                        item = master.param_idx[original]
                        weight = master.get_item(item, only_shard=True, data=cpu_master)
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
            if moments is not None:
                master.nvme.write(f"adam-{master.bucket_id}", moments)
            buffers.commit_cpu_optimizer_bucket(bucket, cpu_master)

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
        """Restore CPU AdamW state from its distributed checkpoint."""
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
                    state.update(
                        {key: value.cpu().flatten() for key, value in zip(_MOMENTS, values)}
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
