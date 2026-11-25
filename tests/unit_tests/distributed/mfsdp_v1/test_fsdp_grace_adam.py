# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.optimizer import clip_grads
from megatron.core.optimizer.cpu_offloading import fsdp_adam


def test_grace_adam_updates_restored_fsdp_state(monkeypatch):
    weights = torch.tensor([1.0, 2.0])
    gradient = torch.tensor([0.25, -0.5])
    param = torch.nn.Parameter(weights.clone())
    param._local_tensor = param.detach()
    param.orig_param = object()
    param.grad = gradient

    def get_item(item_id, *, only_shard, data):
        assert item_id == 0 and only_shard
        return data

    master = SimpleNamespace(
        bucket_id=0,
        dtype=weights.dtype,
        nvme=None,
        param_idx={param.orig_param: 0},
        get_item=get_item,
        read_cpu_data=lambda: weights,
    )
    grad_buffer = SimpleNamespace(
        dtype=gradient.dtype,
        data=gradient,
        param_idx=master.param_idx,
        get_item=get_item,
        new_cpu_data=lambda: torch.empty_like(gradient),
    )
    bucket = SimpleNamespace(main_weight_buffer=master, main_grad_buffer=grad_buffer)
    buffers = SimpleNamespace(
        optimizer_named_parameters=[("weight", param)],
        param_to_param_group={param.orig_param: 0},
        parameter_groups=[bucket],
        commit_cpu_optimizer_bucket=Mock(),
    )
    grace_adam = SimpleNamespace(param_groups=[{"params": []}], state={})

    def step():
        group = grace_adam.param_groups[0]
        assert group["lr"] == 0.1 and group["weight_decay"] == 0.01
        (weight,) = group["params"]
        torch.testing.assert_close(weight.grad, gradient)
        state = grace_adam.state[weight]
        assert type(state["step"]) is int and state["step"] == 3
        state["step"] += 1
        state["exp_avg"].add_(weight.grad)
        state["exp_avg_sq"].addcmul_(weight.grad, weight.grad)
        weight.add_(weight.grad, alpha=-group["lr"])

    grace_adam.step = Mock(side_effect=step)
    monkeypatch.setattr(fsdp_adam, "_create_grace_adam", lambda: grace_adam)
    optimizer = fsdp_adam.FSDPCPUAdam(
        [{"params": [param]}],
        [SimpleNamespace(param_and_grad_buffer=buffers)],
        lr=0.1,
        weight_decay=0.01,
        backend="grace_adam",
    )
    saved_state = optimizer.state_dict()
    saved_state["state"][0] = dict(
        step=torch.tensor(3.0),
        exp_avg=torch.full_like(weights, 0.5),
        exp_avg_sq=torch.ones_like(weights),
    )
    optimizer.load_state_dict(saved_state)

    optimizer.step()

    grace_adam.step.assert_called_once_with()
    assert not grace_adam.state and not grace_adam.param_groups[0]["params"]
    buffers.commit_cpu_optimizer_bucket.assert_called_once_with(bucket, weights)
    torch.testing.assert_close(weights, torch.tensor([0.975, 2.05]))
    state = optimizer.state_dict()["state"][0]
    torch.testing.assert_close(state["step"], torch.tensor(4.0))
    torch.testing.assert_close(state["exp_avg"], 0.5 + gradient)
    torch.testing.assert_close(state["exp_avg_sq"], 1.0 + gradient.square())


@pytest.mark.parametrize("nvme_weights", [False, True])
def test_layer_by_layer_steps_grace_adam_once_per_fsdp_bucket(monkeypatch, nvme_weights):
    params, bucket_params, buckets = [], [], []
    for index, width in enumerate((1, 2, 1, 1)):
        weight = torch.full((width,), 2.0)
        gradient = 0.25 * torch.arange(
            len(params) + 1, len(params) + width + 1, dtype=torch.float32
        )
        group_params = []
        for item in range(width):
            param = torch.nn.Parameter(weight[item : item + 1])
            param._local_tensor = param.detach()
            param.orig_param = object()
            grad = gradient[item : item + 1]
            if index == 0:
                grad._local_tensor = grad
                param.grad = grad
            else:
                param._fsdp_cpu_grad = grad
            group_params.append(param)
        param_idx = {param.orig_param: item for item, param in enumerate(group_params)}

        def make_buffer(tensor, *, cpu_offload=False):
            def get_item(item_id, *, only_shard, data=None):
                assert only_shard
                return (buffer.data if data is None else data)[item_id : item_id + 1]

            buffer = SimpleNamespace(
                bucket_id=index,
                data=None if cpu_offload else tensor,
                data_size=tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
                cpu_offload=cpu_offload,
                nvme=None,
                param_idx=param_idx,
                get_item=get_item,
                read_cpu_data=lambda: tensor,
            )
            return buffer

        params.extend(group_params)
        bucket_params.append(group_params)
        buckets.append(
            SimpleNamespace(
                params=[param.orig_param for param in group_params],
                nvtx_label=f"transformer.{index}.dense.0.0M",
                main_weight_buffer=make_buffer(weight, cpu_offload=index > 0),
                model_weight_buffer=make_buffer(
                    weight.to(torch.bfloat16), cpu_offload=nvme_weights and index == 0
                ),
                main_grad_buffer=make_buffer(gradient),
            )
        )

    events = []

    @contextmanager
    def nvtx_range(message):
        label = message()
        if label.startswith(("fsdp.layer_by_layer.", "GraceAdam ")):
            events.append(label.removeprefix("fsdp.layer_by_layer."))
        yield

    monkeypatch.setattr(fsdp_adam, "nvtx_range", nvtx_range)

    idle_groups = [{"params": [], "bias_correction": True}]
    grace_adam = SimpleNamespace(param_groups=idle_groups, state={})
    grace_calls = []

    def grace_step():
        updated = []
        for group in grace_adam.param_groups:
            assert group["bias_correction"]
            for weight in group["params"]:
                updated.append(
                    (
                        weight.grad.data_ptr(),
                        group["lr"],
                        group["betas"],
                        group["eps"],
                        group["weight_decay"],
                    )
                )
                weight.sub_(weight.grad, alpha=group["lr"])
                grace_adam.state[weight]["step"] += 1
        grace_calls.append(updated)

    grace_adam.step = grace_step
    monkeypatch.setattr(fsdp_adam, "_create_grace_adam", lambda: grace_adam)
    swapped_weights = {}

    def offload_updated_weights(buffer):
        swapped_weights[buffer.bucket_id] = buffer.data.clone()
        buffer.data = None

    buffers = SimpleNamespace(
        optimizer_named_parameters=[
            ("weight", param) for group_params in reversed(bucket_params) for param in group_params
        ],
        param_to_param_group={
            param.orig_param: index
            for index, group_params in enumerate(bucket_params)
            for param in group_params
        },
        parameter_groups=buckets,
        commit_cpu_optimizer_bucket=Mock(),
        _offload_updated_model_weights=offload_updated_weights,
    )
    chunk = SimpleNamespace(
        param_and_grad_buffer=buffers,
        ddp_config=SimpleNamespace(fsdp_layer_by_layer_cpu_optimizer=True),
    )
    optimizer = fsdp_adam.FSDPCPUAdam(
        [
            {"params": params[:2] + params[3:]},
            {
                "params": [params[2]],
                "lr": 0.05,
                "betas": (0.8, 0.9),
                "eps": 1e-6,
                "weight_decay": 0.0,
            },
        ],
        [chunk],
        backend="grace_adam",
        lr=0.1,
    )
    clip_grads.clip_grad_by_total_norm_fp32(params[1:], max_norm=0.5, total_norm=1.0)
    optimizer.step()

    assert events == [
        "GPUAdamW bucket=0 transformer.0.dense.0.0M",
        "GraceAdam bucket=1 transformer.1.dense.0.0M",
        "GraceAdam bucket=2 transformer.2.dense.0.0M",
        "GraceAdam bucket=3 transformer.3.dense.0.0M",
    ]
    grad_ptrs = [param._fsdp_cpu_grad.data_ptr() for param in params[1:]]
    default = (0.1, (0.9, 0.999), 1e-8, 0.01)
    assert grace_calls == [
        [(grad_ptrs[0], *default), (grad_ptrs[1], 0.05, (0.8, 0.9), 1e-6, 0.0)],
        [(grad_ptrs[2], *default)],
        [(grad_ptrs[3], *default)],
    ]
    assert grace_adam.param_groups is idle_groups and not grace_adam.state
    reference = torch.nn.Parameter(torch.tensor([2.0]))
    reference.grad = torch.tensor([0.25])
    torch.optim.AdamW([reference], lr=0.1).step()
    torch.testing.assert_close(params[0], reference)
    model_weight_buffer = buckets[0].model_weight_buffer
    if nvme_weights:
        assert model_weight_buffer.data is None
        torch.testing.assert_close(swapped_weights[0], reference.to(torch.bfloat16))
    else:
        assert not swapped_weights
        torch.testing.assert_close(model_weight_buffer.data, reference.to(torch.bfloat16))
    assert buffers.commit_cpu_optimizer_bucket.call_count == 3
    for index, param in enumerate(params[1:], start=1):
        grad = 0.25 * (index + 1) * (0.5 / (1.0 + 1.0e-6))
        lr = 0.05 if index == 2 else 0.1
        torch.testing.assert_close(param, torch.tensor([2.0 - lr * grad]))
        assert optimizer.state[param]["step"].item() == 1


def test_fsdp_gradient_norm_combines_cpu_and_gpu_before_collective(monkeypatch):
    cpu_grad = torch.tensor([4.0])
    gpu_grad = SimpleNamespace(device=torch.device("cuda"))
    group = object()
    zeros = torch.zeros

    def cuda_scalar_on_cpu(size, *, dtype, device):
        assert device == "cuda"
        return zeros(size, dtype=dtype)

    fused_norm = Mock(return_value=(torch.tensor([3.0]), None))
    all_reduce = Mock()
    monkeypatch.setattr(torch, "zeros", cuda_scalar_on_cpu)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(clip_grads, "multi_tensor_applier", fused_norm)
    monkeypatch.setattr(clip_grads, "multi_tensor_scale_tensor_impl", None)

    grad_norm = clip_grads.get_grad_norm_fp32([cpu_grad, gpu_grad], grad_stats_parallel_group=group)

    assert grad_norm == 5.0
    assert fused_norm.call_args.args[2] == [[gpu_grad]]
    all_reduce.assert_called_once_with(
        torch.tensor([25.0]), op=torch.distributed.ReduceOp.SUM, group=group
    )


def test_layer_by_layer_grace_adam_waits_for_bucket_offload(monkeypatch):
    weights = [torch.tensor([2.0]), torch.tensor([3.0])]
    gradients = [torch.tensor([0.5]), torch.tensor([1.0])]
    compute_weights = [weight.to(torch.bfloat16) for weight in weights]
    params = [torch.nn.Parameter(weight) for weight in weights]
    buckets = []

    for bucket_id, param in enumerate(params):
        param.orig_param = object()
        param._local_tensor = param.detach()
        grad = gradients[bucket_id]
        weight = weights[bucket_id]
        param_idx = {param.orig_param: 0}

        def get_item(item_id, *, only_shard, data):
            assert item_id == 0 and only_shard
            return data

        master = SimpleNamespace(
            bucket_id=bucket_id,
            cpu_offload=True,
            nvme=None,
            param_idx=param_idx,
            get_item=get_item,
            read_cpu_data=Mock(return_value=weight),
        )
        grad_buffer = SimpleNamespace(
            bucket_id=bucket_id,
            cpu_offload=True,
            cpu_data_valid=True,
            param_idx=param_idx,
            get_item=get_item,
            read_cpu_data=Mock(return_value=grad),
        )
        buckets.append(
            SimpleNamespace(
                params=[param.orig_param],
                main_weight_buffer=master,
                main_grad_buffer=grad_buffer,
                nvtx_label=f"transformer.{bucket_id}.dense.0.0M",
            )
        )

    def publish(bucket, master):
        compute_weights[bucket.main_weight_buffer.bucket_id].copy_(master)

    buffers = SimpleNamespace(
        optimizer_named_parameters=list(zip(("early", "tail"), params)),
        param_to_param_group={param.orig_param: index for index, param in enumerate(params)},
        parameter_groups=buckets,
        submit_cpu_optimizer_bucket=None,
        commit_cpu_optimizer_bucket=Mock(side_effect=publish),
    )
    chunk = SimpleNamespace(
        param_and_grad_buffer=buffers,
        ddp_config=SimpleNamespace(
            fsdp_layer_by_layer_cpu_optimizer=True, fsdp_layer_by_layer_overlap_cpu_optimizer=True
        ),
    )
    early_updated = Event()
    grace_adam = SimpleNamespace(
        ds_opt_adam=SimpleNamespace(adam_update_releases_gil=True),
        param_groups=[{"params": []}],
        state={},
    )

    def grace_step():
        (group,) = grace_adam.param_groups
        (weight,) = group["params"]
        state = grace_adam.state[weight]
        state["step"] += 1
        state["exp_avg"].add_(weight.grad)
        weight.sub_(weight.grad, alpha=group["lr"])
        if weight.grad.data_ptr() == gradients[0].data_ptr():
            early_updated.set()

    grace_adam.step = Mock(side_effect=grace_step)
    monkeypatch.setattr(fsdp_adam, "_create_grace_adam", lambda: grace_adam)
    optimizer = fsdp_adam.FSDPCPUAdam([{"params": params}], [chunk], lr=0.25, backend="grace_adam")

    for iteration in range(2):
        offload_waiting = Event()
        offload_ready = Event()
        early_updated.clear()
        early_gradient = buckets[0].main_grad_buffer.read_cpu_data
        early_gradient.reset_mock()

        def synchronize_offload():
            offload_waiting.set()
            assert offload_ready.wait(timeout=5)

        offload_event = SimpleNamespace(synchronize=synchronize_offload)
        buffers.submit_cpu_optimizer_bucket(0, offload_event)
        try:
            assert offload_waiting.wait(timeout=5)
            early_gradient.assert_not_called()
            assert grace_adam.step.call_count == iteration * 2
        finally:
            offload_ready.set()

        assert early_updated.wait(timeout=5)
        torch.testing.assert_close(weights[0], torch.tensor([2.0 - 0.125 * (iteration + 1)]))
        torch.testing.assert_close(
            compute_weights[0], torch.tensor([2.0 - 0.125 * iteration], dtype=torch.bfloat16)
        )
        assert buffers.commit_cpu_optimizer_bucket.call_count == iteration * 2

        # Finalization attaches both gradient views, including the late bucket.
        for param, grad in zip(params, gradients):
            param._fsdp_cpu_grad = grad
        optimizer.step()

        assert grace_adam.step.call_count == (iteration + 1) * 2
        assert buffers.commit_cpu_optimizer_bucket.call_count == (iteration + 1) * 2
        torch.testing.assert_close(compute_weights[0], weights[0].to(torch.bfloat16))
        torch.testing.assert_close(weights[1], torch.tensor([3.0 - 0.25 * (iteration + 1)]))
        torch.testing.assert_close(compute_weights[1], weights[1].to(torch.bfloat16))
        for param, grad in zip(params, gradients):
            assert optimizer.state[param]["step"].item() == iteration + 1
            torch.testing.assert_close(optimizer.state[param]["exp_avg"], grad * (iteration + 1))


def test_layer_by_layer_grace_adam_overlap_requires_gil_release(monkeypatch):
    monkeypatch.setattr(
        fsdp_adam, "_create_grace_adam", lambda: SimpleNamespace(ds_opt_adam=SimpleNamespace())
    )
    chunk = SimpleNamespace(
        param_and_grad_buffer=SimpleNamespace(optimizer_named_parameters=[], parameter_groups=[]),
        ddp_config=SimpleNamespace(
            fsdp_layer_by_layer_cpu_optimizer=True, fsdp_layer_by_layer_overlap_cpu_optimizer=True
        ),
    )
    param = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(RuntimeError, match="GIL"):
        fsdp_adam.FSDPCPUAdam([{"params": [param]}], [chunk], backend="grace_adam")
