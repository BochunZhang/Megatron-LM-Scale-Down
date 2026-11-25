# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

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


def test_layer_by_layer_mixes_gpu_adam_and_staged_grace_adam(monkeypatch):
    params, buckets = [], []
    for index in range(4):
        weight = torch.tensor([2.0])
        gradient = torch.tensor([0.25 * (index + 1)])
        param = torch.nn.Parameter(weight)
        param._local_tensor = param.detach()
        param.orig_param = object()
        gradient._local_tensor = gradient
        param.grad = gradient
        param_idx = {param.orig_param: 0}

        def make_buffer(tensor, *, cpu_offload=False):
            def get_item(item_id, *, only_shard, data=None):
                assert item_id == 0 and only_shard
                return tensor if data is None else data

            return SimpleNamespace(
                bucket_id=index,
                data=tensor,
                data_size=tensor.numel(),
                cpu_offload=cpu_offload,
                nvme=None,
                param_idx=param_idx,
                get_item=get_item,
                read_cpu_data=lambda: tensor,
            )

        params.append(param)
        buckets.append(
            SimpleNamespace(
                params=[param.orig_param],
                main_weight_buffer=make_buffer(weight, cpu_offload=index > 0),
                model_weight_buffer=make_buffer(weight.to(torch.bfloat16)),
                main_grad_buffer=make_buffer(gradient),
            )
        )

    events = []

    @contextmanager
    def nvtx_range(message):
        label = message()
        if label.startswith(("fsdp.layer_by_layer.", "GraceAdam ")):
            events.append(label.split(".")[-1])
        yield

    stream = Mock()
    stream.record_event.side_effect = [
        Mock(synchronize=lambda: events.append("ready 1")),
        Mock(synchronize=lambda: events.append("ready 2")),
        Mock(synchronize=lambda: events.append("ready 3")),
    ]
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: Mock())
    empty = torch.empty
    def empty_host(*args, pin_memory=False, **kwargs):
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_host)
    monkeypatch.setattr(fsdp_adam, "nvtx_range", nvtx_range)

    grace_adam = SimpleNamespace(param_groups=[{"params": []}], state={})

    def grace_step():
        (weight,) = grace_adam.param_groups[0]["params"]
        assert weight.grad.dtype == torch.float32
        weight.sub_(weight.grad)
        grace_adam.state[weight]["step"] += 1

    grace_adam.step = grace_step
    monkeypatch.setattr(fsdp_adam, "_create_grace_adam", lambda: grace_adam)
    buffers = SimpleNamespace(
        optimizer_named_parameters=[(str(index), param) for index, param in enumerate(params)],
        param_to_param_group={param.orig_param: index for index, param in enumerate(params)},
        parameter_groups=buckets,
        cpu_offload_d2h_stream=stream,
        commit_cpu_optimizer_bucket=Mock(),
    )
    chunk = SimpleNamespace(
        param_and_grad_buffer=buffers,
        ddp_config=SimpleNamespace(fsdp_layer_by_layer_cpu_optimizer=True),
    )
    optimizer = fsdp_adam.FSDPCPUAdam([{"params": params}], [chunk], backend="grace_adam", lr=0.1)
    optimizer.step()

    assert events == [
        "GPUAdamW bucket=0",
        "gradient_d2h bucket=1",
        "gradient_d2h bucket=2",
        "ready 1",
        "GraceAdam bucket=1",
        "gradient_d2h bucket=3",
        "ready 2",
        "GraceAdam bucket=2",
        "ready 3",
        "GraceAdam bucket=3",
    ]
    reference = torch.nn.Parameter(torch.tensor([2.0]))
    reference.grad = torch.tensor([0.25])
    torch.optim.AdamW([reference], lr=0.1).step()
    torch.testing.assert_close(params[0], reference)
    torch.testing.assert_close(buckets[0].model_weight_buffer.data, reference.to(torch.bfloat16))
    assert buffers.commit_cpu_optimizer_bucket.call_count == 3
    for index, param in enumerate(params[1:], start=1):
        torch.testing.assert_close(param, torch.tensor([2.0 - 0.25 * (index + 1)]))
        assert optimizer.state[param]["step"].item() == 1
