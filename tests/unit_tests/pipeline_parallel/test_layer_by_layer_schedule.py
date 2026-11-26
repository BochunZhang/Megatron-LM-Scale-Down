# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
from collections import Counter
from contextlib import closing, contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from weakref import ref

import pytest
import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import fine_grained_activation_offload
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    forward_step_calc_loss,
    get_forward_backward_func,
    layer_by_layer_forward_backward_no_pipelining,
)
from megatron.core.tensor_parallel import mappings
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
from megatron.core.transformer.transformer_config import TransformerConfig


class _Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TransformerConfig(
            num_layers=2,
            hidden_size=4,
            num_attention_heads=1,
            calculate_per_token_loss=True,
            grad_scale_func=lambda loss: 2 * loss,
        )
        self.model_type = ModelType.encoder_or_decoder
        self.embedding = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))
        self.head = nn.Linear(4, 1)
        self.layer_calls = []

    def set_input_tensor(self, inputs):
        assert inputs == [None]


class _Plan:
    def __init__(self, model, microbatch, tokens):
        self.model = model
        self.microbatch = microbatch
        self.tokens = tokens

    def preprocess(self):
        return self.model.embedding(self.tokens)

    def num_layers(self):
        return len(self.model.layers)

    def forward_layer(self, index, hidden_states):
        self.model.layer_calls.append((index, self.microbatch, torch.is_grad_enabled()))
        return self.model.layers[index](hidden_states).tanh()

    def postprocess(self, hidden_states):
        return self.model.head(hidden_states).square()


class _CudaTensor:
    is_cuda = True
    device = torch.device("cuda")

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def numel(self):
        return self.data.numel()

    def detach(self):
        return self


def _forward_step(data_iterator, model, *, return_schedule_plan=False):
    microbatch, tokens = next(data_iterator)
    plan = _Plan(model, microbatch, tokens)

    def loss_func(output):
        loss = output.sum()
        count = output.new_tensor(tokens.numel(), dtype=torch.int)
        return loss, count, {"loss": loss.detach()}

    if return_schedule_plan:
        return plan, loss_func
    hidden_states = plan.preprocess()
    for layer in range(plan.num_layers()):
        hidden_states = plan.forward_layer(layer, hidden_states)
    return plan.postprocess(hidden_states), loss_func


@pytest.mark.parametrize("balancing", ["global_aux_loss", "quantile_balancing"])
def test_layer_by_layer_moe_recompute_preserves_router_state(
    balancing, monkeypatch, tmp_path, mock_nvme_aio
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mappings, "get_tensor_model_parallel_group_if_none", lambda group: group)
    group = SimpleNamespace(size=lambda: 1)
    pg_collection = SimpleNamespace(tp=group, pp=group, cp=group, tp_cp=group, tp_dp_cp=group)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_ffn_hidden_size=8,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type=balancing,
        moe_router_enable_expert_bias=balancing == "global_aux_loss",
        moe_aux_loss_coeff=0.2 if balancing == "global_aux_loss" else 0.0,
        moe_z_loss_coeff=0.1,
        calculate_per_token_loss=False,
        cuda_graph_scope=None,
        layer_by_layer_activation_nvme_path=str(tmp_path),
    )

    class ExpertLayer(nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.router = router_module.TopKRouter(config, pg_collection)
            self.router.set_layer_number(layer_number)
            self.experts = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))

        def forward(self, hidden_states):
            probs, _ = self.router(hidden_states.unsqueeze(1))
            expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=1)
            return (probs.unsqueeze(-1) * expert_outputs).sum(dim=1)

    torch.manual_seed(1234)
    model = _Model(config)
    model.layers = nn.ModuleList(ExpertLayer(i + 1) for i in range(config.num_layers))
    baseline = deepcopy(model)
    batches = [torch.tensor(ids) for ids in ((1, 2), (3, 4, 5, 6), (5, 7))]
    tracker = MoEMetricsTracker()
    monkeypatch.setattr(router_module, "get_moe_metrics_tracker", lambda: tracker)
    monkeypatch.setattr(router_module.MoEAuxLossAutoScaler, "main_loss_backward_scale", None)

    baseline_losses = []
    data = iter(enumerate(batches))
    for _ in batches:
        output, loss_func = _forward_step(data, baseline)
        loss, _ = forward_step_calc_loss(
            baseline,
            output,
            loss_func,
            baseline.config,
            None,
            False,
            len(batches),
            baseline_losses,
            cp_group_size=1,
            is_last_stage=True,
        )
        loss.backward()
    expected_metrics = {name: metric.values.clone() for name, metric in tracker.metrics.items()}
    tracker = MoEMetricsTracker()
    losses = layer_by_layer_forward_backward_no_pipelining(
        forward_step_func=_forward_step,
        data_iterator=iter(enumerate(batches)),
        model=model,
        num_microbatches=len(batches),
        seq_length=4,
        micro_batch_size=1,
        pg_collection=pg_collection,
    )

    torch.testing.assert_close(losses, baseline_losses)
    torch.testing.assert_close(
        {name: metric.values for name, metric in tracker.metrics.items()}, expected_metrics
    )
    torch.testing.assert_close(dict(model.named_buffers()), dict(baseline.named_buffers()))
    for actual, expected in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-5)
    for module in (model, baseline):
        torch.optim.SGD(module.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.state_dict(), baseline.state_dict())


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("Megatron activation checkpointing requires CUDA RNG state")
    with torch.cuda.device(int(os.environ.get("LOCAL_RANK", "0"))):
        yield torch.device("cuda", torch.cuda.current_device())


def test_layer_by_layer_checkpoint_reloads_only_the_current_group(monkeypatch):
    class Stream:
        def wait_event(self, event):
            assert event.recorded

        def wait_stream(self, stream):
            assert isinstance(stream, Stream)

    class Event:
        def __init__(self):
            self.recorded = False

        def record(self, stream):
            assert isinstance(stream, Stream)
            self.recorded = True

    class Tensor:
        is_cuda = True
        device = torch.device("cuda")

        def __init__(self, value):
            self.value = value

        def numel(self):
            return 1

        def detach(self):
            return self

        def record_stream(self, stream):
            assert isinstance(stream, Stream)

    hooks = []
    saved, reloaded = [], []

    def capture_hooks(pack, unpack):
        hooks.append((pack, unpack))
        return nullcontext()

    def copy_to_cpu(tensor, *, pin_memory):
        assert pin_memory
        saved.append(tensor.value)
        return tensor.value

    def copy_to_device(device, host, *, non_blocking):
        assert device.type == "cuda" and non_blocking
        reloaded.append(host)
        return Tensor(host)

    manager = SimpleNamespace(d2h_stream=Stream(), h2d_stream=Stream())
    monkeypatch.setattr(
        fine_grained_activation_offload.PipelineOffloadManager, "get_instance", lambda: manager
    )
    monkeypatch.setattr(fine_grained_activation_offload, "saved_tensors_hooks", capture_hooks)
    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_cpu", copy_to_cpu)
    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_device", copy_to_device)
    monkeypatch.setattr(torch.cuda, "current_stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", nullcontext)
    monkeypatch.setattr(torch.cuda.nvtx, "range", nullcontext)

    offload = fine_grained_activation_offload.LayerByLayerActivationOffload()
    decoder_tags = []
    for mb in range(3):
        with offload.checkpoint(0, "transformer.0", microbatch=mb):
            pack_decoder, unpack = hooks[-1]
            decoder_tags.append(pack_decoder(Tensor(mb)))
            assert saved == list(range(mb))
        assert saved == list(range(mb + 1))
    assert decoder_tags == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    output_tags = []
    for mb in range(3):
        with offload.checkpoint(1, "output", microbatch=mb):
            pack_output, _ = hooks[-1]
            output_tags.append(pack_output(Tensor(10 + mb)))
    assert output_tags == [(0, 1, 0), (1, 1, 0), (2, 1, 0)]
    assert saved == [0, 1, 2, 10, 11, 12]
    assert not reloaded

    with offload.backward_group(1):
        assert reloaded == [12]
        with pytest.raises(RuntimeError, match="outside its backward"):
            unpack(decoder_tags[0])
        assert unpack(output_tags[0]).value == 10
        assert reloaded == [12, 11, 10]
        assert [unpack(tag).value for tag in reversed(output_tags[1:])] == [12, 11]

    with offload.backward_group(0):
        assert reloaded == [12, 11, 10, 2]
        assert unpack(decoder_tags[2]).value == 2
        assert reloaded == [12, 11, 10, 2, 1]
        assert unpack(decoder_tags[1]).value == 1
        assert reloaded == [12, 11, 10, 2, 1, 0]
        assert unpack(decoder_tags[0]).value == 0


@pytest.fixture
def nvme_operations(monkeypatch):
    operations = []

    def allocate_file(fd, offset, size):
        assert os.fstat(fd).st_size == offset
        os.ftruncate(fd, offset + size)
        operations.append(("reserve", (offset, size)))

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(os, "posix_fallocate", allocate_file, raising=False)
    return operations


@pytest.fixture
def mock_nvme_aio(monkeypatch, nvme_operations):
    def create_worker(queue_depth):
        assert queue_depth == 16
        return object()

    def submit(_worker, array, filename, write, offset):
        buffer = memoryview(array)
        assert offset % 4096 == 0 and buffer.nbytes % 4096 == 0
        assert array.ctypes.data % os.sysconf("SC_PAGE_SIZE") == 0
        path = Path(filename)
        assert offset + buffer.nbytes <= path.stat().st_size
        operation = "write" if write else "read"
        nvme_operations.append((f"submit_{operation}", (path, offset)))
        return operation, buffer, path, offset

    def wait(request):
        operation, buffer, path, offset = request
        with path.open("r+b" if operation == "write" else "rb") as file:
            file.seek(offset)
            if operation == "write":
                file.write(buffer)
            else:
                assert file.readinto(buffer) == buffer.nbytes
        nvme_operations.append((operation, (path, offset)))

    monkeypatch.setitem(
        sys.modules,
        "megatron_nvme_aio",
        SimpleNamespace(create_worker=create_worker, submit=submit, wait=wait),
    )
    return nvme_operations


@pytest.fixture
def mock_deepspeed_aio(monkeypatch, nvme_operations):
    class AioHandle:
        def __init__(self, **kwargs):
            assert kwargs == {
                "block_size": 128 * 1024,
                "queue_depth": 64,
                "single_submit": False,
                "overlap_events": True,
                "intra_op_parallelism": 1,
            }
            self.pending = None

        def _submit(self, operation, buffer, filename, offset):
            assert self.pending is None and offset % 4096 == 0
            assert buffer.data_ptr() % os.sysconf("SC_PAGE_SIZE") == 0
            assert buffer.numel() % 4096 == 0
            path = Path(filename)
            assert offset + buffer.numel() <= path.stat().st_size
            nvme_operations.append((f"submit_{operation}", (path, offset)))
            self.pending = (operation, buffer, path, offset)
            return 0

        def async_pwrite(self, buffer, filename, offset):
            return self._submit("write", buffer, filename, offset)

        def async_pread(self, buffer, filename, offset):
            return self._submit("read", buffer, filename, offset)

        def wait(self):
            operation, buffer, path, offset = self.pending
            with path.open("r+b" if operation == "write" else "rb") as file:
                file.seek(offset)
                if operation == "write":
                    file.write(memoryview(buffer.numpy()))
                else:
                    data = bytearray(file.read(buffer.numel()))
                    buffer.copy_(torch.frombuffer(data, dtype=torch.uint8))
            nvme_operations.append((operation, (path, offset)))
            self.pending = None
            return 1

    module = SimpleNamespace(aio_handle=AioHandle)
    builder = SimpleNamespace(load=lambda: module)
    monkeypatch.setitem(
        sys.modules, "deepspeed.ops.op_builder", SimpleNamespace(AsyncIOBuilder=lambda: builder)
    )
    return nvme_operations


@pytest.mark.parametrize("backend", ["native", "deepspeed"])
def test_layer_by_layer_nvme_defers_write_wait(tmp_path, monkeypatch, request, backend):
    operations = request.getfixturevalue(
        "mock_nvme_aio" if backend == "native" else "mock_deepspeed_aio"
    )
    allocate = torch.empty
    nvtx_ranges = []

    def allocate_on_cpu(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return allocate(*args, **kwargs)

    def record_nvtx_range(label):
        nvtx_ranges.append(label)
        return nullcontext()

    monkeypatch.setattr(torch, "empty", allocate_on_cpu)
    monkeypatch.setattr(torch.cuda.nvtx, "range", record_nvtx_range)
    offload = fine_grained_activation_offload._LayerByLayerNVMeActivations(
        str(tmp_path), backend=backend
    )
    first = torch.arange(2050, dtype=torch.bfloat16).view(2, 1025).t()
    second = torch.tensor(17, dtype=torch.int64)
    try:
        offload.swap.reserve((4 if backend == "native" else 3) * 4096)
        saved_first = offload.offload((0, 0, 0), first)
        assert [name for name, _ in operations] == ["reserve", "submit_write"]
        saved_second = offload.offload((1, 0, 0), second)
        assert [name for name, _ in operations[1:]] == ["submit_write", "write", "submit_write"]
        assert (saved_first.offset, saved_second.offset) == (0, 8192)

        # Prefetch an already written region while the second write is outstanding.
        stage = {0: SimpleNamespace(_tensors={(0, 0, 0): saved_first})}
        offload.prefetch(stage)
        assert operations[-1][0] == "submit_read"

        # A demand read of the other key waits for its own write, then leaves the
        # prefetched data available for the first activation without rereading it.
        torch.testing.assert_close(offload.reload(saved_second), second)
        assert [name for name, _ in operations[-4:]] == ["read", "write", "submit_read", "read"]
        events = list(operations)
        torch.testing.assert_close(offload.reload(saved_first), first)
        assert operations == events

        # A queued write still owns its pinned buffer when the step exits.
        offload.offload((2, 3, 4), second)
        offload.prefetch(stage)
    finally:
        offload.close()
    assert [name for name, _ in operations[-4:]] == ["submit_write", "submit_read", "write", "read"]
    reservations = [region for name, region in operations if name == "reserve"]
    if backend == "native":
        assert reservations == [(0, 4 * 4096)]
    else:
        assert reservations == [(0, 3 * 4096), (3 * 4096, 4096)]
        assert [
            name for name in nvtx_ranges if name.startswith("DeepSpeed AIO wait activation write ")
        ] == [
            "DeepSpeed AIO wait activation write micro-batch 0 transformer 0 tensor 0",
            "DeepSpeed AIO wait activation write micro-batch 1 transformer 0 tensor 0",
            "DeepSpeed AIO wait activation write micro-batch 2 transformer 3 tensor 4",
        ]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("backend", ["native", "deepspeed"])
def test_layer_by_layer_nvme_reuses_file_and_handles(tmp_path, monkeypatch, request, backend):
    operations = request.getfixturevalue(
        "mock_nvme_aio" if backend == "native" else "mock_deepspeed_aio"
    )
    allocate = torch.empty

    def allocate_on_cpu(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate_on_cpu)
    monkeypatch.setattr(torch.cuda.nvtx, "range", nullcontext)
    model = nn.Module()
    handles = None
    inode = None
    slots = None
    for value in (11, 29):
        first = torch.tensor(value)
        second = torch.tensor(-value)
        with closing(
            fine_grained_activation_offload.LayerByLayerActivationOffload(
                nvme_path=str(tmp_path), nvme_backend=backend, model=model
            )
        ) as step:
            current_handles = (ref(step._nvme.swap._reader), ref(step._nvme.swap._writer))
            if handles is None:
                handles = current_handles
            assert current_handles == handles
            step.reserve_nvme((_CudaTensor(first),), num_groups=2)
            (file,) = tmp_path.glob("*/activations.bin")
            if inode is None:
                inode = file.stat().st_ino
            assert file.stat().st_ino == inode
            assert file.stat().st_size == 8192
            saved_first = step._nvme.offload((0, 0, 0), first)
            saved_second = step._nvme.offload((0, 1, 0), second)
            assert (saved_first.offset, saved_second.offset) == (0, 4096)
            torch.testing.assert_close(step._nvme.reload(saved_second), second)
            torch.testing.assert_close(step._nvme.reload(saved_first), first)
            current_slots = {name: ref(slot) for name, slot in step._nvme._slots.items()}
            if slots is None:
                slots = current_slots
            assert current_slots == slots
        assert file.stat().st_size == 8192
        assert all(handle() is not None for handle in handles)

    reservations = [region for name, region in operations if name == "reserve"]
    assert reservations == [(0, 8192)]
    fine_grained_activation_offload.release_layer_by_layer_activation_aio(model)
    assert all(handle() is None for handle in handles)
    assert all(slot() is None for slot in slots.values())
    assert not list(tmp_path.iterdir())


def test_layer_by_layer_native_drains_read_when_write_fails(tmp_path, monkeypatch, mock_nvme_aio):
    allocate = torch.empty

    def allocate_on_cpu(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return allocate(*args, **kwargs)

    extension = sys.modules["megatron_nvme_aio"]
    wait = extension.wait

    def fail_second_write(request):
        operation, _, _, offset = request
        if operation == "write" and offset == 4096:
            raise OSError("failed NVMe write")
        wait(request)

    monkeypatch.setattr(torch, "empty", allocate_on_cpu)
    monkeypatch.setattr(torch.cuda.nvtx, "range", nullcontext)
    monkeypatch.setattr(extension, "wait", fail_second_write)
    model = nn.Module()
    value = torch.tensor(7)
    with (
        pytest.raises(OSError, match="failed NVMe write"),
        closing(
            fine_grained_activation_offload.LayerByLayerActivationOffload(
                nvme_path=str(tmp_path), model=model
            )
        ) as step,
    ):
        step.reserve_nvme((_CudaTensor(value),), num_groups=2)
        saved = step._nvme.offload((0, 0, 0), value)
        step._nvme.offload((0, 1, 0), value)
        step._nvme.prefetch({0: SimpleNamespace(_tensors={(0, 0, 0): saved})})
    assert [name for name, _ in mock_nvme_aio[-2:]] == ["submit_read", "read"]
    assert not list(tmp_path.iterdir())


def test_layer_by_layer_nvme_uses_separate_reusable_slots(tmp_path, monkeypatch, mock_nvme_aio):
    values = [
        torch.arange(2050, dtype=torch.bfloat16).view(2, 1025).t(),
        torch.tensor(17, dtype=torch.int64),
    ]
    observed_slots = {"d2h": [], "h2d": []}
    pinned_slots = []
    read_buffers = []
    allocate = torch.empty
    copy = torch.Tensor.copy_
    submit = sys.modules["megatron_nvme_aio"].submit

    def track_submit(worker, buffer, path, write, offset):
        if not write:
            read_buffers.append((buffer.ctypes.data, buffer.nbytes))
        return submit(worker, buffer, path, write, offset)

    def allocate_on_cpu(*args, pin_memory=False, **kwargs):
        # Exercise the byte views without requiring a CUDA allocator.
        tensor = allocate(*args, **kwargs)
        if pin_memory:
            assert tensor.device.type == "cpu" and tensor.dtype == torch.uint8
            pinned_slots.append((tensor.data_ptr(), tensor.numel()))
        return tensor

    def copy_to_host(host, tensor, *, non_blocking=False):
        if isinstance(tensor, _CudaTensor):
            assert not non_blocking
            observed_slots["d2h"].append(host.data_ptr())
            mock_nvme_aio.append(("d2h", host.shape))
            tensor = tensor.data
        return copy(host, tensor, non_blocking=non_blocking)

    def copy_to_device(device, host, *, non_blocking):
        assert device.type == "cuda" and not non_blocking
        observed_slots["h2d"].append(host.data_ptr())
        mock_nvme_aio.append(("h2d", host.shape))
        return host.clone()

    hooks = []

    def capture_hooks(pack, unpack):
        hooks.append((pack, unpack))
        return nullcontext()

    monkeypatch.setattr(torch, "empty", allocate_on_cpu)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_to_host)
    monkeypatch.setattr(sys.modules["megatron_nvme_aio"], "submit", track_submit)
    monkeypatch.setattr(torch.cuda, "Event", object)
    monkeypatch.setattr(torch.cuda.nvtx, "range", nullcontext)
    monkeypatch.setattr(fine_grained_activation_offload, "saved_tensors_hooks", capture_hooks)
    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_device", copy_to_device)

    offload = fine_grained_activation_offload.LayerByLayerActivationOffload(nvme_path=str(tmp_path))
    offload.reserve_nvme(tuple(map(_CudaTensor, values)), num_groups=1)
    tags = []
    for mb, tensor in enumerate(values):
        with offload.checkpoint(0, "transformer.0", microbatch=mb):
            pack, unpack = hooks[-1]
            tags.append(pack(_CudaTensor(tensor)))
            empty = _CudaTensor(torch.empty(0))
            assert pack(empty) is empty
    assert [name for name, _ in mock_nvme_aio] == [
        "reserve",
        "d2h",
        "submit_write",
        "write",
        "d2h",
        "submit_write",
    ]
    with offload.backward_group(0):
        for tag, expected in zip(tags, values, strict=True):
            torch.testing.assert_close(unpack(tag), expected)
    written = [region for name, region in mock_nvme_aio if name == "write"]
    assert written == [region for name, region in mock_nvme_aio if name == "read"]
    (file,) = tmp_path.glob("mfsdp-rank0-*/activations.bin")
    assert written == [(file, 0), (file, 8192)]
    assert len(set(observed_slots["d2h"])) == len(set(observed_slots["h2d"])) == 1
    forward, backward = pinned_slots
    write_ptr = observed_slots["d2h"][0]
    assert write_ptr % os.sysconf("SC_PAGE_SIZE") == 0
    assert forward[0] <= write_ptr <= forward[0] + forward[1] - 8192
    read_ptr = observed_slots["h2d"][0]
    assert read_ptr % os.sysconf("SC_PAGE_SIZE") == 0
    assert backward[0] <= read_ptr <= backward[0] + backward[1] - 8192
    assert read_buffers == [(read_ptr, 8192), (read_ptr, 4096)]
    offload.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("activation_tier", ["gpu", "cpu", "nvme", "nvme_deepspeed"])
def test_layer_by_layer_schedule_matches_no_pipeline_update(
    cuda_device, monkeypatch, tmp_path, request, activation_tier
):
    layer_major = get_forward_backward_func(pp_size=1, layer_by_layer=True)
    microbatch_major = get_forward_backward_func(pp_size=1)
    assert layer_major is layer_by_layer_forward_backward_no_pipelining
    assert microbatch_major is forward_backward_no_pipelining

    model = _Model().to(cuda_device)
    activation_offload = activation_tier != "gpu"
    model.config.layer_by_layer_activation_offload = activation_tier == "cpu"
    if activation_tier.startswith("nvme"):
        if activation_tier == "nvme_deepspeed":
            nvme_operations = request.getfixturevalue("mock_deepspeed_aio")
            model.config.layer_by_layer_activation_nvme_backend = "deepspeed"
        else:
            nvme_operations = request.getfixturevalue("mock_nvme_aio")
        model.config.layer_by_layer_activation_nvme_path = str(tmp_path)
    baseline = _Model().to(cuda_device)
    baseline.load_state_dict(model.state_dict())
    batches = [torch.tensor(ids, device=cuda_device) for ids in ((1, 2), (3, 4, 5), (6,))]
    num_tokens = sum(batch.numel() for batch in batches)
    microbatches = range(len(batches))
    layer_indices = range(len(model.layers))
    replay_order = tuple(reversed(microbatches)) if activation_offload else tuple(microbatches)
    single_rank_group = SimpleNamespace(size=lambda: 1)
    pg_collection = SimpleNamespace(tp=None, pp=single_rank_group, cp=single_rank_group)
    finalized = []
    accumulations = []
    nvtx_events = []
    original_boundaries = []
    restored_boundaries = 0
    for name, param in model.named_parameters():
        param.register_post_accumulate_grad_hook(lambda _, name=name: accumulations.append(name))

    copy_to_cpu = fine_grained_activation_offload._copy_tensor_to_cpu
    copy_to_device = fine_grained_activation_offload._copy_tensor_to_device
    nvme_offload = fine_grained_activation_offload._LayerByLayerNVMeActivations.offload

    def track_copy_to_cpu(tensor, *, pin_memory, cpu_tensor_pool=None):
        packed = copy_to_cpu(tensor, pin_memory=pin_memory, cpu_tensor_pool=cpu_tensor_pool)
        assert tensor.is_cuda and packed.device.type == "cpu" and packed.is_pinned()
        assert packed.shape == tensor.shape and packed.dtype == tensor.dtype
        original_boundaries.append(StorageWeakRef(tensor.untyped_storage()))
        return packed

    def track_copy_to_device(device, packed, *, non_blocking):
        nonlocal restored_boundaries
        assert device.type == "cuda" and packed.is_pinned()
        if restored_boundaries == 0:
            assert all(storage.expired() for storage in original_boundaries)
        restored_boundaries += 1
        return copy_to_device(device, packed, non_blocking=non_blocking)

    def track_copy_to_nvme(offload, tag, tensor):
        original_boundaries.append(StorageWeakRef(tensor.untyped_storage()))
        return nvme_offload(offload, tag, tensor)

    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_cpu", track_copy_to_cpu)
    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_device", track_copy_to_device)
    monkeypatch.setattr(
        fine_grained_activation_offload._LayerByLayerNVMeActivations, "offload", track_copy_to_nvme
    )

    @contextmanager
    def capture_nvtx_range(name):
        nvtx_events.append(("enter", name))
        try:
            yield
        finally:
            nvtx_events.append(("exit", name))

    def capture_nvtx_start(name):
        range_id = len(nvtx_events)
        nvtx_events.append(("start", name))
        return range_id

    def capture_nvtx_end(range_id):
        event, name = nvtx_events[range_id]
        assert event == "start"
        nvtx_events.append(("end", name))

    monkeypatch.setattr(torch.cuda.nvtx, "range", capture_nvtx_range)
    monkeypatch.setattr(torch.cuda.nvtx, "range_start", capture_nvtx_start)
    monkeypatch.setattr(torch.cuda.nvtx, "range_end", capture_nvtx_end)

    def finalize(models, num_tokens, *, pg_collection, force_all_reduce):
        assert force_all_reduce
        finalized.append((models[0], num_tokens.item(), pg_collection))
        for param in models[0].parameters():
            param.grad.div_(num_tokens)

    def run(schedule, module):
        module.config.finalize_model_grads_func = finalize
        return schedule(
            forward_step_func=_forward_step,
            data_iterator=iter(enumerate(batches)),
            model=[module],
            num_microbatches=len(batches),
            seq_length=3,
            micro_batch_size=1,
            pg_collection=pg_collection,
            force_all_reduce=True,
        )

    losses = run(layer_major, model)
    expected_losses = run(microbatch_major, baseline)
    expected_boundaries = (len(model.layers) + 1) * len(batches) if activation_offload else 0
    if activation_tier == "cpu":
        expected_boundaries -= 1
    assert len(original_boundaries) == restored_boundaries == expected_boundaries
    if activation_tier.startswith("nvme"):
        assert Counter(
            name for name, _ in nvme_operations if name in ("write", "read")
        ) == dict.fromkeys(("write", "read"), expected_boundaries)
        (file,) = tmp_path.glob("*/activations.bin")
        assert file.stat().st_size == expected_boundaries * 4096
        assert nvme_operations[0] == ("reserve", (0, expected_boundaries * 4096))
        fine_grained_activation_offload.release_layer_by_layer_activation_aio(model)
        assert not list(tmp_path.iterdir())
    assert finalized == [(model, num_tokens, pg_collection), (baseline, num_tokens, pg_collection)]
    assert model.layer_calls == [
        (layer, mb, activation_offload) for layer in layer_indices for mb in microbatches
    ] + [(layer, mb, True) for layer in reversed(layer_indices) for mb in replay_order]
    assert Counter(accumulations) == {name: 1 for name, _ in model.named_parameters()}
    expected_nvtx_events = []
    for name in ("embedding", "transformer.0", "transformer.1", "output"):
        expected_nvtx_events.append(("enter", f"forward.{name}"))
        for microbatch in range(len(batches)):
            expected_nvtx_events.extend(
                (event, f"fwd.microbatch.{microbatch}.{name}") for event in ("enter", "exit")
            )
        expected_nvtx_events.append(("exit", f"forward.{name}"))
    expected_microbatch_events = []
    for stage in ("output", "transformer.1", "transformer.0", "embedding"):
        stage_begin = ("enter", f"backward.{stage}")
        stage_end = ("exit", f"backward.{stage}")
        expected_nvtx_events.append(stage_begin)
        if stage.startswith("transformer."):
            for microbatch in replay_order:
                expected_nvtx_events.extend(
                    (event, f"recompute.microbatch.{microbatch}.{stage}")
                    for event in ("enter", "exit")
                )
        expected_nvtx_events.append(stage_end)
        for microbatch in range(len(batches)):
            name = f"bwd.microbatch.{microbatch}.{stage}"
            start, end = ("start", name), ("end", name)
            expected_microbatch_events.extend((start, end))
            assert (
                nvtx_events.index(stage_begin)
                < nvtx_events.index(start)
                < nvtx_events.index(end)
                < nvtx_events.index(stage_end)
            )
            if stage.startswith("transformer.") and not activation_offload:
                last_recompute = ("exit", f"recompute.microbatch.{len(batches) - 1}.{stage}")
                assert nvtx_events.index(last_recompute) < nvtx_events.index(start)
            elif stage.startswith("transformer."):
                replay = ("enter", f"recompute.microbatch.{microbatch}.{stage}")
                assert nvtx_events.index(start) < nvtx_events.index(replay) < nvtx_events.index(end)
    stage_events = [
        event
        for event in nvtx_events
        if event[0] in ("enter", "exit")
        and not event[1].startswith(("activation ", "DeepSpeed AIO wait activation write "))
    ]
    assert stage_events == expected_nvtx_events
    assert Counter(event for event in nvtx_events if event[0] in ("start", "end")) == Counter(
        expected_microbatch_events
    )
    expected_transfers = []
    if activation_offload:
        offloaded = [("transformer.0", mb) for mb in microbatches]
        offloaded.extend(
            ("transformer.1", mb)
            for mb in (microbatches[:-1] if activation_tier == "cpu" else microbatches)
        )
        offloaded.extend(("output", mb) for mb in microbatches)
        for direction, boundaries in (
            ("offloading", offloaded), ("reloading", reversed(offloaded))
        ):
            for stage, mb in boundaries:
                name = f"activation {direction} {stage} microbatch={mb}"
                expected_transfers.extend((event, name) for event in ("enter", "exit"))

        for stage, mb in offloaded:
            if stage == "output":
                continue
            forward_done = ("exit", f"fwd.microbatch.{mb}.{stage}")
            offload = f"activation offloading {stage} microbatch={mb}"
            next_forward = ("enter", f"fwd.microbatch.{mb + 1}.{stage}")
            end = next_forward if mb + 1 in microbatches else ("exit", f"forward.{stage}")
            assert (
                nvtx_events.index(forward_done)
                < nvtx_events.index(("enter", offload))
                < nvtx_events.index(("exit", offload))
                < nvtx_events.index(end)
            )
            reload_done = ("exit", f"activation reloading {stage} microbatch={mb}")
            replay_start = ("enter", f"recompute.microbatch.{mb}.{stage}")
            assert nvtx_events.index(reload_done) < nvtx_events.index(replay_start)

        for stage in ("transformer.1", "transformer.0"):
            backward_starts = [
                name
                for event, name in nvtx_events
                if event == "start" and name.endswith(f".{stage}")
            ]
            assert backward_starts == [f"bwd.microbatch.{mb}.{stage}" for mb in replay_order]
            first_replay = ("enter", f"recompute.microbatch.{len(batches) - 1}.{stage}")
            last_reload = ("enter", f"activation reloading {stage} microbatch=0")
            assert nvtx_events.index(first_replay) < nvtx_events.index(last_reload)
    assert [
        event
        for event in nvtx_events
        if event[1].startswith(("activation offloading ", "activation reloading "))
    ] == expected_transfers
    torch.testing.assert_close(losses, expected_losses)
    for param, ref_param in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(param.grad, ref_param.grad)

    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(baseline.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.state_dict(), baseline.state_dict())
