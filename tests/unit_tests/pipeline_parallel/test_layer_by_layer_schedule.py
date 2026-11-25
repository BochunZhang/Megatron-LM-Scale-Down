# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from collections import Counter
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import fine_grained_activation_offload
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    get_forward_backward_func,
    layer_by_layer_forward_backward_no_pipelining,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TransformerConfig(
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


@pytest.mark.parametrize("activation_offload", [False, True], ids=["gpu-activations", "cpu-activations"])
def test_layer_by_layer_schedule_matches_no_pipeline_update(
    cuda_device, monkeypatch, activation_offload
):
    layer_major = get_forward_backward_func(pp_size=1, layer_by_layer=True)
    microbatch_major = get_forward_backward_func(pp_size=1)
    assert layer_major is layer_by_layer_forward_backward_no_pipelining
    assert microbatch_major is forward_backward_no_pipelining

    model = _Model().to(cuda_device)
    model.config.layer_by_layer_activation_offload = activation_offload
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

    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_cpu", track_copy_to_cpu)
    monkeypatch.setattr(fine_grained_activation_offload, "_copy_tensor_to_device", track_copy_to_device)

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
    expected_boundaries = (len(model.layers) + 1) * len(batches) - 1 if activation_offload else 0
    assert len(original_boundaries) == restored_boundaries == expected_boundaries
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
        if event[0] in ("enter", "exit") and not event[1].startswith("activation ")
    ]
    assert stage_events == expected_nvtx_events
    assert Counter(event for event in nvtx_events if event[0] in ("start", "end")) == Counter(
        expected_microbatch_events
    )
    expected_transfers = []
    if activation_offload:
        offloaded = [("transformer.0", mb) for mb in microbatches]
        offloaded.extend(("transformer.1", mb) for mb in microbatches[:-1])
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
    assert [event for event in nvtx_events if event[1].startswith("activation ")] == expected_transfers
    torch.testing.assert_close(losses, expected_losses)
    for param, ref_param in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(param.grad, ref_param.grad)

    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(baseline.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.state_dict(), baseline.state_dict())
