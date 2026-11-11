# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import gc
import os

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import testing
from torch.distributed.tensor import DTensor

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    preprocess_state_dict_for_uneven_dtensor,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils

NUM_LAYERS = 4
HIDDEN_SIZE = 64
NUM_ATTENTION_HEADS = 4
VOCAB_SIZE = 128
SEQ_LENGTH = 32
MICRO_BATCH_SIZE = 2
NUM_MICROBATCHES = 2
NUM_STEPS = 3
SEED = 1234


def _make_batches():
    """Deterministic per-rank micro-batches (tokens, labels) for all training steps."""
    generator = torch.Generator().manual_seed(SEED + torch.distributed.get_rank())
    return [
        [
            (
                torch.randint(0, VOCAB_SIZE, (MICRO_BATCH_SIZE, SEQ_LENGTH), generator=generator),
                torch.randint(0, VOCAB_SIZE, (MICRO_BATCH_SIZE, SEQ_LENGTH), generator=generator),
            )
            for _ in range(NUM_MICROBATCHES)
        ]
        for _ in range(NUM_STEPS + 1)
    ]


def _forward(fsdp_model, tokens, labels):
    position_ids = torch.arange(SEQ_LENGTH, device="cuda").unsqueeze(0).expand_as(tokens)
    output = fsdp_model(tokens.cuda(), position_ids, None, labels=labels.cuda())
    return output.float().mean()


def _round_trip_checkpoint(model, optimizer, path):
    metadata = {"distrib_optim_sharding_type": "fsdp_dtensor"}

    def state_dict(is_loading=False):
        state = {
            "model": model.state_dict_for_save_checkpoint(),
            "optimizer": optimizer.sharded_state_dict(is_loading=is_loading, metadata=metadata),
        }
        preprocess_state_dict_for_uneven_dtensor(state)
        return state

    dcp.save(state_dict(), checkpoint_id=path)
    live_state = [
        (local, local.clone())
        for entry in optimizer.state.values()
        for value in entry.values()
        if isinstance(value, torch.Tensor)
        for local in (value._local_tensor if isinstance(value, DTensor) else value,)
    ]
    loaded = state_dict(is_loading=True)
    for local, snapshot in live_state:
        testing.assert_close(local, snapshot, rtol=0, atol=0)
    dcp.load(loaded, checkpoint_id=path)

    # Prefetch different weights before restoring. The next forward must
    # discard those cached buckets and compute with the checkpoint weights.
    perturbed = {
        name: value + 1 if isinstance(value, DTensor) else value
        for name, value in loaded["model"].items()
    }
    model.load_state_dict(perturbed)
    model.start_param_sync()
    unit = next(
        group
        for group in model.param_and_grad_buffer.parameter_groups
        if group.model_weight_buffer is not None and group.model_weight_buffer.cpu_offload
    )
    model.module.all_gather_and_wait_parameters_ready(
        unit.params, prefetch=False, wait_bucket_ready=False
    )
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])


def _train(
    strategy,
    double_buffer,
    offload=(),
    checkpoint_path=None,
    *,
    cpu_grad_accumulation=False,
    cpu_optimizer=False,
    nvme_path=None,
):
    """Train a small GPT model with Megatron-FSDP and gradient accumulation.

    Returns the per-micro-batch losses, the gradient norms, the final main weights
    and the loss of a forward pass after the last optimizer step.
    """
    torch.manual_seed(SEED)
    model_parallel_cuda_manual_seed(SEED)
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LENGTH,
    ).cuda()
    model = Float16Module(config, model)

    ddp_config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy=strategy,
        overlap_grad_reduce=True,
        overlap_param_gather=strategy == "optim_grads_params",
        use_distributed_optimizer=True,
        use_megatron_fsdp=True,
        fsdp_double_buffer=double_buffer,
        megatron_fsdp_main_grads_dtype=torch.float32,
        fsdp_cpu_offload=offload,
        fsdp_cpu_grad_accumulation=cpu_grad_accumulation,
        fsdp_cpu_optimizer=cpu_optimizer,
        fsdp_nvme_path=nvme_path,
    )
    offload_weight = "weight" in offload
    offload_grad = "gradient" in offload
    cpu_offload = offload_weight or offload_grad
    fsdp_model = FullyShardedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
        # CPU offloading sets these FSDP units by default. The baseline uses the
        # same units, so that both runs use identical buckets and collectives.
        fsdp_unit_modules=(
            None if cpu_offload else [TransformerLayer, MoETransformerLayer, MambaLayer]
        ),
    )
    buffers = fsdp_model.param_and_grad_buffer
    master_buffers = [
        group.main_weight_buffer
        for group in buffers.parameter_groups
        if group.main_weight_buffer is not None and group.main_weight_buffer.cpu_offload
    ]
    assert bool(master_buffers) == ("optimizer" in offload)
    for buf in master_buffers:
        assert buf.cpu_data_valid
        if nvme_path:
            assert buf.cpu_data is None
        else:
            assert buf.cpu_data.is_pinned()
        assert buf.data.untyped_storage().nbytes() == 0
    if cpu_offload:
        assert ddp_config.fsdp_double_buffer
        for group in fsdp_model.param_and_grad_buffer.parameter_groups:
            wbuf, gbuf = group.model_weight_buffer, group.main_grad_buffer
            if wbuf is not None and wbuf.cpu_offload:
                assert wbuf.cpu_data_valid and wbuf.data is None
                assert wbuf.cpu_data is None if nvme_path else wbuf.cpu_data.is_pinned()
            if gbuf is not None and gbuf.cpu_offload:
                assert not gbuf.cpu_data_valid and gbuf.data is None
                assert gbuf.cpu_data is None if nvme_path else gbuf.cpu_data.is_pinned()
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        [fsdp_model],
    )
    if strategy == "optim_grads_params" and offload == ("gradient",):
        fsdp_model.start_param_sync(force_sync=True)

    batches = _make_batches()
    losses, grad_norms = [], []
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        for microbatch_idx, (tokens, labels) in enumerate(batches[step]):
            if microbatch_idx < NUM_MICROBATCHES - 1:
                with fsdp_model.no_sync():
                    loss = _forward(fsdp_model, tokens, labels)
                    loss.backward()
            else:
                loss = _forward(fsdp_model, tokens, labels)
                loss.backward()
            losses.append(loss.detach())
        finalize_model_grads([fsdp_model])
        _, grad_norm, _ = optimizer.step()
        grad_norms.append(grad_norm)
        if step == 0 and checkpoint_path is not None:
            _round_trip_checkpoint(fsdp_model, optimizer, checkpoint_path)
        for buf in master_buffers:
            assert buf.data.untyped_storage().nbytes() == 0
        if master_buffers:
            tensors = [
                value._local_tensor if isinstance(value, DTensor) else value
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor) and value.ndim > 0
            ]
            if nvme_path:
                assert not tensors  # Adam moments have been written to NVMe.
            elif cpu_optimizer:
                assert tensors and all(t.device.type == "cpu" for t in tensors)
            else:
                assert tensors and all(t.device.type == "cpu" and t.is_pinned() for t in tensors)

    with buffers.optimizer_weights_on_gpu():
        main_weights = {
            name: param._local_tensor.detach().clone()
            for name, param in buffers.optimizer_named_parameters
        }
    # Forward with the model weights installed by the last optimizer step.
    with torch.no_grad():
        tokens, labels = batches[NUM_STEPS][0]
        eval_loss = _forward(fsdp_model, tokens, labels)

    if cpu_offload:
        groups = fsdp_model.param_and_grad_buffer.parameter_groups
        wbufs = [g.model_weight_buffer for g in groups if g.model_weight_buffer.cpu_offload]
        gbufs = [
            g.main_grad_buffer
            for g in groups
            if g.main_grad_buffer and g.main_grad_buffer.cpu_offload
        ]
        assert bool(wbufs) == offload_weight and bool(gbufs) == offload_grad
        staging_alloc = fsdp_model.param_and_grad_buffer.offload_grad_staging_alloc
        if offload_grad and strategy != "optim":
            assert staging_alloc is not None
            assert staging_alloc.size == 2 and not staging_alloc.using_buffer
        for wbuf in wbufs:
            assert wbuf.cpu_data_valid
        for gbuf in gbufs:
            assert gbuf.cpu_data_valid
        optimizer.zero_grad()
        assert all(not gbuf.cpu_data_valid for gbuf in gbufs)
        for buf in wbufs + gbufs:
            # Offloaded buffers only keep their persistent data in pinned CPU memory.
            assert buf.data is None and buf.cpu_data.is_pinned()

    torch.cuda.synchronize()
    del fsdp_model, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return losses, grad_norms, main_weights, eval_loss


class TestMegatronFSDPCpuOffload:
    """Megatron-FSDP CPU offloading of model weights / main gradients must reproduce the
    training results of the non-offloaded baseline bitwise."""

    # Baseline results, keyed by (sharding strategy, double buffer).
    baselines = {}

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("double_buffer", [False, True])
    @pytest.mark.parametrize(
        "offload",
        [
            pytest.param(("weight",), id="weight"),
            pytest.param(("gradient",), id="grad"),
            pytest.param(("weight", "gradient"), id="weight_grad"),
            pytest.param(("optimizer",), id="optimizer"),
            pytest.param(("optimizer", "weight", "gradient"), id="all"),
        ],
    )
    @pytest.mark.parametrize("strategy", ["optim", "optim_grads", "optim_grads_params"])
    def test_cpu_offload_matches_baseline(self, strategy, offload, double_buffer):
        if not is_torch_min_version("2.4.0"):
            pytest.skip("Megatron FSDP requires torch >= 2.4.0")
        if Utils.world_size < 2 and strategy != "optim_grads_params":
            pytest.skip("Run the FSDP-1/2 comparisons with DP >= 2.")

        key = (strategy, double_buffer)
        if key not in self.baselines:
            self.baselines[key] = _train(strategy, double_buffer)
        ref_losses, ref_grad_norms, ref_main_weights, ref_eval_loss = self.baselines[key]

        losses, grad_norms, main_weights, eval_loss = _train(strategy, double_buffer, offload)

        for i, (loss, ref_loss) in enumerate(zip(losses, ref_losses)):
            testing.assert_close(loss, ref_loss, rtol=0, atol=0, msg=f"Loss mismatch at {i}")
        assert grad_norms == ref_grad_norms, f"{grad_norms} != {ref_grad_norms}"
        assert main_weights.keys() == ref_main_weights.keys()
        for name, weight in main_weights.items():
            testing.assert_close(
                weight, ref_main_weights[name], rtol=0, atol=0, msg=f"{name} mismatch"
            )
        testing.assert_close(eval_loss, ref_eval_loss, rtol=0, atol=0)

    def test_optimizer_offload_dcp_round_trip(self, tmp_path):
        rank = torch.distributed.get_rank()
        checkpoint_path = [str(tmp_path / "checkpoint") if rank == 0 else None]
        torch.distributed.broadcast_object_list(checkpoint_path, src=0)
        expected = _train("optim_grads_params", True, ("optimizer", "weight", "gradient"))
        resumed = _train(
            "optim_grads_params", True, ("optimizer", "weight", "gradient"), checkpoint_path[0]
        )
        testing.assert_close(resumed, expected, rtol=0, atol=0)

    def test_host_gradient_accumulation_matches_gpu_accumulation(self):
        reference = _train("optim_grads_params", True, ("gradient",))
        actual = _train("optim_grads_params", True, ("gradient",), cpu_grad_accumulation=True)
        testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_cpu_adamw_matches_gpu_adamw(self):
        offload = ("optimizer", "weight", "gradient")
        reference = _train("optim_grads_params", True, offload)
        actual = _train(
            "optim_grads_params", True, offload, cpu_grad_accumulation=True, cpu_optimizer=True
        )
        testing.assert_close(actual, reference, rtol=3e-3, atol=3e-4)

    def test_nvme_offload_matches_cpu_and_resumes(self, tmp_path):
        nvme_path = os.environ.get("MEGATRON_FSDP_TEST_NVME_PATH")
        if not nvme_path:
            pytest.skip("Set MEGATRON_FSDP_TEST_NVME_PATH to a local NVMe directory with the native AIO extension installed.")
        rank = torch.distributed.get_rank()
        checkpoint = [str(tmp_path / "nvme-checkpoint") if rank == 0 else None]
        torch.distributed.broadcast_object_list(checkpoint, src=0)
        offload = ("optimizer", "weight", "gradient")
        kwargs = dict(cpu_grad_accumulation=True, cpu_optimizer=True)
        reference = _train("optim_grads_params", True, offload, **kwargs)
        actual = _train(
            "optim_grads_params", True, offload, checkpoint[0], nvme_path=nvme_path, **kwargs
        )
        testing.assert_close(actual, reference, rtol=0, atol=0)
