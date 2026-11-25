# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 dense mock pretraining units for the layer-by-layer schedule."""

import hashlib
from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class _EmbeddingUnit(nn.Module):
    def __init__(self, embedding: LanguageModelEmbedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, value: Tensor, _labels: Tensor) -> Tensor:
        return self.embedding(value, position_ids=None)


class _DecoderUnit(nn.Module):
    def __init__(self, layer: nn.Module, rotary_pos_emb: Tensor) -> None:
        super().__init__()
        self.layer = layer
        # Every layer shares the same immutable position tensor.
        self.rotary_pos_emb = rotary_pos_emb

    def forward(self, value: Tensor, _labels: Tensor) -> Tensor:
        hidden_states, _ = self.layer(value, rotary_pos_emb=self.rotary_pos_emb)
        return hidden_states


class _OutputUnit(nn.Module):
    def __init__(self, norm: nn.Module, projection: tensor_parallel.ColumnParallelLinear) -> None:
        super().__init__()
        self.norm = norm
        self.projection = projection

    def _chunk_loss(self, hidden: Tensor, labels: Tensor) -> Tensor:
        logits, _ = self.projection(hidden)
        return F.cross_entropy(logits.float(), labels, reduction="sum")

    def forward(self, value: Tensor, labels: Tensor) -> Tensor:
        hidden = self.norm(value).flatten(0, 1)
        labels = labels.transpose(0, 1).reshape(-1)
        total_loss = hidden.new_zeros((), dtype=torch.float32)
        # Bound FP32 loss memory for the 248,320-token vocabulary. During the backward
        # recompute, checkpoint each chunk so the head does not retain a full logits graph.
        for hidden_chunk, label_chunk in zip(hidden.split(512), labels.split(512), strict=True):
            if torch.is_grad_enabled():
                chunk_loss = checkpoint(
                    self._chunk_loss,
                    hidden_chunk,
                    label_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk_loss = self._chunk_loss(hidden_chunk, label_chunk)
            total_loss = total_loss + chunk_loss
        return total_loss / labels.numel()


class Qwen35DenseWorkload:
    """Build Qwen3.5-27B text decoder units on demand with independent mock data.

    Uses the Qwen3.5-27B text recipe's BF16 architecture with an untied output head
    and no MTP. The complete decoder has 64 layers. Lower layer counts use the prefix
    of the same Gated DeltaNet / Gated DeltaNet / Gated DeltaNet / attention pattern.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        seq_length: int,
        micro_batch_size: int,
        seed: int,
        device: torch.device,
        pg_collection: ProcessGroupCollection,
        tiny: bool = False,
    ) -> None:
        if min(num_layers, seq_length, micro_batch_size) < 1:
            raise ValueError("Layer count, sequence length and micro-batch size must be positive")
        if device.type != "cuda":
            raise ValueError("The Qwen3.5 layer-by-layer workload requires CUDA")
        if any(
            group is None or group.size() != 1
            for group in (pg_collection.tp, pg_collection.pp, pg_collection.cp)
        ):
            raise ValueError("The Qwen3.5 layer-by-layer workload requires TP=PP=CP=1")

        self.device = device
        self.seed = seed
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.pg_collection = pg_collection
        self.vocab_size = 1024 if tiny else 248320
        self.config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=256 if tiny else 5120,
            ffn_hidden_size=768 if tiny else 17408,
            num_attention_heads=4 if tiny else 24,
            num_query_groups=2 if tiny else 4,
            kv_channels=64 if tiny else 256,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64 if tiny else 128,
            linear_value_head_dim=64 if tiny else 128,
            linear_num_key_heads=4 if tiny else 16,
            linear_num_value_heads=12 if tiny else 48,
            attention_output_gate=True,
            qk_layernorm=True,
            apply_rope_fusion=True,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            layernorm_epsilon=1e-6,
            add_bias_linear=False,
            add_qkv_bias=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            init_method_std=0.02,
            transformer_impl="transformer_engine",
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=False,
            disable_parameter_transpose_cache=True,
        )

    def make_units(self) -> Iterator[nn.Module]:
        """Yield the embedding, decoder layers, and loss head one unit at a time."""
        config = self.config
        groups = self.pg_collection
        with torch.cuda.device(self.device):
            torch.random.default_generator.manual_seed(self.seed)
            tensor_parallel.model_parallel_cuda_manual_seed(self.seed)
            block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config, pp_rank=0
            )
            rotary_pos_emb = RotaryEmbedding(
                config.kv_channels, rotary_percent=0.25, rotary_base=10_000_000, cp_group=groups.cp
            )(self.seq_length)

            yield _EmbeddingUnit(
                LanguageModelEmbedding(
                    config,
                    self.vocab_size,
                    self.seq_length,
                    position_embedding_type="rope",
                    tp_group=groups.tp,
                )
            )
            for layer_number, layer_spec in enumerate(block_spec.layer_specs, start=1):
                yield _DecoderUnit(
                    build_module(
                        layer_spec, config=config, layer_number=layer_number, pg_collection=groups
                    ),
                    rotary_pos_emb,
                )
            yield _OutputUnit(
                build_module(
                    block_spec.layer_norm,
                    config=config,
                    hidden_size=config.hidden_size,
                    eps=config.layernorm_epsilon,
                ),
                tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    gather_output=False,
                    tp_group=groups.tp,
                ),
            )

    def make_batch(self, step: int, microbatch: int, rank: int) -> tuple[Tensor, Tensor]:
        """Generate reproducible rank-local tokens and their next-token labels."""
        key = f"{self.seed}:{step}:{microbatch}:{rank}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        rng = torch.Generator(device=self.device).manual_seed(seed)
        sequence = torch.randint(
            self.vocab_size,
            (self.micro_batch_size, self.seq_length + 1),
            generator=rng,
            device=self.device,
        )
        return sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()
