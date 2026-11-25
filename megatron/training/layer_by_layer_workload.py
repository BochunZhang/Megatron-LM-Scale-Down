# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 model geometry for the native layer-by-layer experiments."""

import torch
from torch.nn import functional as F

from megatron.core.transformer.transformer_config import TransformerConfig

QWEN35_LAYER_BY_LAYER_VOCAB_SIZE = 248320
QWEN35_LAYER_BY_LAYER_TINY_VOCAB_SIZE = 1024


def build_qwen35_layer_by_layer_config(num_layers: int, *, tiny: bool) -> TransformerConfig:
    """Build the default Qwen3.5 transformer geometry for the layer-by-layer workload."""
    return TransformerConfig(
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
