# Dense MLP Activation Offload and Recompute

## Scope

This document records the configuration contract for adding `mlp_act` support
to Megatron-Core fine-grained activation offloading and selective recompute.
The runtime hooks described below are now implemented in the dense MLP path;
numerical and GPU memory validation remains a follow-up.

`mlp_act` has a dense-MLP-specific meaning:

- The activation input is the output of `linear_fc1`, after the FC1 projection
  and before bias/activation processing.
- Offloading `mlp_act` moves that FC1 output through the fine-grained activation
  offload manager.
- Recomputing `mlp_act` reruns only bias handling, activation, GLU/clamp logic,
  and per-token scaling. It does not rerun FC1 or FC2.
- `moe_act` remains the separate name for grouped MoE expert activation logic.

## Configuration Changes

The following `TransformerConfig` fields use the new module name:

| Field | New value | Meaning |
| --- | --- | --- |
| `recompute_modules` | `"mlp_act"` | Dense activation-only output-discarding checkpoint |
| `offload_modules` | `"mlp_act"` | Dense FC1 output offload group |

The allowlists and field documentation live in
`megatron/core/transformer/transformer_config.py`.

`mlp_act` is classified with the existing output-discarding checkpoint modules
(`moe_act`, `layernorm`, `mla_up_proj`, and `gdn_norm_out`). Full `mlp`
recompute remains a normal module checkpoint.

## Validation Rules

The configuration layer enforces the following rules:

1. `mlp_act` is a valid selective recompute module.
2. `mlp_act` is a valid fine-grained offload module.
3. `mlp` and `mlp_act` must not both appear in `recompute_modules`; full MLP
   recompute already includes the activation function.
4. `mlp_act` must not be offloaded while the complete dense `mlp` is being
   recomputed; the two mechanisms would overlap and can conflict in the
   checkpoint/offload lifetime.
5. FP8 validation follows the existing activation recompute restrictions:
   delayed scaling is rejected, and Transformer Engine `>= 2.6.0dev0` is
   required for activation-only recompute.
6. The existing restrictions for CPU activation offload, PP, and CUDA graphs
   remain unchanged. This configuration change does not imply that every CUDA
   graph scope supports dense `mlp_act` yet.

The global module list is also used by hybrid models. Runtime code gates
`mlp_act` to dense MLP instances and does not enable it for routed experts or
shared experts.

## Change Summary

| Area | Implementation | File |
| --- | --- | --- |
| Parameter configuration | Added `mlp_act` to the selective-recompute and fine-grained-offload allowlists; added conflict checks for full `mlp` recompute and FP8 restrictions. | `megatron/core/transformer/transformer_config.py` |
| Offload | Starts an `mlp_act` offload group around the FC1 output and commits it after FC2, with delayed commit support for CUDA graph execution. | `megatron/core/transformer/mlp.py` |
| Recompute | Uses `CheckpointWithoutOutput` around only the dense activation helper and registers recomputation on the FC2 output. FC1 and FC2 are not recomputed. | `megatron/core/transformer/mlp.py` |
| Transformer Engine | Falls back from the op-fused MLP to the explicit MLP path when the FC1 activation tensor must be offloaded or recomputed. | `megatron/core/extensions/transformer_engine.py` |
| MoE isolation | Routed experts and shared experts do not inherit the dense `mlp_act` policy. | `megatron/core/transformer/moe/shared_experts.py` |
| Layer integration | Tracks dense `mlp_act` offload for the existing CUDA graph stream/event checks. | `megatron/core/transformer/transformer_layer.py` |
| Bridge | No conversion or public Bridge API change is required; Bridge passes these fields through deferred MCore config finalization. | `src/megatron/bridge/training/config.py` |

## Runtime Implementation

The dense MLP implementation mirrors the lifecycle already used by `moe_act` in
`megatron/core/transformer/moe/experts.py`:

1. Run FC1 and obtain `intermediate_parallel`.
2. Start an offload group named `mlp_act` on that tensor when enabled.
3. Run the activation helper normally or through `CheckpointWithoutOutput`.
4. Run FC2.
5. Register activation recompute on the FC2 output so activation data is
   regenerated before FC2 backward consumes it.
6. Commit the `mlp_act` offload group after FC2, releasing the FC1 output only
   after the FC2 forward has finished.

The activation helper preserves all existing dense MLP cases: bias,
Transformer Engine activation modules, fused bias activation, gated
SwiGLU/GeGLU, clamp values, and `per_token_scale`.

The implementation is in `megatron/core/transformer/mlp.py`. Routed experts
(`is_expert=True`) are excluded. `SharedExpertMLP` explicitly disables this
dense policy because its activation is part of the MoE shared-expert path.
When Transformer Engine's op-fuser is selected, `TEFusedMLP` raises a
`ValueError` if `mlp_act` offload or recompute is enabled; the fused path does
not expose the FC1 output required by the feature. Disable
`use_transformer_engine_op_fuser` or remove `mlp_act` from the corresponding
module list.

`TransformerLayer` marks dense MLP activation offload as participating in a
CUDA graph when the MLP scope is captured, preserving the existing stream/event
checks. The initial experiment matrix still sets `cuda_graph_impl="none"`.

## Bridge Impact

Bridge's deferred `TransformerConfig.finalize()` calls the MCore
`TransformerConfig.__post_init__()`, so no HF checkpoint conversion or public
conversion API changes are required for this configuration addition.

The Qwen fine-grained offload experiment already passes explicit overrides for
`recompute_modules`, `offload_modules`, and CUDA graph settings. Before running
the benchmark, verify that the 35B-A3B recipe resolves to the intended Qwen3.8
model identity; the current recipe naming and model identifier should not be
treated as proof of the final checkpoint identity.

## Verification To Add

- Config tests for both allowlists and the invalid `mlp` + `mlp_act` combinations.
- Dense MLP forward/backward parity with and without `mlp_act` recompute.
- Fine-grained offload tests checking output, gradients, and peak memory.
- Saved-tensor or memory-snapshot checks confirming the FC1 output, dtype,
  shape, element count, and copy bytes.
- Separate BF16 and MXFP8 coverage; CUDA graphs remain disabled for the first
  experiment matrix until their dense `mlp_act` integration is reviewed.
