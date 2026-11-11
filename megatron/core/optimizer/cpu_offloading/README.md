## How to use ?

Add these flags to enable optimizer cpu offload in MCore.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## Configuration Recommendations

Gradient copy from GPU to CPU, CPU optimizer step, and subsequent parameter copy from CPU to GPU can be time-consuming operations, and it is recommended to use the flag `--overlap-cpu-optimizer-d2h-h2d` to execute them concurrently.

## Megatron-FSDP CPU storage with GPU updates

Select one or more components with `--fsdp-cpu-offload`. For example, to offload
all three components with fully sharded data parallelism, use:

```bash
--use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params \
--ckpt-format fsdp_dtensor --fsdp-cpu-offload optimizer weight gradient
```

- `optimizer` stores the local master weights and optimizer tensor state in pinned CPU memory.
  It restores the full local working set for each GPU optimizer step and writes back the updates.
- `weight` stores model compute weights in pinned CPU memory and reloads them before all-gather.
- `gradient` stores the main gradient accumulation in pinned CPU memory and reloads the final
  local shards for the GPU optimizer.

Programmatic callers can set `DistributedDataParallelConfig.fsdp_cpu_offload` to a tuple
such as `("optimizer", "gradient")`. Weight and gradient offload enable GPU double buffering.
The `optimizer` selection requires MCore-owned master weights and GPU Adam with CUDA graphs disabled.
TE fused MoE with gradient accumulation fusion requires gradient buckets during
forward and is incompatible with offload configurations that double-buffer gradients.
