# Qwen3.5-27B layer-by-layer capacity test

Run on one GB200 node with four GPUs available to the same CUDA/NCCL environment. The sweep uses four-way full sharding with TP, PP and CP each set to 1. It keeps the Qwen3.5-27B dense text-model widths and varies decoder depth. The full model has 64 decoder layers. Training uses BF16, randomly initialized weights, and synthetic token batches.

## 32-layer offload cases

Run the numbered cases from the repository root. Both use the Qwen3.5-27B text attention mix, 32 decoder layers, sequence length 4096, MBS 1, GBS 16, four micro-batches per rank, and 10 optimizer steps.

```bash
study=examples/scale-down/02-layer-by-layer/01-baseline
bash "$study/01_cpu_offload.sh" --dry-run
bash "$study/01_cpu_offload.sh" --nvme /tmp
bash "$study/02_nvme_offload.sh" --nvme /tmp
```

`01_cpu_offload.sh` uses `cpu_param + cpu_act`. `02_nvme_offload.sh` uses `nvme_param + nvme_act`. Both create private scratch directories under `/tmp` by default. Set `--nvme DIR` or `NVME_PATH` if the node's local NVMe is mounted elsewhere.

Both cases use Nsight Systems **2026.6.1** to capture CUDA, NVTX, and GPU metrics at 20 kHz. The default window is iteration 8 (step 7, zero-based). With `--train-iters 3`, capture covers iteration 3. The launcher checks the version at `NsightSystems-2026.6.1/bin/nsys` under the repository root, or at the executable set through `NSYS_BIN`. Use `--profile none` for an unprofiled run. [Nsight Systems CLI reference](https://docs.nvidia.com/nsight-systems/UserGuide/)

On each node the launcher checks IPv4 addresses on up interfaces. It honors valid `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` values. Automatic selection tries `eth0`, then `bond1`, then an IPv4 default-route interface. NCCL uses an exact interface name. The selected interfaces and Nsight version are recorded in `run_info.txt`.

Results land under `results/02-layer-by-layer/01-baseline/<case>/model_.../node0/`. The directory contains `launch.json`, `command.txt`, `run_info.txt`, `train.log`, `summary.csv`, source provenance, and `train.nsys-rep` when profiling is enabled. Use `--print-result-dir` to resolve the path or `--output-dir` for a fresh repeat.

A pass requires `status=0`, a nonempty report when profiling, and **1320 D2H / 2640 H2D** transfers per rank for 10 steps. A three-step run expects 396 and 792 copies per rank.

## Capacity sweep

```bash
bash examples/scale-down/02-layer-by-layer/run_sweep.sh \
  --nvme-dir /path/to/local/nvme --layers 16 32 64
```

Defaults are sequence length 4096, micro-batch size 1, four micro-batches per rank per step, three optimizer steps, `nvme_param`, and `nvme_act`. Select `--parameter-tier cpu_param` to keep parameter shards in host RAM. Use an already configured Megatron CUDA environment with Transformer Engine and the Gated DeltaNet kernels. Set `PYTHON` to the interpreter in that environment if needed.

Choose the activation tier independently of the parameter tier.

| `--activation-tier` | Layer-boundary activations |
| --- | --- |
| `gpu_act` | Retain in GPU memory through backward. |
| `cpu_act` | Offload to pinned host memory and reload for forward and backward. |
| `nvme_act` | Offload through pinned host buffers to NVMe and reload for forward and backward. |

All three modes still rematerialize the layer internals and use NVMe for most optimizer states. Pass a local `--nvme-dir` in every mode.

Each decoder layer processes all micro-batches before advancing. Two GPU weight slots alternate across layers, and all micro-batches of a layer share the same gathered weights. Two activation slots stage the next micro-batch's boundary input. Backward rematerializes each layer and reduce-scatters its FP32 accumulated gradient. The next layer's weight read and H2D copy can overlap current-layer computation. The shared communicator issues the gradient reduce-scatter before the next weight all-gather. A CPU worker runs AdamW concurrently with GPU work while a pinned gradient slot is available. The first decoder layer keeps its FP32 master weights and moments in RAM. Other units read and write those states on NVMe. The prototype uses fixed learning rate, with gradient scaling and gradient clipping disabled.

When the existing `megatron_nvme_aio` extension from `src/nvme/deepspeed` is installed, the trainer uses direct I/O. Otherwise it uses buffered I/O, with file sync and cache eviction advice where supported. Checkpoint and rollback support are outside the prototype's training entry point.

For a quick smoke test, add `--tiny --layers 4 --seq-length 128 --microbatches 2 --steps 2`. Add `--dry-run` to inspect the commands. To find the limit after 64 layers passes, try more depths with `--layers 64 80 96`, then narrow the interval around a capacity failure. The NVMe filesystem must have room for all four ranks' files. At full depth the FP32 master weights and two Adam moment tensors alone require roughly 324 GB in total. Staging and temporary files add to that footprint.

To verify CPU activation offload, run the smoke test with `--activation-tier cpu_act`. The log's `LAYER_BY_LAYER_RESULT` records `activation_d2h_completed_by_rank` and `activation_h2d_completed_by_rank` after CUDA synchronization. Expect `[20, 20, 20, 20]` and `[40, 40, 40, 40]` for this smoke test. In general, each rank offloads `(layers + 1) * microbatches * steps` boundary tensors and reloads each twice. `nvme_act` has the same copy counts and additionally reads and writes activation files. `gpu_act` reports zero for both counts.

Each sweep creates a new result directory under `./results`, or at the unused path passed as `--results-dir`. It retains the Git revision, SHA-256 hashes of the layer-by-layer sources in `source_sha256.json`, commands, training logs, and `summary.csv` with the status, final loss, peak GPU memory, and activation copy counts per rank. Each case uses a fresh scratch directory on NVMe. A pass requires every rank to finish the requested steps with finite losses, finite gradient shards, and the expected activation copy counts. CUDA allocation errors, explicit host allocation errors, and NVMe space or quota errors get separate statuses. Other errors require inspecting the corresponding log. The reported depth is the largest case observed to pass in the sweep.
