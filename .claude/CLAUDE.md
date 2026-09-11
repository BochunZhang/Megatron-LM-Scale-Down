# Claude Code Configuration for Megatron-LM

## Python Import Path Mapping

请在解析 Python import 时使用以下本地源码路径：
- `import transformer_engine` → `/Users/zhangbochun/Desktop/AI-Infra/envs/ali-gb200/python-package/TransformerEngine`
- `import torch` → `/Users/zhangbochun/Desktop/AI-Infra/envs/ali-gb200/python-package/PyTorch`
- `import megatron` → 本项目自身 (`/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/megatron-lm`)

## NCCL Source Path

请在分析 NCCL 时使用以下本地源码路径：
- **NCCL 2.28.9** → `/Users/zhangbochun/Desktop/AI-Infra/NCCL/nccl-2.28.9`（默认版本）
- **NCCL 2.30.4** → `/Users/zhangbochun/Desktop/AI-Infra/NCCL/nccl-2.30.4`

默认使用 NCCL 2.28.9 分析。
