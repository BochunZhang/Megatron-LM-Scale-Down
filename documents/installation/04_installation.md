Megatron-Bridge & Megatron-LM 使用 uv 管理项目的 python 环境, 其环境配置最好放在 docker 虚拟机里面直接完成.

流程为:
1. 拉取 docker, 已配置 python=3.12, torch==2.13.0, torchvision==0.28.0, triton==3.7.1, 这些 package 安装在 system python 的 site-packages 里面, uv venv 时配置了 --system-site-packages, 因此不会被 uv sync 删除
2. 修改 torch 源码, 增加一个 patch
3. 在 sytem python 里面安装 hybrid-ep, 拉取源码后也需要执行一个 patch 后再编译
4. 安装 Megatron-Bridge, 其中 transformer-engine 锁定为 2.17 版本, 且需要从源码编译

Megatron-LM 的 main 分支使用 transformer engine 2.18 + nccl 2.30.7+, 支持新特性 nccl_ep, core_v0.19.0 使用 transformer engine 2.17.

Megatron-LM 虚拟机默认使用 hybrid-ep, 而非 deep-ep, 为测试不同分支, 我们统一配置为
- python 3.12.12 (通过 uv 下载 clang 编译版本), 存储在 .python 路径下
- torch==2.13.0, torchvision==0.28.0, triton==3.7.1
- transformer_engin 2.18 + nccl 2.30.7
- transformers 最新版本
- .tmp 下
- .env-base 存放 python3.12-torch2.13, 安装 torch==2.13.0, torchvision==0.28.0, triton==3.7.1
- .venv/python3.12-torch2.13-engin2.18-hybird-ep 存放 hybrid ep 虚拟环境
- .venv/python3.12-torch2.13-engin2.18-deep-ep 存放 deep ep 虚拟环境

应该按照如下流程来实现
```bash
# Set environment variables for GB200
export NVTE_BUILD_NUM_PHILOX_ROUNDS=3
export TORCH_CUDA_ARCH_LIST="10.0"  # Blackwell architecture
export NVCC_THREADS=16
export FLASH_MLA_DISABLE_SM90=1

# Set UV environment
export UV_HTTP_TIMEOUT=120
export UV_LINK_MODE=copy

export CFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"
export CXXFLAGS="-I/usr/local/cuda/include/cccl -DNDEBUG"

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE

# 1. 安装 python
uv python install 3.12.12 --install-dir .python/ --trusted_host https://github.com

# 2. 创建虚拟环境
virtualenv -p .python/cpython-3.12.12-linux-aarch64-gnu .env-base/python3.12-torch2.13
source activate .env-base/python3.12-torch2.13/bin/activate

# 3. 安装 torch
pip install -y triton==3.7.1 torch==2.13.0 torchvision==0.28.0
pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13
pip install -y nvidia-cutlass-dsl[cu13]==4.5.0

# 应用 torch patch
bash install/torch-patch.sh

# 测试
python - <<'PY'
import torch
from torch.library import Library, _clear_torch_ops_cache

namespace_name = "_mbridge_finalizer_test"
library = Library(namespace_name, "DEF")
library.define("uncached_op() -> None")
namespace = getattr(torch.ops, namespace_name)
original_getattr = type(namespace).__getattr__


def fail_getattr(self, name):
    raise AssertionError(f"Torch op cache cleanup called __getattr__ for {name}")


type(namespace).__getattr__ = fail_getattr
try:
    _clear_torch_ops_cache({f"{namespace_name}::uncached_op"})
finally:
    type(namespace).__getattr__ = original_getattr
PY

# 4. 构建 hybrid-ep venv 环境
uv venv .venv/python3.12-torch2.13-engin2.18-hybird-ep --python .env-base/python3.12-torch2.13/bin/python --system-site-packages
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> ".venv/python3.12-torch2.13-engin2.18-hybird-ep/bin/activate"
source .venv/python3.12-torch2.13-engin2.18-hybird-ep/bin/activate

echo $VIRTUAL_ENV
echo $UV_PROJECT_ENVIRONMENT

uv sync --only-group build --inextra
uv sync --link-mode copy --all-extras --all-groups --no-group diffusion --inextra
uv sync --upgrade-package transformers

bash install/install-hybrid-ep.sh


# 5. 构建 deep-ep 虚拟环境
uv venv .venv/python3.12-torch2.13-engin2.18-deep-ep --python .env-base/python3.12-torch2.13/bin/python --system-site-packages
echo "export UV_PROJECT_ENVIRONMENT=\"\$VIRTUAL_ENV\"" >> ".venv/python3.12-torch2.13-engin2.18-deep-ep/bin/activate"
source .venv/python3.12-torch2.13-engin2.18-deep-ep/bin/activate

echo $VIRTUAL_ENV
echo $UV_PROJECT_ENVIRONMENT

uv sync --only-group build --inextra
uv sync --link-mode copy --all-extras --all-groups --no-group diffusion --inextra
uv sync --upgrade-package transformers

bash install/install-deep-ep.sh

```