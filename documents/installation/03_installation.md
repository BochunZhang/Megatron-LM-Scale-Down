# 安装 Megatron-LM 和 Megaton-Bridge

Megatron-Bridge 和 Megatron-LM 支持使用 uv 管理 package.

有两个特殊的 package 会影响编译: flash-mla 和 fast-hadamard-transform
1. flash-mla 基于 cuda 12.9+ 编译, 不支持 cuda 13.0+, 因为后者将 cuda/include 路径下的 std/utility 移动到了 cuda/include/cccl, 导致编译的时候找不到 std/utility
2. fast-hadamard-transform 缺少 cu130 + torch2.14 的 wheel, 需要手动编译而不是从 release 下载 wheel

解决方案
1. 设置 c++ 编译的 flag 里面包含 /usr/local/cuda/include/cccl 即可
- export FLASH_MLA_DISABLE_SM90=1 NVCC_THREADS=16 CFLAGS="-I/usr/local/cuda/include/cccl" CXXFLAGS="-I/usr/local/cuda/include/cccl"  (参考 dockerfile)
- export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl:$CPLUS_INCLUDE_PATH
2. FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE 跳过 download 尝试, 直接本地编译


```bash
# 创建 python 3.12 虚拟环境 (当前版本只支持 python 3.12, 最好用这个版本)
virtualenv -p /path/to/python3.12 /path/to/new-env
source /path/to/new-env/bin/activate

uv install setuptools
uv install torch torchvision # 安装最新版本的 torch

# 进入 Megatron-Bridge, 确保 Megatron-LM 已经正确安装在 3rdparty 路径下
# --active 使用激活的 venv
# --all-groups
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl:$CPLUS_INCLUDE_PATH
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE uv sync --active --all-groups

# 安装后 torch 和 transformer-engine 不见了, 切换到仓库外重新安装
uv pip install torch torchvision
uv pip install 
```



----



```bash
# 创建 python 3.12 虚拟环境 (当前版本只支持 python 3.12, 最好用这个版本)
virtualenv -p /path/to/python3.12 /path/to/new-env
source /path/to/new-env/bin/activate

uv install setuptools
uv install torch torchvision # 安装最新版本的 torch

# Megatron-Bridge uv 对 fast-hadamard-transform 1.10 的支持存在问题, 没有启用 --no-build-isolation
# 编译时会找不到 torch

git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
git checkout 1cc807efbd6cc001df359822d60bf6052dd66859   # 切换至 1.1.0
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE pip install -v --no-build-isolation .

# 进入 Megatron-Bridge, 确保 Megatron-LM 已经正确安装在 3rdparty 路径下
# --active 使用激活的 venv
# --all-groups
uv sync --active --all-groups
```