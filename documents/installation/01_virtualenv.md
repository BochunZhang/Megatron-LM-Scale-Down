# 环境安装

为避免每次起任务都要通过 pip 安装依赖, 我们使用 virtualenv 创建虚拟环境并复用 /opt/venv/bin/python 的 site-packages.

理论上可以使用 --system-site-packages 复用指定 python 的 site-packages, 但实测下来会指向错误的 site-packages.

解决方案是利用 custom_sys_path.pth, 每行定义一个第三方 python 的 site-packages 路径.

具体流程
```bash
# 安装 virtual env
pip3 install virtualenv

# 在指定路径创建 virtual env
virtualenv /path/to/new-env
# 或指定 python 解析器路径
virtualenv -p /path/to/python /path/to/new-env


# 查看目标 python 的 site-packages 路径
python -m site

# sys.path = [
#     '/Users/zhangbochun/Desktop/AI-Infra/Megatron/Megatron-Scale-Down/megatron-lm',
#     '/Users/zhangbochun/.pyenv/versions/3.12.13/lib/python312.zip',
#     '/Users/zhangbochun/.pyenv/versions/3.12.13/lib/python3.12',
#     '/Users/zhangbochun/.pyenv/versions/3.12.13/lib/python3.12/lib-dynload',
#     '/Users/zhangbochun/.pyenv/versions/3.12.13/lib/python3.12/site-packages', -> 目标路径
# ]
# USER_BASE: '/Users/zhangbochun/.local' (exists)
# USER_SITE: '/Users/zhangbochun/.local/lib/python3.12/site-packages' (doesn't exist)
# ENABLE_USER_SITE: True


vim /path/to/new-env/lib/python3.xx/site-packages/custom_sys_path.pth
cat << 'EOF' > /path/to/new-env/lib/python3.xx/site-packages/custom_sys_path.pth
/Users/zhangbochun/.pyenv/versions/3.12.13/lib/python3.12/site-packages
EOF

source /path/to/new-env/bin/activate
```