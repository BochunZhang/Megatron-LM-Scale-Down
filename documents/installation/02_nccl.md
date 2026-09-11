# 升级 NCCL 版本至 2.30.7+


访问 https://developer.nvidia.com/nccl/nccl-legacy-downloads, 下载合适版本的 NCCL.

e.g. 下载 Local installers (ARM)
https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.30.7/ubuntu2404/aarch64sbsa/nccl-local-repo-ubuntu2404-2.30.7-cuda13.3_1.0-1_arm64.deb/


```bash
dpkg -i /path/to/nccl/deb # nccl-local-repo-ubuntu2404-2.30.7-cuda13.3_1.0-1_arm64.deb

# 会提示生成的 keyring.gpg 位置, 并复制到 /usr/share/keyrings
cp /path/to/keyring /usr/share/keyrings

apt-get update
apt-get install -y libnccl2 libnccl-dev
```

