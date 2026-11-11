# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the optional Megatron-FSDP native NVMe backend."""

import os
import sys
from pathlib import Path

from setuptools import Extension, setup

if sys.platform != "linux":
    raise SystemExit("megatron-nvme-aio requires Linux libaio.")

nvtx_include = Path(
    os.environ.get(
        "NVTX_INCLUDE_DIR", Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"
    )
)

setup(
    name="megatron-nvme-aio",
    version="0.1.0",
    license_files=["LICENSE"],
    ext_modules=[
        Extension(
            "megatron_nvme_aio",
            ["aio.cpp"],
            include_dirs=[str(nvtx_include)],
            libraries=["aio", "dl"],
            extra_compile_args=["-std=c++17", "-pthread"],
            extra_link_args=["-pthread"],
            language="c++",
        )
    ],
)
