# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the independently installed libaio extension."""

import mmap
import os
import tempfile
import unittest
from pathlib import Path

import megatron_nvme_aio


class NativeAioTest(unittest.TestCase):
    def test_round_trip_across_queue_windows(self):
        size = (128 * 1024 * 18) + 4096
        worker = megatron_nvme_aio.create_worker()
        with (
            tempfile.TemporaryDirectory(dir=os.environ.get("MEGATRON_FSDP_TEST_NVME_PATH")) as root,
            mmap.mmap(-1, size) as source,
            mmap.mmap(-1, size) as destination,
        ):
            path = str(Path(root) / "gradient-0")
            # Distinct blocks catch any offset reuse when the queue is replenished.
            source[:] = b"".join(i.to_bytes(4, "little") * 1024 for i in range(size // 4096))
            megatron_nvme_aio.transfer(worker, source, path, True)
            megatron_nvme_aio.transfer(worker, destination, path, False)
            self.assertEqual(source[:], destination[:])

    def test_worker_recovers_after_failed_io(self):
        worker = megatron_nvme_aio.create_worker()
        with (
            tempfile.TemporaryDirectory(dir=os.environ.get("MEGATRON_FSDP_TEST_NVME_PATH")) as root,
            mmap.mmap(-1, 8192) as buffer,
        ):
            path = str(Path(root) / "master-0")
            with self.assertRaises(OSError):
                megatron_nvme_aio.transfer(worker, buffer, path, False)

            with memoryview(buffer)[:4096] as one_page:
                one_page[:] = b"x" * 4096
                megatron_nvme_aio.transfer(worker, one_page, path, True)
            with self.assertRaisesRegex(OSError, "short pread"):
                megatron_nvme_aio.transfer(worker, buffer, path, False)

            with memoryview(buffer)[:4096] as one_page:
                one_page[:] = b"\0" * 4096
                megatron_nvme_aio.transfer(worker, one_page, path, False)
                self.assertEqual(one_page.tobytes(), b"x" * 4096)

    def test_requires_alignment_and_writable_read_buffer(self):
        worker = megatron_nvme_aio.create_worker()
        with mmap.mmap(-1, 8192) as buffer:
            with memoryview(buffer)[1:4097] as unaligned:
                with self.assertRaises(ValueError):
                    megatron_nvme_aio.transfer(worker, unaligned, "unused", True)
            with memoryview(buffer).toreadonly() as read_only:
                with self.assertRaises(ValueError):
                    megatron_nvme_aio.transfer(worker, read_only, "unused", False)


if __name__ == "__main__":
    unittest.main()
