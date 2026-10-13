# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Read small device-side HybridEP metadata on the host without a stream sync.

``_HybridEPManager.setup_metadata`` pads uneven dispatch inputs to the EP-group
max token count and reads that max back with ``.item()``, which synchronizes the
whole compute stream. This module replaces the ``.item()`` with an SM copy: a
single-CTA Triton kernel stores the values into pinned host memory, then
publishes a sequence number with a system-scope release store. The host spins on
that flag instead of calling into the driver.
"""

import threading
import time
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


# Spin iterations between GIL releases and timeout checks.
_SPINS_PER_YIELD = 1024
_WAIT_TIMEOUT_SECONDS = 300.0


if HAVE_TRITON:

    @triton.jit(do_not_specialize=["seq", "num_values"])
    def _sm_copy_with_flag_kernel(
        src_ptr, dst_ptr, flag_ptr, seq, num_values, BLOCK: tl.constexpr
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < num_values
        values = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, values, mask=mask)
        # Order every thread's value store before the flag store; the system-scope
        # release makes the values visible to the host no later than the flag.
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, seq, sem="release", scope="sys")


class HostCountsReader:
    """Copy a small int64 CUDA tensor to the host with an SM copy and a host spin-wait.

    Args:
        capacity (int): Initial number of int64 values the pinned buffer can hold.
    """

    def __init__(self, capacity: int = 256):
        if not HAVE_TRITON:
            raise ImportError("moe_hybridep_sync_with_sm requires Triton.")
        self._capacity = 0
        self._host_values: Optional[torch.Tensor] = None
        self._host_flag = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._flag_view = self._host_flag.numpy()
        self._seq = 0
        self._verified = False
        # Acquiring a lock is a full barrier on the host, so reads of the values
        # cannot be reordered before the read that observed the flag.
        self._fence = threading.Lock()
        self._ensure_capacity(capacity)

    def _ensure_capacity(self, num_values: int) -> None:
        if num_values <= self._capacity:
            return
        self._capacity = triton.next_power_of_2(max(num_values, 1))
        self._host_values = torch.zeros(self._capacity, dtype=torch.int64, pin_memory=True)

    def read(self, src: torch.Tensor) -> torch.Tensor:
        """Return a CPU copy of ``src`` once all prior work on the current stream is done.

        Args:
            src (torch.Tensor): 1-D int64 CUDA tensor produced on the current stream.

        Returns:
            torch.Tensor: A CPU int64 tensor with the same values as ``src``.
        """
        assert src.is_cuda and src.dtype == torch.int64 and src.dim() == 1, (
            f"Expected a 1-D int64 CUDA tensor, got {src.dtype} on {src.device} "
            f"with shape {tuple(src.shape)}"
        )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("moe_hybridep_sync_with_sm cannot run inside CUDA graph capture.")
        num_values = src.numel()
        self._ensure_capacity(num_values)
        self._seq += 1
        _sm_copy_with_flag_kernel[(1,)](
            src,
            self._host_values,
            self._host_flag,
            self._seq,
            num_values,
            BLOCK=self._capacity,
            num_warps=1,
        )

        with torch.cuda.nvtx.range("hybrid-ep sm-sync wait"):
            spins = 0
            deadline = None
            while int(self._flag_view[0]) != self._seq:
                spins += 1
                if spins % _SPINS_PER_YIELD == 0:
                    now = time.monotonic()
                    if deadline is None:
                        deadline = now + _WAIT_TIMEOUT_SECONDS
                    elif now > deadline:
                        raise RuntimeError(
                            f"Timed out after {_WAIT_TIMEOUT_SECONDS}s waiting for the "
                            f"HybridEP SM-copy flag (expected {self._seq}, "
                            f"saw {int(self._flag_view[0])})."
                        )
                    # Let other Python threads run during long waits.
                    time.sleep(0)
            with self._fence:
                values = self._host_values[:num_values].clone()

        if not self._verified:
            # One-time check that the device can store into this pinned buffer.
            expected = src.cpu()
            if not torch.equal(values, expected):
                raise RuntimeError(
                    "HybridEP SM copy returned wrong values: "
                    f"{values.tolist()} != {expected.tolist()}"
                )
            self._verified = True
        return values


_reader: Optional[HostCountsReader] = None


def read_device_counts_with_sm(src: torch.Tensor) -> torch.Tensor:
    """Copy a small int64 CUDA tensor to the host without ``cudaStreamSynchronize``.

    Args:
        src (torch.Tensor): 1-D int64 CUDA tensor produced on the current stream.

    Returns:
        torch.Tensor: A CPU int64 tensor with the same values as ``src``.
    """
    global _reader
    if _reader is None:
        _reader = HostCountsReader()
    return _reader.read(src)
