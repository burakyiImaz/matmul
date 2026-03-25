from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


def _is_cuda_compatible(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.ndim != 2 or b.ndim != 2:
        return False
    if a.device.type != "cuda" or b.device.type != "cuda":
        return False
    if a.shape[1] != b.shape[0]:
        return False
    if a.dtype != b.dtype:
        return False
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            offs_k += BLOCK_K

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not _HAS_TRITON or not _is_cuda_compatible(a, b):
        return torch.matmul(a, b)

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


@dataclass
class GpuMatmulPolicy:
    enabled: bool = True
    min_dim: int = 256
    min_mnk: int = 4_000_000
    benchmark_repeats: int = 40


class GpuMatmulAutoDispatch:
    """Shape-aware GPU dispatcher that picks torch or triton by measured latency."""

    def __init__(self, policy: Optional[GpuMatmulPolicy] = None) -> None:
        self.policy = policy or GpuMatmulPolicy()
        self._cache: Dict[Tuple[int, int, int, torch.dtype, int], str] = {}

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._can_try_gpu_opt(a, b):
            return torch.matmul(a, b)

        m, k = a.shape
        _, n = b.shape
        key = (m, k, n, a.dtype, a.device.index if a.device.index is not None else 0)

        backend = self._cache.get(key)
        if backend is None:
            backend = self._benchmark_pick_backend(a, b)
            self._cache[key] = backend

        if backend == "triton":
            return triton_matmul(a, b)
        return torch.matmul(a, b)

    def _can_try_gpu_opt(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        if not self.policy.enabled:
            return False
        if not _is_cuda_compatible(a, b):
            return False
        if not _HAS_TRITON:
            return False

        m, k = a.shape
        _, n = b.shape
        if min(m, k, n) < self.policy.min_dim:
            return False
        if (m * k * n) < self.policy.min_mnk:
            return False

        return True

    def _benchmark_pick_backend(self, a: torch.Tensor, b: torch.Tensor) -> str:
        # Warmup
        for _ in range(8):
            _ = torch.matmul(a, b)
            _ = triton_matmul(a, b)
        torch.cuda.synchronize(device=a.device)

        torch_t = self._bench(lambda: torch.matmul(a, b), a.device, self.policy.benchmark_repeats)
        triton_t = self._bench(lambda: triton_matmul(a, b), a.device, self.policy.benchmark_repeats)

        return "triton" if triton_t < torch_t else "torch"

    @staticmethod
    def _bench(fn, device: torch.device, repeats: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            _ = fn()
        end.record()
        torch.cuda.synchronize(device=device)
        return float(start.elapsed_time(end)) / repeats


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This module targets NVIDIA GPUs.")
    else:
        dispatch = GpuMatmulAutoDispatch()
        a = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

        c = dispatch.matmul(a, b)
        print("Output:", tuple(c.shape), c.dtype, c.device)
