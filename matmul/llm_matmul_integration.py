from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

try:
    import fastgemm  # type: ignore
except ImportError:
    fastgemm = None

try:
    from gpu_matmul_triton import GpuMatmulAutoDispatch  # type: ignore
except ImportError:
    GpuMatmulAutoDispatch = None


@dataclass
class LlmMatmulPolicy:
    """Runtime policy for deciding fastgemm vs torch.matmul per operation."""

    enabled: bool = True
    min_dim: int = 512
    min_mnk: int = 8_000_000
    allowed_ops: set[str] = field(
        default_factory=lambda: {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "ffn_up",
            "ffn_gate",
            "ffn_down",
        }
    )
    op_overrides: Dict[str, int] = field(default_factory=dict)


class LlmMatmulEngine:
    """Matmul dispatcher for LLM hot paths on CPU workloads.

    Notes:
    - fastgemm in this repository currently supports CPU + float32/float64 + 2D row-major inputs.
    - Non-matching workloads automatically fall back to torch.matmul.
    """

    def __init__(self, policy: Optional[LlmMatmulPolicy] = None) -> None:
        self.policy = policy or LlmMatmulPolicy()
        self.gpu_dispatch = GpuMatmulAutoDispatch() if GpuMatmulAutoDispatch is not None else None

    def configure_fastgemm(self, blas_min_dim: int = 512, blas_min_mnk: int = 8_000_000) -> None:
        """Apply BLAS fallback thresholds inside fastgemm if module is available."""
        if fastgemm is None:
            return
        fastgemm.set_blas_min_dim(int(blas_min_dim))
        fastgemm.set_blas_min_mnk(int(blas_min_mnk))

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        op_name: str = "ffn_up",
    ) -> torch.Tensor:
        """Compute x @ weight.T (+ bias), with optional fastgemm acceleration.

        Supported input shapes:
        - x: [M, K] or [B, T, K]
        - weight: [N, K]
        """
        if x.ndim not in (2, 3):
            return torch.nn.functional.linear(x, weight, bias)

        if x.ndim == 3:
            bsz, seq, in_dim = x.shape
            x2d = x.reshape(bsz * seq, in_dim)
            out2d = self.matmul(x2d, weight.t(), op_name=op_name)
            out = out2d.reshape(bsz, seq, weight.shape[0])
        else:
            out = self.matmul(x, weight.t(), op_name=op_name)

        if bias is not None:
            out = out + bias
        return out

    def matmul(self, a: torch.Tensor, b: torch.Tensor, op_name: str = "") -> torch.Tensor:
        """Dispatch order: Triton(CUDA) -> fastgemm(CPU) -> torch.matmul fallback."""
        if self.gpu_dispatch is not None and a.device.type == "cuda" and b.device.type == "cuda":
            return self.gpu_dispatch.matmul(a, b)

        if not self._can_use_fastgemm(a, b, op_name):
            return torch.matmul(a, b)

        # Detach to avoid autograd-tracked NumPy conversion; fallback remains torch path.
        a_np = a.detach().contiguous().numpy()
        b_np = b.detach().contiguous().numpy()

        c_np = fastgemm.matmul(a_np, b_np)
        return torch.from_numpy(c_np)

    def _can_use_fastgemm(self, a: torch.Tensor, b: torch.Tensor, op_name: str) -> bool:
        if not self.policy.enabled:
            return False
        if fastgemm is None:
            return False
        if op_name and op_name not in self.policy.allowed_ops:
            return False

        # fastgemm binding is designed for CPU float32/float64 contiguous 2D arrays.
        if a.ndim != 2 or b.ndim != 2:
            return False
        if a.device.type != "cpu" or b.device.type != "cpu":
            return False
        if a.dtype != b.dtype:
            return False
        if a.dtype not in (torch.float32, torch.float64):
            return False
        if a.shape[1] != b.shape[0]:
            return False

        m, k = a.shape
        _, n = b.shape

        op_min_dim = self.policy.op_overrides.get(op_name, self.policy.min_dim)
        if min(m, k, n) < op_min_dim:
            return False

        mnk = m * k * n
        if mnk < self.policy.min_mnk:
            return False

        return True


if __name__ == "__main__":
    # Minimal usage example
    engine = LlmMatmulEngine()
    engine.configure_fastgemm(blas_min_dim=512, blas_min_mnk=8_000_000)

    x = torch.randn(1, 1536, 768, dtype=torch.float32)
    w = torch.randn(3072, 768, dtype=torch.float32)

    y = engine.linear(x, w, op_name="ffn_up")
    print("Output shape:", tuple(y.shape))
