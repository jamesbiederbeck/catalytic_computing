"""PyTorch backend for Cook-Mertz v2.

v1 bugs fixed:
  - 'iimport torch' typo → proper import with availability check
  - float64 tensors → int64 throughout (float precision silently
    breaks modular arithmetic for primes > ~2^53)
  - Explicit % p after every operation

Uses int64 tensors for exact modular arithmetic up to p < 2^31.
For larger primes, fall back to the Python reference backend.
"""

from __future__ import annotations

from ..field import IRField, primitive_roots_Fp, to_base
from ..ir_nodes import IRRotate, IRPoly, IRMatPow

# v2 fix: proper import, not 'iimport torch'
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


class TorchBackend:
    """GPU-accelerated evaluator using int64 modular arithmetic.

    All operations use torch.int64 tensors with explicit % p
    after every arithmetic op to prevent overflow.
    """

    def __init__(self, ir_field: IRField, device: str = 'cuda'):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch"
            )
        if ir_field.p >= 2**31:
            raise ValueError(
                f"p={ir_field.p} exceeds int64 safe range for modular "
                f"multiply. Use PythonBackend for large primes."
            )

        self.field = ir_field
        self.p = ir_field.p
        self.q = ir_field.q
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Root table as int64 tensor (NOT float — avoid precision loss)
        roots = primitive_roots_Fp(ir_field.p)
        self.root_table = torch.tensor(
            roots, dtype=torch.int64, device=self.device
        )

    def eval_rotate(self, node: IRRotate,
                    reg_tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Multiply register tensor by ω^j in F_p.

        Both operands are int64; result is reduced mod p.
        """
        omega_j = self.root_table[node.j % len(self.root_table)]
        return (reg_tensor * omega_j) % self.p

    def eval_poly(self, node: IRPoly,
                  x_tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Horner evaluation of polynomial over F_q, reduce mod p.

        Each multiply-add step is reduced mod q to prevent int64 overflow.
        Final result is reduced mod p for output field.
        """
        q, p = self.q, self.p
        coeffs = torch.tensor(
            node.coeffs[::-1], dtype=torch.int64, device=self.device
        )

        result = torch.zeros_like(x_tensor)
        for c in coeffs:
            result = (result * x_tensor + c) % q  # explicit mod every step
        return result % p

    def eval_matpow(self, node: IRMatPow,
                    M: 'torch.Tensor') -> 'torch.Tensor':
        """Compute M^d using base-δ decomposition (Lemma 3.12).

        M: (n, n) int64 tensor with entries in [0, p).
        Returns M^d mod p as int64 tensor.
        """
        n = M.shape[0]
        delta = node.delta
        digits = to_base(node.d, delta)

        result = torch.eye(n, dtype=torch.int64, device=M.device)
        Mk = M.clone()

        for alpha in digits:
            if alpha > 0:
                block = self._mat_pow_small(Mk, alpha)
                result = self._mat_mul_mod(result, block)
            Mk = self._mat_pow_small(Mk, delta)

        return result

    def _mat_mul_mod(self, A: 'torch.Tensor',
                     B: 'torch.Tensor') -> 'torch.Tensor':
        """Matrix multiply mod p using int64.

        For small matrices, torch.mm on int64 is exact.
        We reduce mod p after the matmul.
        """
        # torch.mm doesn't support int64 on all devices, so use float64
        # with explicit rounding and mod, or use a manual loop.
        # For correctness, do it element-wise for safety.
        n = A.shape[0]
        C = torch.zeros(n, n, dtype=torch.int64, device=A.device)
        for i in range(n):
            for j in range(n):
                s = torch.tensor(0, dtype=torch.int64, device=A.device)
                for k in range(n):
                    s = s + A[i, k] * B[k, j]
                C[i, j] = s % self.p
        return C

    def _mat_pow_small(self, M: 'torch.Tensor',
                       k: int) -> 'torch.Tensor':
        """M^k mod p via repeated squaring."""
        n = M.shape[0]
        if k == 0:
            return torch.eye(n, dtype=torch.int64, device=M.device)
        if k == 1:
            return M.clone()

        result = torch.eye(n, dtype=torch.int64, device=M.device)
        base = M.clone()

        while k > 0:
            if k & 1:
                result = self._mat_mul_mod(result, base)
            base = self._mat_mul_mod(base, base)
            k >>= 1

        return result
