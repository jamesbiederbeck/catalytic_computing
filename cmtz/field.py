"""Field arithmetic for Cook-Mertz v2.

Provides:
  - IRField: frozen dataclass tagging every IR node with its working field
  - primitive_roots_Fp(p): enumerate all m-th roots of unity in F_p (m = p-1)
  - to_base(n, b): base-b digit decomposition for Lemma 3.12 dispatch
  - compute_working_field(p, d, n): compute q per Corollary 3.4.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import sympy


@dataclass(frozen=True)
class IRField:
    """Defines the arithmetic context for a Cook-Mertz computation.

    p: prime characteristic.  Output ring is F_p = Z/pZ.
    q: working field size.    q >= p, must be prime.
       Intermediate computation happens mod q to prevent overflow
       (Corollary 3.4.2: q = O(2^n * p^(d+1))).
       q == p is valid for small-degree cases.
    """
    p: int
    q: int

    def __post_init__(self):
        if not sympy.isprime(self.p):
            raise ValueError(f"p={self.p} is not prime")
        if not sympy.isprime(self.q):
            raise ValueError(f"q={self.q} is not prime")
        if self.q < self.p:
            raise ValueError(f"q={self.q} must be >= p={self.p}")

    @property
    def m(self) -> int:
        """Order of F_p*; m = p - 1.
        This is the root-of-unity order used in the ω-gate."""
        return self.p - 1

    def validate_exponent(self, j: int) -> int:
        """Normalize rotation exponent to [0, m)."""
        return j % self.m

    def __repr__(self) -> str:
        if self.p == self.q:
            return f"IRField(p={self.p})"
        return f"IRField(p={self.p}, q={self.q})"


@lru_cache(maxsize=64)
def primitive_roots_Fp(p: int) -> tuple[int, ...]:
    """Compute {ω^0, ω^1, ..., ω^(m-1)} in F_p where ω is a
    primitive (p-1)-th root of unity (i.e. a generator of F_p*).

    Returns a tuple (immutable, cacheable) of m = p-1 integers.

    This replaces v1's cyclotomic_poly(n, x) which computed the
    minimal polynomial Φ_n(x) — a completely different object.
    """
    if not sympy.isprime(p):
        raise ValueError(f"p={p} is not prime")

    g = int(sympy.primitive_root(p))
    m = p - 1
    return tuple(pow(g, j, p) for j in range(m))


def generator_Fp(p: int) -> int:
    """Return a generator of F_p* (smallest primitive root mod p)."""
    return int(sympy.primitive_root(p))


def to_base(n: int, b: int) -> list[int]:
    """Decompose n into base-b digits, least-significant first.

    Used by IRMatPow for Lemma 3.12 base-δ decomposition:
      M^d = M^(α_0) · (M^δ)^(α_1) · (M^(δ^2))^(α_2) · ...
    where d = Σ α_i · δ^i.
    """
    if b < 2:
        raise ValueError(f"base must be >= 2, got {b}")
    if n == 0:
        return [0]
    digits = []
    while n > 0:
        digits.append(n % b)
        n //= b
    return digits


def compute_working_field(p: int, degree: int, n: int) -> int:
    """Compute the working field size q per Corollary 3.4.2.

    q must be a prime >= 2^n * p^(degree + 1) to guarantee
    that polynomial evaluation over F_q does not alias when
    reduced mod p.

    Returns the smallest prime >= the lower bound.
    """
    lower_bound = (2 ** n) * (p ** (degree + 1))
    return int(sympy.nextprime(lower_bound - 1))


def horner_eval_Fq(coeffs: list[int], x: int, q: int) -> int:
    """Evaluate polynomial at x using Horner's method, all ops mod q.

    coeffs[i] is the coefficient of x^i (ascending order).
    """
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % q
    return result
