"""Field arithmetic for Cook-Mertz v4.

Provides:
  - IRField: frozen dataclass tagging every IR node with its working field
  - primitive_roots_Fp(p): enumerate all m-th roots of unity in F_p (m = p-1)
  - to_base(n, b): base-b digit decomposition for Lemma 3.12 dispatch
  - compute_working_field(p, d, n): compute q per Corollary 3.4.2
  - find_nonresidue(p): smallest quadratic non-residue mod p (for F_{p²})
  - F_{p²} arithmetic: add_Fp2, mul_Fp2, conj_Fp2, magsq_Fp2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import sympy


@dataclass(frozen=True)
class IRField:
    """Defines the arithmetic context for a Cook-Mertz computation.

    p: prime characteristic.  Output ring is F_p = Z/pZ.
    q: working field size.    q >= p, must be prime.
       Intermediate computation happens mod q to prevent overflow
       (Corollary 3.4.2: q = O(2^n * p^(d+1))).
       q == p is valid for small-degree cases.
    c: quadratic non-residue mod p (optional).
       When set, registers in this field are elements of
       F_{p²} = F_p[i]/(i² - c).  is_complex returns True.
    """
    p: int
    q: int
    c: Optional[int] = None

    def __post_init__(self):
        if not sympy.isprime(self.p):
            raise ValueError(f"p={self.p} is not prime")
        if not sympy.isprime(self.q):
            raise ValueError(f"q={self.q} is not prime")
        if self.q < self.p:
            raise ValueError(f"q={self.q} must be >= p={self.p}")
        if self.c is not None:
            # Validate c is a quadratic non-residue mod p
            if pow(self.c, (self.p - 1) // 2, self.p) != self.p - 1:
                raise ValueError(
                    f"c={self.c} is not a quadratic non-residue mod p={self.p}"
                )

    @property
    def m(self) -> int:
        """Order of F_p*; m = p - 1.
        This is the root-of-unity order used in the ω-gate."""
        return self.p - 1

    @property
    def is_complex(self) -> bool:
        """True iff this field is F_{p²} (complex extension)."""
        return self.c is not None

    def real_field(self) -> IRField:
        """Return the corresponding real field (c=None, same p and q)."""
        if not self.is_complex:
            return self
        return IRField(p=self.p, q=self.q)

    def validate_exponent(self, j: int) -> int:
        """Normalize rotation exponent to [0, m)."""
        return j % self.m

    def __repr__(self) -> str:
        if self.c is not None:
            return f"IRField(p={self.p}, q={self.q}, c={self.c})"
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


@lru_cache(maxsize=64)
def find_nonresidue(p: int) -> int:
    """Return the smallest quadratic non-residue mod p.

    A quadratic non-residue c satisfies c^((p-1)/2) ≡ -1 (mod p),
    i.e. c has no square root in F_p.  Used to construct F_{p²}.
    """
    if not sympy.isprime(p):
        raise ValueError(f"p={p} is not prime")
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            return c
    raise ValueError(f"No quadratic non-residue found for p={p}")  # unreachable


# ── F_{p²} arithmetic ────────────────────────────────────────────────────────
# Elements are (a, b) representing a + b·i where i² ≡ c (mod p).

Fp2Element = tuple[int, int]


def add_Fp2(x: Fp2Element, y: Fp2Element, p: int) -> Fp2Element:
    """Add two F_{p²} elements: (a+c, b+d) mod p."""
    return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)


def mul_Fp2(x: Fp2Element, y: Fp2Element, p: int, c: int) -> Fp2Element:
    """Multiply two F_{p²} elements using the quotient ring identity.

    (a+bi)(c_+di) = (ac_ + bdc) + (ad + bc_)i  in F_p[i]/(i²-c)
    """
    a, b = x
    c_, d = y
    re = (a * c_ + b * d * c) % p
    im = (a * d + b * c_) % p
    return (re, im)


def conj_Fp2(x: Fp2Element, p: int) -> Fp2Element:
    """Complex conjugate: a+bi → a-bi mod p."""
    return (x[0], (-x[1]) % p)


def magsq_Fp2(x: Fp2Element, p: int, c: int) -> int:
    """Magnitude squared: |a+bi|² = a² + b²·c mod p.

    Note: |a+bi|² = (a+bi)(a-bi) = a² - b²·(i²) = a² - b²·c = a² + b²·(-c)
    But since i² = c (not -c), we have (a+bi)(a-bi) = a² - b²·c.
    Wait — let me be precise. In F_p[i]/(i²-c):
      (a+bi)(a-bi) = a² - b²·i² = a² - b²·c
    So magsq = (a² - b²·c) mod p.
    """
    a, b = x
    return (a * a - b * b * c) % p
