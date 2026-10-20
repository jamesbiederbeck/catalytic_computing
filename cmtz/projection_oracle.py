"""Reference model of the Cook-Mertz roots-of-unity projection over F_p.

Plain integer arithmetic only: nothing here imports the compiler, the ISA
simulator or `field.py`, so the cmtz-core programs in `isa/projection.py`
are checked against an independent definition.

The identity. Let ω have multiplicative order m in F_p and let P be a
polynomial in the variables x_i of total degree d < m. For any shifts τ_i,
P(x + z·τ) is a polynomial in z of degree <= d, and the sum of ω^(jk) over
j in [0, m) is m when m divides k and 0 otherwise. So

    Σ_{j=0}^{m-1} P(x + ω^j·τ) = m · P(x)                      (exact)

whatever τ is. Over F_17 with ω = 3 (order 16), m = 16 ≡ -1, so
`y - Σ_j P(x + ω^j τ)` is `y + P(x)`: subtracting the sum adds P once.

Masked gate. A two-input Boolean gate g has the multilinear extension
ĝ(u, v) = k0 + k1·u + k2·v + k3·u·v (degree 2), equal to g on {0, 1}^2.
Its catalytic evaluation keeps the masks in two registers τ1, τ2 that start
with arbitrary values and adds g(a, b) to an output register y:

    repeat m times:
        τ1 += a; τ2 += b                (registers hold a + ω^j τ1, ...)
        y  -= ĝ(τ1, τ2)                 (accumulate)
        τ1 -= a; τ2 -= b                (uncompute)
        τ1 *= ω; τ2 *= ω                (rotate; ω^m = 1 restores them)

Gate tree. f(g1(a, b), g2(c, d)) runs the same loop one level up: the
parent's registers τl, τr receive g1 and g2 by a full child loop (the child
adds into τl), the parent accumulates f̂, and the same child loop run with
the opposite sign removes g1 and g2 again. The children share one pair of
mask registers. Nothing intermediate is ever held outside the registers
that are restored.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

P, OMEGA = 17, 3

# truth table bit (2a + b) holds g(a, b)
GATES: dict[str, int] = {
    "FALSE": 0b0000, "NOR": 0b0001, "B_AND_NOT_A": 0b0010, "NOT_A": 0b0011,
    "A_AND_NOT_B": 0b0100, "NOT_B": 0b0101, "XOR": 0b0110, "NAND": 0b0111,
    "AND": 0b1000, "XNOR": 0b1001, "B": 0b1010, "A_IMPLIES_B": 0b1011,
    "A": 0b1100, "B_IMPLIES_A": 0b1101, "OR": 0b1110, "TRUE": 0b1111,
}


def gate(tt: int, a: int, b: int) -> int:
    return (tt >> (2 * a + b)) & 1


def order(w: int, p: int = P) -> int:
    """Multiplicative order of w mod p (w != 0)."""
    k, x = 1, w % p
    while x != 1:
        x, k = x * w % p, k + 1
    return k


@lru_cache(maxsize=None)
def coeffs(tt: int, p: int = P) -> tuple[int, int, int, int]:
    """(k0, k1, k2, k3) of the multilinear extension, mod p."""
    g00, g01, g10, g11 = (gate(tt, a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    return (g00 % p, (g10 - g00) % p, (g01 - g00) % p,
            (g11 - g10 - g01 + g00) % p)


def extension(tt: int, u: int, v: int, p: int = P) -> int:
    k0, k1, k2, k3 = coeffs(tt, p)
    return (k0 + k1 * u + k2 * v + k3 * u * v) % p


def projection_sum(poly: Callable[[int, int], int], x: int, y: int,
                   t1: int, t2: int, p: int = P, omega: int = OMEGA) -> int:
    """Σ_j poly(x + ω^j t1, y + ω^j t2) mod p, over j < order(ω)."""
    s, w = 0, 1
    for _ in range(order(omega, p)):
        s += poly((x + w * t1) % p, (y + w * t2) % p)
        w = w * omega % p
    return s % p


def masked_gate(tt: int, a: int, b: int, t1: int, t2: int, y: int,
                p: int = P, omega: int = OMEGA) -> tuple[int, int, int]:
    """Run the masked-gate procedure; returns (y, τ1, τ2) afterwards."""
    for _ in range(order(omega, p)):
        t1, t2 = (t1 + a) % p, (t2 + b) % p
        y = (y - extension(tt, t1, t2, p)) % p
        t1, t2 = (t1 - a) % p, (t2 - b) % p
        t1, t2 = t1 * omega % p, t2 * omega % p
    return y, t1, t2


def gate_tree(f: int, g1: int, g2: int, a: int, b: int, c: int, d: int,
              regs: dict[str, int], p: int = P, omega: int = OMEGA
              ) -> dict[str, int]:
    """Clean evaluation of f(g1(a, b), g2(c, d)).

    `regs` holds the starting values of the mask registers t1, t2 (shared
    by both children), tl, tr (the parent's) and the output y; returns their
    values afterwards.
    """
    r = dict(regs)
    m = order(omega, p)

    def child(tt: int, x: int, z: int, acc: str, sign: int) -> None:
        # sign -1 negates the accumulator around the loop: -(-acc + g) = acc - g
        if sign < 0:
            r[acc] = -r[acc] % p
        y, r["t1"], r["t2"] = masked_gate(tt, x, z, r["t1"], r["t2"], r[acc], p, omega)
        r[acc] = y if sign > 0 else -y % p

    for _ in range(m):
        child(g1, a, b, "tl", +1)
        child(g2, c, d, "tr", +1)
        r["y"] = (r["y"] - extension(f, r["tl"], r["tr"], p)) % p
        child(g1, a, b, "tl", -1)
        child(g2, c, d, "tr", -1)
        r["tl"], r["tr"] = r["tl"] * omega % p, r["tr"] * omega % p
    return r
