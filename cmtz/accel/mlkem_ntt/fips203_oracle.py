"""Independent FIPS 203 (ML-KEM) NTT oracle.

A direct transliteration of Algorithms 9 (NTT) and 10 (NTT^-1) with plain ``%``
arithmetic. It deliberately imports nothing from the rest of the package and shares
no reduction or twiddle code with the hardware generator, so a mistake made in both
places cannot hide.  ``direct_ntt`` evaluates the transform from its defining
equation (FIPS 203 section 4.3) as a third, structurally different route.
"""

Q = 3329
N = 256
ZETA = 17          # primitive 256-th root of unity mod q (FIPS 203 section 4.3)


def bitrev7(i):
    return int(format(i, "07b")[::-1], 2)


ZETAS = [pow(ZETA, bitrev7(k), Q) for k in range(128)]       # zeta^BitRev7(k)


def ntt(f):
    """FIPS 203 Algorithm 9. Input and output are 256 coefficients in [0, q)."""
    assert len(f) == N and all(0 <= c < Q for c in f)
    f = list(f)
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % Q
                f[j + length] = (f[j] - t) % Q
                f[j] = (f[j] + t) % Q
        length //= 2
    return f


def ntt_inv(f):
    """FIPS 203 Algorithm 10, including the final multiplication by 128^-1 = 3303."""
    assert len(f) == N and all(0 <= c < Q for c in f)
    f = list(f)
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
        length *= 2
    return [(c * 3303) % Q for c in f]


def direct_ntt(f):
    """The defining equation: f^[2i] = sum_j f[2j] zeta^((2 br7(i)+1) j), and the same
    with f[2j+1] for f^[2i+1]. O(n^2); used only to cross-check ``ntt``."""
    out = [0] * N
    for i in range(128):
        w = pow(ZETA, 2 * bitrev7(i) + 1, Q)
        out[2 * i] = sum(f[2 * j] * pow(w, j, Q) for j in range(128)) % Q
        out[2 * i + 1] = sum(f[2 * j + 1] * pow(w, j, Q) for j in range(128)) % Q
    return out
