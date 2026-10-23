"""Twiddle tables, built independently of ``fips203_oracle`` (iterative powers, own bit
reversal) and encoded for the datapath's reduction scheme.  A test compares them with
the oracle's ``pow``-based table."""
from .arith import Q, twiddle_encode


def _powers_of_17():
    p, out = 1, []
    for _ in range(256):
        out.append(p)
        p = (p * 17) % Q
    return out


def _brv7(k):
    r = 0
    for _ in range(7):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


_POW = _powers_of_17()
ZETA = [_POW[_brv7(k)] for k in range(128)]


def table(reduction, inverse):
    """z values for k = 0..127 as the butterfly lane expects them."""
    return [twiddle_encode(reduction, ZETA[k], inverse) for k in range(128)]
