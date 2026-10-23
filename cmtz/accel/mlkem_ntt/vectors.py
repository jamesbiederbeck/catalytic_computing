"""Test polynomials shared by the Python and RTL tests."""
import random


def test_vectors(n_random=6, seed=2026):
    rng = random.Random(seed)
    v = {
        "zeros": [0] * 256,
        "max": [3328] * 256,
        "ones": [1] * 256,
        "ramp": [i * 13 % 3329 for i in range(256)],
        "alternating": [3328 if i % 2 else 0 for i in range(256)],
        "impulse0": [1] + [0] * 255,
        "impulse255": [0] * 255 + [1],
        "impulse_mid_max": [0] * 100 + [3328] + [0] * 155,
        "step": [3328 if i >= 128 else 0 for i in range(256)],
        "q_minus_ramp": [3328 - (i * 7 % 3329) for i in range(256)],
    }
    for n in range(n_random):
        v[f"rand{n}"] = [rng.randrange(3329) for _ in range(256)]
    return v
