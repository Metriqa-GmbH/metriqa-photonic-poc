"""calc_kan_layer_fprop sums its terms in bfloat16: error grows with fan-in (simulator).

Compares the toolkit against (a) float64 on the bf16-rounded inputs and (b) the same
per-term bf16 roundings as the toolkit (k*x, +phi, cos*a) but an exact (float64) sum.
"""

import numpy as np
from ml_dtypes import bfloat16 as bf
from qant_native_computing_toolkit import ai

rng = np.random.default_rng(0)
n_out, n_ks, batch = 16, 4, 64
ks = np.arange(1, n_ks + 1).astype(bf)


def rel_rms(a, ref):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - ref) ** 2) / np.mean(ref**2)))


print(f"{'n_in':>5} {'toolkit':>9} {'terms bf16, exact sum':>22}   (relative RMS vs float64)")
for n_in in (4, 16, 64, 256):
    x = rng.uniform(-1, 1, (batch, n_in)).astype(bf)
    phi = rng.uniform(-3, 3, (n_out, n_in, n_ks)).astype(bf)
    a = rng.uniform(-1, 1, (n_out, n_in, n_ks)).astype(bf) / bf(n_in)
    got = ai.calc_kan_layer_fprop(x, phi, a, ks)

    xf, pf, af, kf = (np.asarray(v, np.float64) for v in (x, phi, a, ks))
    arg = kf * xf[:, None, :, None] + pf[None]  # (batch, out, in, k)
    ref64 = (np.cos(arg) * af[None]).sum(axis=(2, 3))

    kx = (kf * xf[:, None, :, None]).astype(bf).astype(np.float64)
    u = (kx + pf[None]).astype(bf).astype(np.float64)
    t = (np.cos(u).astype(bf).astype(np.float64) * af[None]).astype(bf).astype(np.float64)
    print(f"{n_in:5d} {rel_rms(got, ref64):9.4f} {rel_rms(t.sum(axis=(2, 3)), ref64):22.4f}")
