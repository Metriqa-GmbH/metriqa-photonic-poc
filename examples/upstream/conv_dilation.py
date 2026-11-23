"""conv_fprop / conv_transpose_fprop document `dilation` but reject dilation != 1 (simulator)."""

import numpy as np
from ml_dtypes import bfloat16 as bf
from qant_native_computing_toolkit import ai

rng = np.random.default_rng(0)
x = rng.normal(size=(1, 1, 8, 8)).astype(bf)
w = rng.normal(size=(1, 1, 3, 3)).astype(bf)  # conv weights: (out, in, h, w)
wt = w  # conv_transpose weights: (in, out, h, w); same shape here

calls = {
    "conv_fprop": lambda d: ai.conv_fprop(x, w, 0, 1, d),
    "conv_transpose_fprop": lambda d: ai.conv_transpose_fprop(x, wt, 0, 1, d, 0),
}
for name, call in calls.items():
    for d in (1, 2):
        try:
            print(f"{name} dilation={d}: ok, output shape {call(d).shape}")
        except Exception as e:
            print(f"{name} dilation={d}: {type(e).__name__}: {e}")
