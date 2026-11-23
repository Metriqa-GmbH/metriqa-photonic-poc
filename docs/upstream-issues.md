# Draft upstream issues for the Q.ANT Native Computing Toolkit

**Status: drafts, not filed.** Nothing here has been posted, and nobody has been contacted.
As of 2026-11-23 none of these duplicates an existing item: toolkit issues #3, #4, #6, #8,
#13, #14 and discussions #5, #7, #9, #10, #11, #12, #15, #16 cover other topics. Discussion
#11 mentions host execution only informally (a maintainer wrote that "only the matrix
multiplications run on the card").

Common to all items:

- Toolkit: v2.5.0, commit `09d8b70`
  (https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit), Apache-2.0.
- Environment: `cpu-backend` simulator wheel, Windows x86_64, Python 3.12, numpy and ml_dtypes.
  **Hardware was not tested.** Statements about hardware below are limited to what the source code
  does on the host, before or after driver calls.
- Source paths are relative to the toolkit repository root.
- Scripts live in `examples/upstream/` of this repository and need only numpy, ml_dtypes and
  the toolkit.

---

## 1. `ai.calc_kan_layer_fprop` accumulates its sum in bfloat16

**Type:** defect (numerical accuracy)

### Reproduction

`examples/upstream/kan_bf16_accumulation.py` (fixed seed). It compares the toolkit with
(a) float64 evaluated on the bf16-rounded inputs, and (b) an emulation that rounds each term to
bf16 the way the toolkit does (bf16 after `k*x`, after `+phi`, after `cos * a`) but sums exactly.
Setup: 16 outputs, 4 frequencies, batch 64.

```
 n_in   toolkit  terms bf16, exact sum   (relative RMS vs float64)
    4    0.0074                 0.0052
   16    0.0105                 0.0053
   64    0.0187                 0.0052
  256    0.0378                 0.0048
```

### Expected vs actual

Expected: error roughly independent of fan-in, as in column (b) (about 0.5%, the cost of bf16
rounding of each term). Actual: the toolkit's relative RMS error grows from 0.7% at 4 inputs to
3.5-3.8% at 256 inputs (two seeds tried), up to about 8x worse than summing the same bf16 terms
exactly.

### Root cause

`src/non_linear.rs`, function `calc_kan_layer`, lines 230-236: the accumulator is
`let mut sum = Value::from_f32(0.);` and `sum += chunk_ikb[flat_idx];`, where `Value` is
`half::bf16` (`src/global_settings.rs:3`). Each of the `n_in * n_ks` additions rounds to bf16
(8 significant bits), so the rounding error of a long sum grows with the number of terms. This
loop runs on the host inside the toolkit, so it is not specific to the simulator and is also
executed when hardware is attached (hardware accuracy was not tested).

### Suggested fix

Accumulate in f32 and round once, as the matrix-vector path already does
(`src/qant_driver_import/gen2/backend_cpu.rs:101`, `:105`, `:108` and `:248`, `:251`, `:257`):

```rust
let mut sum = 0.0f32;                  // line 230, was: Value::from_f32(0.)
sum += chunk_ikb[flat_idx].to_f32();   // line 234, was: sum += chunk_ikb[flat_idx];
Value::from_f32(sum)                   // line 237, return value of the closure: round once
```

### Impact

Layers with large fan-in lose accuracy that is unrelated to the device's precision, which makes it
hard to separate device error from host summation error. The cost of the fix is negligible.

---

## 2. `dilation` is documented for convolutions but `dilation != 1` always fails

**Type:** defect / documentation gap

### Reproduction

`examples/upstream/conv_dilation.py`:

```
conv_fprop dilation=1: ok, output shape (1, 1, 6, 6)
conv_fprop dilation=2: RuntimeError: Rust core error: Toolkit Error: dilation != 1 not implemented yet
conv_transpose_fprop dilation=1: ok, output shape (1, 1, 10, 10)
conv_transpose_fprop dilation=2: RuntimeError: Rust core error: Toolkit Error: dilation != 1 not implemented yet
```

### Expected vs actual

The Python docstrings of `conv_fprop`, `conv_fprop_preloaded_weights`, `conv_transpose_fprop` and
`conv_transpose_fprop_preloaded_weights` (`python/qant_native_computing_toolkit/_wrapper.py`,
lines 151, 208, 253, 313) describe `dilation` as an ordinary parameter ("Dilation ("zoom out") of
the kernel"), and it is a required positional argument (also in the C header
`src/c_bridge/qant_native_computing_toolkit_ai.h`, lines 101 and 122). The Python wrapper only
checks that it is a non-negative integer (lines 178, 239, 282, 346). Any other value than 1 fails
late, from the Rust core, with a generic `RuntimeError`.

### Root cause

`src/conv.rs`: `conv_fprop_preloaded_weights` rejects it at lines 112-114 and
`conv_transpose_fprop_preloaded_weights` at lines 220-222 (the latter also rejects
`output_padding != 0` just below). `im2row_transpose` hardcodes `let dilation = 1;` (line 411).
`conv_transpose_fprop` behaves like `conv_fprop`: dilation 2 is rejected (run above), so the
limitation applies to both and to their preloaded variants.

### Suggested fix

Either state "currently only 1 is supported" in the docstrings and the C header, or validate early
in the Python wrapper with a clear error, e.g.
`if dilation != 1: raise ValueError("dilation != 1 is not implemented yet")`.

### Impact

Users porting dilated convolutions (e.g. from PyTorch) only find out at call time, and the message
does not name the operation or the argument.

---

## 3. Docs do not say which operations run on the NPU and which on the host

**Type:** documentation

### Observation (source evidence, no script)

The Python docstrings of the `ai` and `native` functions describe `device_id` as "The identifier of
the NPU on which the operation is performed", including for functions that never call the driver
(e.g. `relu_fprop`, `_wrapper.py:474-489`). The Rust sources say it correctly ("This operation is
not dispatched to any NPU", `src/non_linear.rs:19,41,58`, `src/bias.rs:10,41`,
`src/pooling.rs:36,157,212,238,265,295`), but that text does not reach the Python docs. The Sphinx
index (`doc/python/source/index.rst:35`) says `native` is "directly on the NPU with no data
processing on the host", but gives no such statement for `ai`.

Execution per function, from reading the source:

| Function | Executes on | Evidence |
|---|---|---|
| `relu_fprop`, `sigmoid_fprop`, `softmax_fprop` | host, bf16 | `src/non_linear.rs:26,48,65`: only `check_npu_id` |
| `add_bias_fprop`, `add_bias_for_conv2d_fprop` | host, bf16 | `src/bias.rs:18,49` |
| `maxpool2d`, `avgpool2d`, adaptive variants | host, bf16 | `src/pooling.rs:58,176` |
| `linear_fprop`, `linear_fprop_preloaded*` | driver (matrix-vector) | `src/linear.rs:54` |
| `mul_elementwise` | driver | `src/elementwise.rs:19` |
| `calc_scaled_periodic_nl_fprop` | driver | `src/non_linear.rs:98` |
| `dft` | driver (Gen2p) | `src/dft.rs:22` |
| `calc_kan_layer_fprop` | hybrid: `k*x` and `tcos*a` on driver; `+phi` and the sum on host | `src/non_linear.rs:198,214-221,230-236` |
| `conv_fprop`, `conv_transpose_fprop` | hybrid: patch extraction and layout change on host, matrix product on driver | `src/conv.rs:289-316` |
| `batchnorm2d_fprop` | hybrid: normalisation and bias on host, scale multiply on driver | `src/norm.rs:66-90` |

### Suggested fix

Add one line to each docstring, e.g. "Executes on: host (bf16); `device_id` is only validated.",
"Executes on: NPU (matrix-vector product).", or "Executes on: hybrid (matrix product on NPU, rest
on host).", reword the `device_id` text for host-only functions, and add a table like the one
above to the docs.

### Impact

Users sizing an experiment, or reading hardware measurements, cannot tell from the public docs
which parts of a network exercise the device. (Hardware behaviour was not tested; this concerns
only what the toolkit code calls.)

---

## 4. Feature request: device-resident chaining for non-linear operations

**Type:** feature request (not a defect)

### Observation

Only the matrix product has a memory-handle path: `linear_fprop_preloaded` takes features and
result handles (`_wrapper.py:793`, `src/memory_management.rs:92`). `mul_elementwise`,
`calc_scaled_periodic_nl_fprop` and `calc_kan_layer_fprop` take and return host arrays. A network
that alternates matrix products and the native non-linearity therefore needs an upload and a
download around every layer.

`calc_kan_layer` additionally sends an expanded term tensor (`src/non_linear.rs:198-221`). For
`n_in` inputs, `n_out` outputs, `n_ks` frequencies and `N = n_out * n_in * n_ks`, elements per
sample:

| Step | To device | From device |
|---|---|---|
| `k*x` (`mul`) | 2 * n_in * n_ks | n_in * n_ks |
| `tcos(arg) * a` (`scaled_periodic_nl`) | 2 * N (argument and amplitudes) | N |
| Total | 2 * N + 2 * n_in * n_ks | N + n_in * n_ks |

For n_in = 256, n_out = 16, n_ks = 4: N = 16384, so 34,816 elements go to the device and 17,408
come back per sample, versus 256 in and 16 out as the layer's logical input and output. The
amplitudes are identical for every sample but are resent for each.

### Suggested change

Memory-handle variants of `mul_elementwise` and `calc_scaled_periodic_nl_fprop` (device in, device
out), and/or a fused "linear then non-linearity" call, and/or a KAN call that keeps the amplitudes
resident and expands `k*x + phi` on the device side. Even the first alone would let users chain
layers without host round trips.

### Impact

For multi-layer networks the host-device traffic could dominate measured throughput. This is an
inference from the data volume above; transfer cost was not measured.

---

## 5. Suggestion: execute `calc_kan_layer_fprop` as a Fourier basis plus one matrix product

**Type:** feature request / implementation suggestion (addresses items 1 and 4 together)

### Idea

The KAN layer's formula factorises exactly:

```
a · cos(k x + φ) = (a cos φ) · cos(k x) − (a sin φ) · sin(k x),      sin(u) = cos(u − π/2)
```

So `y_j = b_j + Σ_{i,l} a_jil cos(k_l x_i + φ_jil)` equals a fixed basis
`[cos(k_l x_i), sin(k_l x_i)]` (2·n_in·n_ks values per sample, **independent of n_out**) followed by
one matrix product with weights `α = a cos φ`, `β = −a sin φ` and the bias as an extra column. Both
steps are NPU primitives: `calc_scaled_periodic_nl_fprop` for the basis (with `v = 1`) and a
preloaded `linear_fprop` for the mix, which accumulates in float32. The weight matrix can be computed
once at upload time from `phis` and `ampls`.

### Measurements (cpu-backend simulator, this repository)

`photonic_poc.kan_paths` implements this as `HybridOptions(kan_impl="factorized")`; the unit test
`tests/test_research_units.py::test_factorized_kan_identity_is_exact` checks the identity in float64.
On randomly initialised layers (1,000 inputs, relative RMS deviation from float32):

| model | toolkit `calc_kan_layer_fprop` | same math, float32 sum | factorised | PyTorch bf16 |
|---|---|---|---|---|
| kan-4-g4 (19→4→1) | 2.84% | 1.55% | 1.28% | 1.68% |
| kan-8-g8 (19→8→1) | 8.05% | 3.91% | 3.46% | 4.24% |

Static cost per sample (see `photonic_poc.op_profile`):

| model | path | NPU share of ops | nonlinear evaluations | elements to/from device |
|---|---|---|---|---|
| kan-8-g8 | toolkit | 52% | 1,280 | 4,488 |
| kan-8-g8 | factorised | 97% | 432 | 2,387 |

The factorised path performs more multiply-accumulates (the mix matrix), so total operations rise by
about 20%; all of them are NPU operations.

### Caveat

The identity requires the periodic function to be an exact cosine. With a hardware `tcos` that
deviates from a cosine the two forms are no longer identical; a model trained against the factorised
form (or against measured `tcos`) would still be valid. This has not been tested on hardware.
