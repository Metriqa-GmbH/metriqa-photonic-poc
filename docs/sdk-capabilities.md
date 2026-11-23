# Q.ANT Native Computing Toolkit: what it actually supports

Findings from reading the toolkit source and running its `cpu-backend` simulator. Everything
here refers to **toolkit v2.5.0** (released 2026-11-17, repository commit `09d8b70`) and the
examples repository at commit `e0868cf`. Statements marked *source* come from reading the code;
statements marked *verified* were also executed against the simulator in this repository.

## Distribution

- Official API for Q.ANT NPUs: a Rust core with Python (PyO3) and C bindings, Apache-2.0.
- Not on PyPI. GitHub releases ship one abi3 wheel per OS and variant (Python 3.11 to 3.14):
  - `default`: needs the Q.ANT driver and hardware; Linux x86_64 only.
  - `cpu-backend`: standalone simulator; Linux x86_64, Windows x86_64, macOS arm64.
- [`scripts/install_qant_sdk.py`](../scripts/install_qant_sdk.py) installs the `cpu-backend`
  wheel with a pinned SHA-256.
- Python dependencies are only `numpy` and `ml-dtypes`. PyTorch is not required by the toolkit.

## Data types

All AI and native operations take and return **bfloat16** NumPy arrays (`ml_dtypes.bfloat16`).
The DFT uses `bcomplex32` (two bf16 values). Passing float32 raises a `TypeError` (*verified*).
There is no float32 or integer path.

## Devices (simulator)

The `cpu-backend` exposes six simulated NPUs (*verified* with `photonic-poc sdk-info`):

| IDs | Generation | Capabilities |
|---|---|---|
| 0, 1, 2 | Gen2 | `Mul` (element-wise and matrix-vector), `NonLin` (periodic nonlinearity), `MemHandling` |
| 10, 11, 12 | Gen2p | `Dft` only |

`device_id=-1` (the default everywhere) selects the first NPU that supports the operation.
Asking a device for an operation it lacks fails, e.g. `NPU10 does not support MULTIPLICATION`
(*verified*). Code written against capabilities rather than fixed IDs therefore carries over to
future device generations.

## Operations and where they run

"NPU" means the toolkit calls the driver (on the simulator: the CPU backend). "Host" means the
toolkit computes it itself in Rust, in bf16, **even when hardware is attached** (*source*:
those functions only validate the device ID and never call the driver; the Rust doc comments say
"not dispatched to any NPU", the Python docstrings do not). A maintainer confirmed it publicly
(toolkit discussion #11): "only the matrix multiplications run on the card", and "the PCIe bus is
currently the bottleneck".

| Python call | Runs on | Notes |
|---|---|---|
| `ai.linear_fprop`, `native.linear_fprop` | NPU (`Mul`) | `features @ weights.T`; batch dims allowed |
| `ai.linear_fprop_preloaded_weights`, `generic.upload/allocate/download`, `linear_fprop_preloaded` | NPU (`Mul`, `MemHandling`) | weights stay in device memory |
| `native.mul_elementwise` | NPU (`Mul`) | |
| `native.calc_scaled_periodic_nl_fprop` | NPU (`NonLin`) | `tcos(u) * v`; on hardware `tcos` is a cosine-like optical response |
| `ai.calc_kan_layer_fprop` | hybrid | `k*x` on NPU (`Mul`), `+phi` on host, `tcos * a` on NPU (`NonLin`), sum over inputs and frequencies on host **in bf16** |
| `ai.conv_fprop`, `ai.conv_transpose_fprop` (+ preloaded variants) | hybrid | im2row on host, matrix product on NPU; `dilation != 1` and (transpose) `output_padding != 0` raise "not implemented yet" although the Python API accepts and documents them (*verified*) |
| `ai.batchnorm2d_fprop` | hybrid | normalisation on host, scale via NPU `mul` |
| `native.dft` | NPU (`Dft`, Gen2p) | unnormalised, last axis |
| `ai.add_bias_fprop` | host | |
| `ai.relu_fprop`, `ai.sigmoid_fprop`, `ai.softmax_fprop` | host | |
| `ai.maxpool2d_fprop`, `ai.avgpool2d_fprop`, adaptive variants | host | adaptive pooling needs input size divisible by output size |
| `info.get_capabilities`, `generic.get_available_npus`, `info.get_perf_counter`, `info.get_sensor_info` | n/a | perf counters are always zero and sensor values are constants on the simulator |

All AI operations are forward passes (`*_fprop`). There is no backward pass, so training happens
in PyTorch (or similar) and only inference runs through the toolkit. This is also how Q.ANT's own
examples work (`QLinear`/`QFourier` layers switch to the toolkit in `eval()` mode).

Not available as toolkit operations: tanh, GELU, layer normalisation, attention as a fused op,
embedding lookup, integer or float32 arithmetic.

## What the simulator does and does not model

The CPU backend describes itself as "a lightweight simulation backend ... intentionally minimal
and not fully featured" (*source*: `src/qant_driver_import/gen2/backend_cpu.rs`). Concretely:

- **Noise-free.** No analog noise, drift, calibration or encoding limits are modelled. The device
  config comments state that simulated chips are noise-free.
- **bf16 arithmetic.** Matrix-vector products round each element-wise product to bf16, accumulate
  in float32 and round the output to bf16 (*source*). Our op study measures 0.235% relative RMS
  error independent of fan-in, against 0.166% for PyTorch bf16 (which does not round the
  products). See [results](../results/RESULTS.md#operation-level-numerics-of-the-toolkit).
- **Exact cosine.** The periodic nonlinearity is `cos` in bf16 (*source* and *verified*: 0.14%
  relative RMS against float64 cosine). Hardware uses its own `tcos`; the test of Q.ANT's intro
  notebook allows a mean error of 0.2 for the KAN layer, commented as comparing "the almost
  cosine against the exact cosine". Models whose accuracy depends on the exact shape of the
  cosine cannot be validated on the simulator.
- **No timing.** Simulator wall-clock time measures CPU emulation and FFI overhead only.

## Finding: the KAN layer accumulates in bf16

`calc_kan_layer` sums its `n_in * n_ks` terms with a bf16 accumulator on the host (*source*:
`src/non_linear.rs`, `let mut sum = Value::from_f32(0.)` with `Value = bf16`). The error
therefore grows with fan-in (*verified*, 16 outputs, grid 4):

| fan-in | toolkit | same bf16 terms, exact sum | PyTorch bf16 |
|---|---|---|---|
| 4 | 0.65% | 0.36% | 0.42% |
| 64 | 1.72% | 0.37% | 0.44% |
| 256 | 3.54% | 0.41% | 0.46% |

Summing the identical bf16 terms exactly removes the growth, so the accumulator is the cause.
Because this summation is host code in the toolkit, it applies to hardware runs too. Accumulating
in float32 (as the matrix-vector path already does) would remove it. In a stacked Fourier
network the error is amplified downstream: the next layer multiplies its input by frequencies up
to `k = grid_size`, so a hidden-layer error `e` shifts the cosine argument by up to `k * e`.

## Workaround: execute KAN layers as a Fourier basis plus one matrix product

`a·cos(kx + φ) = (a cos φ)·cos(kx) − (a sin φ)·sin(kx)` turns a KAN layer into 2·n_in·n_k basis
evaluations (`calc_scaled_periodic_nl_fprop`, independent of the output width) followed by one
preloaded `linear_fprop` with float32 accumulation. `HybridOptions(kan_impl="factorized")` implements
it; on trained models it is more accurate than PyTorch bf16 and puts 95–97% of the layer's operations
on the NPU (research report, §5–6). The identity assumes an exact cosine; with a hardware `tcos` the
two forms differ and a model should be trained against the form that is executed.

## Public hardware information (vendor claims)

Gen 2: thin-film lithium niobate photonic processor on a PCIe card in a rack server ("Native
Processing Server"); quoted at 8 GOPS (operations "more complex than multiply-accumulates"), about
150 W and 8 parallel channels; "up to 30x lower energy use and 50x higher performance" for selected
workloads (EE Times 2025-11-26, Quantum Insider 2025-11-19). Gen 2+ adds optical DFT (shown at SC26).
No public paper reports precision or noise. Q.ANT and IONOS run a free Gen 2 pilot on bare-metal
servers for partners with a concrete use case.

## Hardware-only questions

These cannot be answered with the simulator and are left open:

- Accuracy of `tcos` and of analog matrix products; noise level and its distribution.
- Value-range limits for optical encoding (inputs, weights, outputs), if any, and whether the
  driver rescales internally.
- Throughput, latency and energy per operation; the cost of host round-trips between NPU and
  host operations (bias, ReLU, sigmoid run on the host).
- Behaviour of `get_perf_counter` and sensor readings.
