# Running on physical Q.ANT hardware

Nothing in this repository has been run on a Q.ANT NPU yet. This page lists what changes for a
hardware run and which experiments need one.

## Setup

1. Install the Q.ANT driver (Linux x86_64) as documented by Q.ANT.
2. Install the toolkit's **default** wheel (not `cpu-backend`) from the same release, e.g.
   `uv pip install qant_native_computing_toolkit-2.5.0-cp311-abi3-linux_x86_64.whl`.
3. Check the device: `uv run photonic-poc sdk-info`. `qant_driver` must no longer read
   `cpu-backend`, and at least one NPU must list `Mul` and `NonLin`.

No code changes are needed. `QantBackend` uses `device_id=-1` (first NPU that supports the
operation), preloads weights only when an NPU reports `MemHandling`, and records the driver string
and device list in every report. Reports flag `"simulated": true` when the driver is the CPU
backend, so simulator and hardware results cannot be confused.

## Experiments that need hardware

| Question | How to run it | Why the simulator cannot answer |
|---|---|---|
| Matching quality on the NPU | `photonic-poc evaluate --backends torch-fp32 qant` | real `tcos` shape, analog noise and encoding limits are not simulated |
| Agreement with float32 (logit error, decision flips) | same; see `agreement_vs_fp32` | the simulator is a noise-free bf16 emulation |
| Does noise-aware training help? | compare `mlp` vs `mlp-noise0.05` and `fourier` vs `fourier-noise0.05` on `qant` | our noise model is an assumption |
| Operation-level error vs fan-in | `photonic-poc op-numerics` | analog accumulation error is unknown |
| Latency and throughput | `photonic-poc evaluate` latency columns, plus `info.get_perf_counter` | simulator timings measure CPU emulation |
| Energy per inference | external power measurement; not implemented here | no model of power exists in the toolkit |

## Things to watch

- **Host round-trips.** Bias, ReLU and sigmoid run on the host in the toolkit. An MLP therefore
  alternates NPU and host work per layer. The Fourier network keeps more of its computation on
  the NPU (products and cosines), but its sums also run on the host. Measure end-to-end latency,
  not only matrix products.
- **Value ranges.** Our features are in `[0, 1]`; trained weights and logits are small. If the
  hardware imposes encoding ranges, check the `describe()` output of the backend and the driver
  documentation before comparing numbers.
- **The KAN bf16 accumulator** (see [sdk-capabilities.md](sdk-capabilities.md)) affects hardware
  runs as well, because the summation happens in toolkit host code.
- **Noise budget.** The [seed study and noise sweep](../results/RESULTS.md) show how much
  additive noise each model tolerates before F1 drops. Once a hardware noise figure is known, it
  can be compared against those tables directly.
