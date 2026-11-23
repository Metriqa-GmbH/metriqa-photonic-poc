# Running on physical Q.ANT hardware

Nothing in this repository has been run on a Q.ANT NPU yet. This page lists what changes for a
hardware run and which experiments need one.

## Setup

1. Install the Q.ANT driver (Linux x86_64) as documented by Q.ANT.
2. Install the toolkit's **default** wheel (not `cpu-backend`) from the same release, e.g.
   `uv pip install qant_native_computing_toolkit-2.5.0-cp311-abi3-linux_x86_64.whl`.
3. Check the device: `uv run photonic-poc sdk-info`. `qant_driver` must no longer read
   `cpu-backend`, and at least one NPU must list `Mul` and `NonLin`.

No code changes are needed. `QantBackend` and `HybridExecutor` use `device_id=-1` (the first NPU
that supports the operation) and preload weights only when an NPU reports `MemHandling`; every
report records the driver string and device list. Reports flag `"simulated": true` when the driver is the CPU
backend, so simulator and hardware results cannot be confused.

## Experiments that need hardware

The research report specifies five experiments with hypotheses, measurements and success criteria
([§9](research-report.md#9-why-physical-qant-npu-access-is-necessary)); the
[pilot plan](pilot-plan.md) schedules them. In short:

| | Experiment | Command |
|---|---|---|
| H1 | error model of the NPU primitives, measured `tcos` | `photonic-poc op-numerics` (+ repeated calls) |
| H2 | decision stability of trained models | `photonic-poc research --only robustness` |
| H3 | latency, transfer, batch and multi-NPU scaling | `photonic-poc research --only hybrid` + `info.get_perf_counter` |
| H4 | energy per inference | external power measurement during H3 runs |
| H5 | 8-hour stability across recalibration | repeated `robustness` runs + `info.get_sensor_info` |

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
