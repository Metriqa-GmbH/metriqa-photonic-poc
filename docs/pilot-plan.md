# Proposed Q.ANT NPU Gen 2 pilot plan

A four-week plan for the Q.ANT / IONOS Gen 2 pilot (bare-metal server with root access). It
turns the simulator results of this repository into hardware measurements. The experiments
H1–H5 are specified in the [research report](research-report.md#9-why-physical-qant-npu-access-is-necessary).

## Starting point

- The full harness runs unchanged on hardware: install the toolkit's `default` wheel and the
  driver, then `photonic-poc sdk-info` must report a non-`cpu-backend` driver with `Mul` and
  `NonLin`. Every report records driver string, device list and `simulated: false`.
- All experiments are configured in `configs/research.toml`; trained models are cached, so the
  pilot spends its time on measurement, not training.
- Simulator baselines for every hardware number already exist in `results/research/`.

## Objectives and success criteria

| # | Objective | Measured quantity | Success criterion |
|---|---|---|---|
| O1 | Characterise numerical fidelity of the NPU primitives (H1) | per-op bias and noise σ of `linear_fprop`, `calc_scaled_periodic_nl_fprop`, `calc_kan_layer_fprop`; the measured `tcos` transfer curve | error model with 95% CIs for every primitive; tcos sampled at ≥ 4,096 points per period |
| O2 | Decide whether compact matching models keep their decisions on hardware (H2) | ΔF1, decision flips, calibration (ECE) vs float32 for the champion models, plain and noise-trained | at least one compact model with \|ΔF1\| ≤ 0.005 and flips ≤ 0.5% on 13,500 test pairs; flips explained by the measured noise (margin analysis) |
| O3 | Calibrate the hybrid cost model (H3) | per-call latency, host↔NPU bandwidth, sustained NPU rate per primitive vs size and batch | the fitted model predicts held-out end-to-end timings within 20% |
| O4 | Throughput scaling and the break-even point (H3, H4) | pairs/s vs batch size (1 to 65,536) and number of NPUs/channels; perf-counter utilisation | measured break-even layer width (or proof that none exists below 4,096) for the matrix and Fourier-basis paths |
| O5 | Energy per inference (H4) | J per pair for NPU path vs CPU-only, steady state | repeatable to ±10% over 3 runs; reported with the measurement method |
| O6 | Operational stability (H5) | output drift over 8 h, across driver recalibration cycles, vs temperature sensors | drift bounded and reported; decisions of the champion models stable across the soak |

## Schedule

| Week | Work | Output |
|---|---|---|
| 0 (before access) | freeze configs, cache all models, dry-run on the simulator wheel on the pilot OS image | reproducible baseline |
| 1 | O1: op-level characterisation, repeatability (30 repeats per input), dense tcos sweep | error model per primitive |
| 2 | O2: model-level fidelity; retrain with measured tcos / measured noise if O1 shows a gap (hardware-in-the-loop fine-tuning) | decision-stability report |
| 3 | O3, O4: latency, bandwidth, batch and multi-NPU scaling, cost-model fit | calibrated break-even analysis |
| 4 | O5, O6: energy, 8 h soak; write-up; private aggregate evaluation (only if approved) | updated research report, upstream contributions |

## What we need from Q.ANT

- Pilot access with the Gen 2 driver and the matching toolkit `default` wheel.
- A recommended way to measure NPU power (board sensor, server PDU) and permission to report it.
- Clarification of value-range limits for inputs and weights, if any.
- Permission to publish aggregated results and the updated report; Q.ANT may review before release.

## What Q.ANT gets

- An independent, reproducible characterisation of Gen 2 on a non-vision, non-LLM workload, with
  open code that other pilot users can rerun.
- Upstream contributions: the KAN accumulation fix and the factorised KAN execution (drafts in
  [upstream-issues.md](upstream-issues.md)), plus any defects found on hardware.
- A calibrated cost model telling users which layer sizes benefit from offloading.

## Risks

| Risk | Mitigation |
|---|---|
| Hardware noise far above the simulator: decisions unstable | noise-aware training and hardware-in-the-loop fine-tuning are prepared (O2, week 2) |
| No end-to-end advantage for compact models (expected from Experiment C) | the pilot measures the break-even width directly (O4); a negative result is reported as such |
| Limited access time | everything is scripted and cached; each objective is one command |
| Private data must stay private | private models run only through `photonic-poc private-eval` into directories outside the repository; publication requires `approve-summary` |
