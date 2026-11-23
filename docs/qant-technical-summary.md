# Technical summary for Q.ANT

**From:** Metriqa GmbH · **Re:** NPU Gen 2 pilot · **Toolkit:** v2.5.0 (`cpu-backend` simulator)
**Full report:** [research-report.md](research-report.md) · **Code:** this repository (Apache-2.0)

## What we did

We evaluated photonic-ready inference for entity resolution (deciding whether two company records
describe the same company) with the Native Computing Toolkit: 352 compact models (logistic
regression, ReLU MLPs, cosine MLPs, Fourier-KAN), 5 seeds each, 13,500 synthetic test pairs, every
model executed through the toolkit simulator and compared with PyTorch float32 and bfloat16. All data
is synthetic; no hardware was used, and no performance or energy claim is made.

## Findings relevant to Q.ANT

1. **Decisions survive bf16 execution.** The simulator flips 0.03–0.06% of MLP decisions and ~0.15%
   of KAN decisions; calibration and precision at 99–99.9% precision thresholds are unchanged.
   Flips are confined to a margin band of |logit| ≲ 0.1 (≈ 0.5 for KAN), and margin alone explains
   which pair types flip.
2. **`calc_kan_layer_fprop` sums in bfloat16** (`src/non_linear.rs`). Relative error grows from 0.7%
   to 3.5–3.8% between 4 and 256 inputs and roughly doubles the decision flips of KAN models. This is
   host code, so it affects hardware runs. Fix: float32 accumulation. Reproduction:
   `examples/upstream/kan_bf16_accumulation.py`.
3. **KAN layers can run as a Fourier basis plus one matrix product.** Since a·cos(kx+φ) =
   (a cos φ)·cos(kx) − (a sin φ)·sin(kx), the layer needs only 2·n_in·n_k nonlinear evaluations
   (independent of the output width) and one preloaded `linear_fprop` with float32 accumulation. On
   trained models this is more accurate than PyTorch bf16 (0.43–0.46% vs 0.62–0.72% relative RMS),
   puts 95–97% of the layer's operations on the NPU (toolkit call: 52%) and moves up to half as
   many elements over the bus. It could become the toolkit's implementation of the KAN call.
4. **No parameter advantage for Fourier layers on tabular data.** Cosine MLPs and Fourier-KAN
   match ReLU MLPs at every size (all differences within ±0.003 F1, intervals include zero); the
   periodic-function advantage of the Fourier demo does not transfer.
5. **For compact models the host side, not the NPU, sets the speed.** Measured host work that a hybrid
   pipeline keeps on the CPU (bias and ReLU in the toolkit, fp32↔bf16 conversions, ≥ 1.5 µs per call)
   costs as much as or more than complete CPU inference, so offloading cannot pay off below layer widths
   of roughly 1,000–4,000. This matches the maintainer's remark that the PCIe bus is the bottleneck.
6. **API requests:** memory-handle (device-resident) variants of the nonlinearity and element-wise
   calls or a fused linear+nonlinearity call; per-function "executes on host/NPU" notes in the Python
   docs; validating `dilation`/`output_padding` early in conv calls (currently accepted, then rejected).

## What we would measure on Gen 2 hardware

| | Experiment | Success criterion |
|---|---|---|
| H1 | Error model of `linear_fprop`, the periodic nonlinearity and the KAN call; measured `tcos` curve | 95% CIs per primitive; tcos within 10⁻³ |
| H2 | Decision stability of trained matching models (plain, noise-trained, factorised KAN) | \|ΔF1\| ≤ 0.005, flips ≤ 0.5% for a compact model |
| H3 | Per-call latency, bandwidth, batch and multi-NPU scaling; cost-model calibration | model within 20% of measured; break-even width |
| H4 | Energy per pair, NPU path vs CPU path | ±10% repeatability |
| H5 | 8-hour stability across recalibration cycles | ≤ 0.1% decision changes between runs |

The harness runs unchanged with the hardware wheel. As a dry run of H2/H3 it has already measured a
real electronic NPU end to end (AMD XDNA 2 via Windows ML): accuracy within 0.0003 F1, 120–230 µs fixed
cost per call, break-even only for KAN models at large batches. The four-week plan is in
[pilot-plan.md](pilot-plan.md). Results would be published openly after your review, and the
toolkit fixes contributed upstream.
