# metriqa-photonic-poc

Compact entity-matching networks executed with the
[Q.ANT Native Computing Toolkit](https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit),
evaluated against PyTorch on a fully synthetic, reproducible benchmark.

The goal is not to show that photonic hardware beats GPUs. It is to show that a realistic
inference workload can already be written, executed and measured for Q.ANT NPUs, using only the
official toolkit and its CPU simulator, and to report what we learned about the toolkit.

> **Status:** all Q.ANT results in this repository come from the toolkit's `cpu-backend`
> **simulator** (a noise-free bfloat16 emulation on the host CPU). Nothing has been run on a
> physical NPU yet. There are no speed or energy claims. [docs/hardware.md](docs/hardware.md)
> lists what a hardware run needs.

## The workload

Entity matching: given two company records (name, street, house number, postcode, city,
country, phone, website), decide whether they describe the same company. A record pair becomes
19 generic similarity features in `[0, 1]` (character-trigram Jaccard, `difflib` ratio,
exact-match and missing-value flags), and a small network outputs one logit.

The data is generated, never collected:

- Brand names are built from random syllables, so they do not name real companies. Streets,
  postcodes and phone numbers are synthetic; city names are public DACH geography.
- Companies come in families (sister companies sharing a brand, sometimes an address or a
  domain), and some unrelated companies share business-centre addresses.
- Positive pairs: `duplicate` (typos, abbreviations, legal-form spellings, missing fields),
  `relocated` (new address), `legal_form_change`. Negative pairs: `random`, `similar_name`
  (brand one or two edits away), `sister`, `shared_address`.
- 6,000 companies give 12,600 / 2,700 / 2,700 train / validation / test pairs, one third of
  them matches. Splits share no company, family or brand. The same seed gives the same bytes.

## Models

All three are `nn.Sequential` models built only from layers that map onto toolkit calls.

| Model | Layers | Parameters | Toolkit calls |
|---|---|---|---|
| `logreg` | Linear(19→1) | 20 | `linear_fprop_preloaded_weights`, `add_bias_fprop` |
| `mlp` | Linear(19→32), ReLU, Linear(32→16), ReLU, Linear(16→1) | 1,185 | `linear_fprop_preloaded_weights`, `add_bias_fprop`, `relu_fprop` |
| `fourier` | FourierKAN(19→4), FourierKAN(4→1), grid 4 | 645 | `calc_kan_layer_fprop`, `add_bias_fprop` |

`FourierKAN` computes `y_j = b_j + Σ_{i,l} a_jil · cos(k_l · x_i + φ_jil)` with `k_l = 1..4`.
This is the exact form of the toolkit's KAN layer, whose cosine the NPU evaluates optically.
Models are trained in PyTorch, because the toolkit has no backward pass. The `*-noise0.05`
variants are trained with Gaussian noise injected at every NPU-layer output, as in Q.ANT's own
Fourier example.

## Architecture

```
src/photonic_poc/
  data/            synthetic companies, families, perturbations, labelled pairs (JSONL)
  features.py      19 generic pair-similarity features
  models.py        logreg, mlp, FourierKAN
  training.py      deterministic CPU training (Adam, BCE, best validation loss)
  inference.py     InferenceBackend protocol; torch-fp32, torch-bf16, noisy reference
  qant_backend.py  runs an nn.Sequential through the toolkit; layer support check
  benchmark.py     model × backend benchmark and noise sweep (the reusable entry point)
  seed_study.py    retraining over seeds
  op_numerics.py   operation-level numerics of the toolkit
  reporting.py     renders results/RESULTS.md
  cli.py           the `photonic-poc` command
```

Three backends implement one protocol (`predict_logits(features) -> logits`):

- `torch-fp32`: the reference.
- `torch-bf16`: the same model in PyTorch bfloat16. It separates generic bf16 effects from
  toolkit-specific ones.
- `qant`: converts weights to bf16 once, uploads `Linear` weights to device memory when the NPU
  supports it, and executes the model layer by layer with toolkit calls. The same code targets
  the simulator and hardware; every report records which one ran.

## Results

From [results/RESULTS.md](results/RESULTS.md) (toolkit 2.5.0 `cpu-backend`, PyTorch 2.15 CPU,
test split of 2,700 pairs, threshold at probability 0.5). Seed study: each model retrained with
5 seeds, mean ± standard deviation.

| Model | Params | F1 torch-fp32 | F1 qant simulator | max abs ΔF1 | decisions flipped (mean / max of 2,700) |
|---|---|---|---|---|---|
| `logreg` | 20 | 0.9110 ± 0.0016 | 0.9109 ± 0.0014 | 0.0010 | 1.2 / 2 |
| `mlp` | 1,185 | 0.9540 ± 0.0022 | 0.9537 ± 0.0018 | 0.0017 | 1.0 / 3 |
| `fourier` | 645 | 0.9535 ± 0.0026 | 0.9537 ± 0.0014 | 0.0046 | 5.2 / 8 |
| `mlp-noise0.05` | 1,185 | 0.9550 ± 0.0018 | 0.9550 ± 0.0018 | 0.0006 | 0.8 / 2 |
| `fourier-noise0.05` | 645 | 0.9544 ± 0.0017 | 0.9545 ± 0.0017 | 0.0012 | 4.2 / 7 |

What the numbers say:

1. **The workload runs end to end through the toolkit** and keeps its matching quality on the
   simulator. For every model the F1 difference to float32 is at most 0.005, below or around
   the spread between training seeds.
2. **The Fourier network matches the MLP with about half the parameters** (F1 0.954 for both,
   645 vs 1,185 parameters). Both clearly beat logistic regression (0.911), mainly on
   relocated companies, look-alike brands and sister companies (per-kind table in RESULTS.md).
3. **The Fourier network is numerically more sensitive.** Its logits deviate 1.3–1.6% (relative
   RMS) from float32 on the simulator, against 0.4–0.5% for the MLP, and it flips about five
   times as many decisions. PyTorch bf16 shows roughly half that deviation, so most of the
   excess is toolkit-specific (next point).
4. **Toolkit finding: `calc_kan_layer_fprop` accumulates its sum in bfloat16.** Its error grows
   with fan-in (0.65% at 4 inputs, 3.5% at 256 inputs, relative RMS) while the same terms summed
   exactly stay at about 0.4%. This happens in toolkit host code, so it would affect hardware
   runs too; float32 accumulation would fix it. The matrix-vector path accumulates in float32
   and its error (0.235%) does not depend on fan-in. Details in
   [docs/sdk-capabilities.md](docs/sdk-capabilities.md#finding-the-kan-layer-accumulates-in-bf16).
5. **Under an assumed noise model** (additive Gaussian, σ = 0.1 at every NPU-layer output) F1
   drops by about 0.006 for the MLP and 0.007 for the Fourier network. Noise-aware training
   recovers part of it for the Fourier network (0.9464 → 0.9493) and makes no clear difference
   for the MLP. This is a robustness probe; hardware noise has not been measured.

Host latency is reported in RESULTS.md for completeness only: it measures the CPU emulation and
Python overhead, not an NPU.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Linux x86_64, Windows x86_64 or macOS arm64 (the
platforms of the toolkit's simulator wheel).

```bash
uv sync                                       # Python dependencies, CPU-only PyTorch
uv run python scripts/install_qant_sdk.py     # official toolkit wheel, pinned v2.5.0 + SHA-256
uv run photonic-poc sdk-info                  # toolkit version, driver, simulated NPUs
```

`uv sync` removes packages that are not in the lock file, including the toolkit. Re-run the
install script after `uv sync`; `uv run` keeps it.

## Reproduce

```bash
uv run photonic-poc reproduce                      # everything below, a few minutes on a laptop CPU
```

Or step by step:

```bash
uv run photonic-poc generate                       # data/{train,val,test}.jsonl
uv run photonic-poc train                          # artifacts/{logreg,mlp,fourier}.pt
uv run photonic-poc train --models mlp fourier --noise-std 0.05
uv run photonic-poc evaluate                       # results/benchmark.json, noise_sweep.json
uv run photonic-poc op-numerics                    # results/op_numerics.json
uv run photonic-poc seed-study                     # results/seed_study.json
uv run photonic-poc report                         # results/RESULTS.md
```

Data generation and training are deterministic for a given seed. Floating-point results can
differ in the last digits across CPUs and PyTorch builds; RESULTS.md lists the environment it was
produced in.

## Tests and CI

```bash
uv run pytest                   # simulator tests skip if the toolkit is missing
uv run ruff check . && uv run mypy
```

GitHub Actions runs lint, type checks and tests on Linux, Windows and macOS with the simulator
installed; there the toolkit tests must pass instead of being skipped (`REQUIRE_QANT=1`).

## Limitations

- Simulator only. The simulator is noise-free, uses an exact cosine instead of the hardware
  `tcos`, and its timings measure CPU emulation. See
  [docs/sdk-capabilities.md](docs/sdk-capabilities.md).
- The noise results use an assumed additive Gaussian model, not hardware data.
- The data is synthetic. It contains deliberately hard cases, but real-world matching quality
  depends on real data and is out of scope.
- Features are computed on the host; only the network runs through the toolkit.
- The models are small on purpose. Larger matrices were only studied at operation level.

## Further reading

- [docs/sdk-capabilities.md](docs/sdk-capabilities.md): what the toolkit supports, where each
  operation runs, what the simulator models, the KAN accumulation finding.
- [docs/hardware.md](docs/hardware.md): running on a physical NPU; experiments that need one.
- [docs/private-integration.md](docs/private-integration.md): reusing the benchmark with private
  models and data.
- [results/RESULTS.md](results/RESULTS.md): all numbers of the latest run.

## License

Apache-2.0, see [LICENSE](LICENSE). The Q.ANT Native Computing Toolkit is a separate project by
Q.ANT GmbH under its own Apache-2.0 license. It is downloaded at install time, not redistributed
here.
