# Evaluating private models and data with this repository

This repository contains only synthetic data, generic features and demonstration models. A
private system can run the backends and the benchmark on its own models and data, locally,
and publish only aggregate numbers that a person has approved. Nothing is published automatically.

## Rules

- Depend on this repository from the private one (pin a tag or commit). Never the other way round.
- Private feature extraction, data, model definitions, weights and results stay in the private
  repository or on private paths. Output directories inside this repository are refused.
- Never copy private records into this repository, and never derive public examples from them.

## Quick start

1. In the private environment, install this package and expose a factory that returns an
   `nn.Sequential` (`my_private_pkg.models:build_matcher`). It is imported from your environment
   and called with no arguments; load weights inside it.
2. Save features and labels as an `.npz`:

   | array | required | content |
   |---|---|---|
   | `x_test` | yes | float32, shape (n, F) |
   | `y_test` | yes | 0/1, shape (n,) |
   | `x_val`, `y_val` | no | validation split, used to tune high-precision thresholds |
   | `kinds_test` | no | string label per test row, for per-kind accuracy (private report only) |

3. Run the evaluation, writing outside this repository:

   ```
   photonic-poc private-eval --model my_private_pkg.models:build_matcher \
       --data /private/eval.npz --out /private/results/run1
   ```

   Default backends are `torch-fp32`, `torch-bf16` and `qant`. The toolkit backend is skipped,
   with the reason recorded, if the toolkit is not installed or the model has unsupported layers
   (`check_supported` names them).

## Outputs

| file | content | shareable |
|---|---|---|
| `private_report.json` | full report: benchmark rows, latency, stability, thresholds, per-kind results, stage lists, environment | never |
| `aggregate_summary.json` | whitelisted, coarsened numbers | only after approval |
| `aggregate_summary.md` | the same, readable, headed "NOT APPROVED FOR PUBLICATION" | only after approval |

The summary is built from a whitelist: test size rounded to 100, parameters and operations to
two significant digits, offload fraction, per-backend F1, precision, recall and ROC AUC,
qant-versus-float32 deltas (F1, flip rate, relative RMS deviation, ECE), and for high-precision
targets the candidate precision and the change in false-positive rate. It has no per-record
data, arrays, stage lists, layer shapes, feature names, file paths, kind names, counts or
thresholds.

## Approval

Read `aggregate_summary.md` and decide whether even these numbers may leave the company. Then:

```
photonic-poc approve-summary --summary /private/results/run1/aggregate_summary.json \
    --approved-by "Name" --out /private/results/run1/approved_summary.json
```

This re-validates the summary against the whitelist (unknown keys are rejected), sets
`approved_for_publication` to true and records the approver, a UTC timestamp and the sha256 of
the canonical JSON (sorted keys, compact separators) of the summary without its `approval`
block. It writes one file and never commits, pushes or uploads. Publishing the approved file is
a separate, manual step. Editing the file afterwards breaks the hash.

## Python interface

`photonic_poc.private_eval` exposes `load_model`, `load_arrays`, `evaluate_private`,
`aggregate_summary`, `write_private_outputs` and `approve_summary`. Lower level: pass any
`nn.Sequential`, a float32 feature matrix and labels to
`photonic_poc.benchmark.run_benchmark`. Supported layers: `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`,
`photonic_poc.models.FourierKAN`, plus `nn.Identity` and `nn.Dropout` (no-ops at inference).

For a different execution target, implement the `InferenceBackend` protocol from
`photonic_poc.inference` (`name`, `predict_logits(features) -> logits`, `describe()`).

## What a public result may contain

Only aggregate, non-identifying statements that the owner has approved, for example "a private
model with about 12,000 parameters kept F1 within 0.002 of float32 on the simulator".
Architectures, feature lists, weights, data and per-record outputs stay private.
