# Reusing the benchmark with private models and data

This repository contains only synthetic data, generic features and demonstration models. A
private system can reuse the backends and the benchmark without contributing anything back.

## Rules

- Depend on this repository from the private one (pin a tag or commit). Never the other way round.
- Private feature extraction, data, model definitions, weights and results stay in the private
  repository. Write private reports to private paths only.
- Never copy private records into this repository, and never derive public examples from them.

## Interface

The benchmark needs three things: an `nn.Sequential`, a float32 feature matrix and labels. It
does not know how features were computed.

```python
from torch import nn

from photonic_poc.benchmark import run_benchmark
from photonic_poc.qant_backend import check_supported

model: nn.Sequential = load_private_model()  # private code
x_test, y_test = private_features_and_labels()  # private code, float32 (n, F), (n,)

problems = check_supported(model)  # works without the toolkit installed
assert not problems, problems

report = run_benchmark(
    {"private-model": model}, x_test, y_test, backends=("torch-fp32", "torch-bf16", "qant")
)
save_privately(report)  # JSON-serialisable dict
```

Supported layers: `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `photonic_poc.models.FourierKAN`, plus
`nn.Identity` and `nn.Dropout` (no-ops at inference). `check_supported` names every other layer.
`run_benchmark` returns F1, precision, recall, ROC AUC, logit deviation and decision flips against
float32, parameter count and host latency per backend. `noise_sweep` gives F1 under the assumed
noise model.

For a different execution target, implement the `InferenceBackend` protocol from
`photonic_poc.inference` (`name`, `predict_logits(features) -> logits`, `describe()`).

## What a public result may contain

Only aggregate, non-identifying statements that the owner has approved, for example "a private
model with N parameters kept F1 within X of float32 on the simulator". Architectures, feature
lists, weights, data and per-record outputs stay private.
