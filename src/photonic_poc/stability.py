"""Decision-stability analysis of a candidate backend against a reference (usually float32).

All functions take logits (log-odds), never probabilities, so nothing saturates. They are pure
NumPy and reusable for any model, public or private.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

MARGIN_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, np.inf)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z.astype(np.float64)))


def calibration(labels: np.ndarray, logits: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """Expected calibration error (equal-width bins), Brier score and the reliability bins."""
    p = _sigmoid(logits)
    y = labels.astype(np.float64)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            conf, freq = float(p[mask].mean()), float(y[mask].mean())
            ece += mask.mean() * abs(conf - freq)
            bins.append({"bin": b, "n": int(mask.sum()), "confidence": conf, "frequency": freq})
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)), "bins": bins}


def margin_profile(
    reference: np.ndarray,
    candidate: np.ndarray,
    edges: Sequence[float] = MARGIN_EDGES,
    threshold: float = 0.0,
) -> list[dict[str, float]]:
    """Flip rate by distance of the reference logit from the decision threshold."""
    margin = np.abs(reference - threshold)
    flips = (reference > threshold) != (candidate > threshold)
    rows = []
    for lo, hi in pairwise(edges):
        mask = (margin >= lo) & (margin < hi)
        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "flips": int(flips[mask].sum()),
                "rate": float(flips[mask].mean()) if mask.any() else 0.0,
            }
        )
    return rows


def deviation_by_magnitude(
    reference: np.ndarray,
    candidate: np.ndarray,
    edges: Sequence[float] = (0, 0.5, 1, 2, 4, 8, np.inf),
) -> list[dict[str, float]]:
    """|Δlogit| statistics by |reference logit|: bf16 error is relative, so it grows with size."""
    mag, dev = np.abs(reference), np.abs(candidate - reference)
    rows = []
    for lo, hi in pairwise(edges):
        mask = (mag >= lo) & (mag < hi)
        if mask.any():
            rows.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(mask.sum()),
                    "median": float(np.median(dev[mask])),
                    "p99": float(np.quantile(dev[mask], 0.99)),
                }
            )
    return rows


def kind_attribution(
    kinds: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    edges: Sequence[float] = MARGIN_EDGES,
) -> list[dict[str, Any]]:
    """Observed flips per pair kind versus flips expected from the kind's margins alone.

    The expectation applies the overall flip rate of each margin band to the kind's pairs. If
    observed ≈ expected, kinds differ only because their pairs sit closer to the boundary.
    """
    margin = np.abs(reference)
    flips = (reference > 0) != (candidate > 0)
    band = np.digitize(margin, edges[1:-1])
    band_rate = np.array(
        [flips[band == b].mean() if (band == b).any() else 0.0 for b in range(len(edges) - 1)]
    )
    rows = []
    for kind in sorted(set(kinds.tolist())):
        mask = kinds == kind
        rows.append(
            {
                "kind": kind,
                "n": int(mask.sum()),
                "observed_flips": int(flips[mask].sum()),
                "expected_flips": float(band_rate[band[mask]].sum()),
                "share_within_0.1": float(np.mean(margin[mask] < 0.1)),
            }
        )
    return rows


def precision_threshold(labels: np.ndarray, logits: np.ndarray, target: float) -> float | None:
    """Lowest logit threshold whose precision (predict ``logit >= t``) reaches ``target``."""
    order = np.argsort(-logits, kind="mergesort")
    y = labels[order].astype(bool)
    tp, fp = np.cumsum(y), np.cumsum(~y)
    precision = tp / (tp + fp)
    ok = np.nonzero(precision >= target)[0]
    if ok.size == 0:
        return None
    return float(logits[order][ok.max()])


def threshold_transfer(
    val_labels: np.ndarray,
    val_reference: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    targets: Sequence[float] = (0.99, 0.995, 0.999),
) -> list[dict[str, Any]]:
    """Thresholds tuned on reference validation logits, applied to both test logit sets."""
    y = labels.astype(bool)
    rows = []
    for target in targets:
        t = precision_threshold(val_labels, val_reference, target)
        if t is None:
            rows.append({"target_precision": target, "threshold": None})
            continue
        row: dict[str, Any] = {"target_precision": target, "threshold": t}
        for name, logits in (("reference", reference), ("candidate", candidate)):
            pred = logits >= t
            tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
            row[name] = {
                "precision": tp / (tp + fp) if tp + fp else 1.0,
                "recall": tp / int(y.sum()),
                "false_positives": fp,
            }
        row["flips_at_threshold"] = int(((reference >= t) != (candidate >= t)).sum())
        row["share_within_0.05"] = float(np.mean(np.abs(reference - t) < 0.05))
        rows.append(row)
    return rows


def decision_stability(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    kinds: np.ndarray | None = None,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    targets: Sequence[float] = (0.99, 0.995, 0.999),
) -> dict[str, Any]:
    """Everything above in one call. ``validation`` = (labels, reference logits) for thresholds."""
    diff = candidate.astype(np.float64) - reference
    report: dict[str, Any] = {
        "rel_rms_dev": float(
            np.sqrt(np.mean(diff**2) / np.mean(reference.astype(np.float64) ** 2))
        ),
        "max_abs_dev": float(np.max(np.abs(diff))),
        "p999_abs_dev": float(np.quantile(np.abs(diff), 0.999)),
        "flips": int(((reference > 0) != (candidate > 0)).sum()),
        "flip_rate": float(np.mean((reference > 0) != (candidate > 0))),
        "margin_profile": margin_profile(reference, candidate),
        "deviation_by_magnitude": deviation_by_magnitude(reference, candidate),
        "calibration_reference": calibration(labels, reference),
        "calibration_candidate": calibration(labels, candidate),
    }
    if kinds is not None:
        report["kinds"] = kind_attribution(kinds, reference, candidate)
    if validation is not None:
        report["thresholds"] = threshold_transfer(
            *validation, labels, reference, candidate, targets
        )
    return report
