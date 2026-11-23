"""Small statistics helpers: Pareto fronts, paired bootstrap of F1 differences, rank correlation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pareto_front(costs: Sequence[float], scores: Sequence[float]) -> list[int]:
    """Indices of points not dominated by any other (lower cost and higher score is better)."""
    order = sorted(range(len(costs)), key=lambda i: (costs[i], -scores[i]))
    front, best = [], -np.inf
    for i in order:
        if scores[i] > best:
            front.append(i)
            best = scores[i]
    return front


def _f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    out = np.zeros_like(denom, dtype=np.float64)
    return np.asarray(np.divide(2 * tp, denom, out=out, where=denom > 0))


def paired_bootstrap_f1(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Seed-averaged F1(a) - F1(b) with a 95% percentile interval, resampling test pairs.

    ``preds_a`` and ``preds_b`` are boolean arrays of shape (n_seeds, n_pairs). Both models are
    evaluated on the same resampled pairs, so the interval reflects test-set sampling noise
    on top of the seed spread already averaged in.
    """
    y = labels.astype(bool)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=n_boot).T  # (n, B)

    def f1(preds: np.ndarray) -> np.ndarray:
        p = preds.astype(np.float64)
        tp = (p * y) @ weights
        fp = (p * ~y) @ weights
        fn = ((1 - p) * y) @ weights
        return _f1_from_counts(tp, fp, fn).mean(axis=0)  # mean over seeds -> (B,)

    diff = f1(preds_a) - f1(preds_b)
    observed = float(_point_f1(y, preds_a) - _point_f1(y, preds_b))
    low, high = np.percentile(diff, [2.5, 97.5])
    return {
        "delta_f1": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_a_better": float(np.mean(diff > 0)),
    }


def _point_f1(y: np.ndarray, preds: np.ndarray) -> float:
    p = preds.astype(bool)
    tp = (p & y).sum(axis=1)
    fp = (p & ~y).sum(axis=1)
    fn = (~p & y).sum(axis=1)
    return float(_f1_from_counts(tp, fp, fn).mean())


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])
