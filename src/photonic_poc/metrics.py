"""Matching-quality and numerical-agreement metrics (numpy only, no sklearn dependency)."""

from __future__ import annotations

import numpy as np


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1, accuracy at logit 0 (probability 0.5), plus threshold-free ROC AUC."""
    y = labels.astype(bool)
    pred = logits > 0
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(np.mean(pred == y)),
        "roc_auc": roc_auc(y, logits),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U formulation with average ranks for ties."""
    y = labels.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def output_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """How far a backend's logits are from the float32 reference, and how many decisions flip."""
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    ref_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))) if diff.size else 0.0
    return {
        "rel_rms_diff": float(np.sqrt(np.mean(diff**2))) / ref_rms if ref_rms else 0.0,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "decision_flips": int(np.sum((reference > 0) != (candidate > 0))),
        "decision_flip_rate": float(np.mean((reference > 0) != (candidate > 0))),
    }
