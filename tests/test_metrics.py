import numpy as np
import pytest

from photonic_poc.metrics import classification_metrics, output_agreement, roc_auc


def test_classification_metrics_at_logit_zero() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    logits = np.array([2.0, 0.5, -1.0, 0.1, -2.0, -3.0])  # tp=2, fn=1, fp=1
    m = classification_metrics(labels, logits)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_roc_auc_handles_perfect_ranking_and_ties() -> None:
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(labels, np.zeros(4)) == pytest.approx(0.5)


def test_output_agreement_counts_decision_flips() -> None:
    ref = np.array([1.0, -1.0, 0.01])
    cand = np.array([1.1, -0.9, -0.01])
    a = output_agreement(ref, cand)
    assert a["decision_flips"] == 1
    assert a["max_abs_diff"] == pytest.approx(0.1)
