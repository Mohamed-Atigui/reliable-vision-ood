import numpy as np

from reliable_vision.ood import energy_score, msp_score, ood_metrics, risk_coverage


def test_ood_metrics_detect_separated_scores():
    metrics = ood_metrics(np.array([0.05, 0.10, 0.15]), np.array([0.8, 0.9, 0.95]))
    assert metrics["auroc"] == 1.0
    assert metrics["fpr_at_95_tpr"] == 0.0


def test_scores_and_risk_coverage_have_expected_shapes():
    logits = np.array([[5.0, 0.0], [0.2, 0.1], [0.0, 4.0]])
    labels = np.array([0, 1, 1])
    assert msp_score(logits).shape == (3,)
    assert energy_score(logits).shape == (3,)
    curve = risk_coverage(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True), labels)
    assert len(curve["coverage"]) == len(labels)
    assert curve["coverage"][-1] == 1.0
