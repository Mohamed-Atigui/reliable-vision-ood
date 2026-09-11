from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .calibration import softmax


def msp_score(logits: np.ndarray) -> np.ndarray:
    """OOD score based on one minus maximum softmax probability."""
    return 1.0 - softmax(logits).max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
    return -temperature * logsumexp


def ood_metrics(in_scores: np.ndarray, out_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    scores = np.concatenate([in_scores, out_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[eligible[0]]) if eligible.size else 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_out": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": fpr95,
    }


def risk_coverage(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    order = np.argsort(-confidence)
    cumulative_accuracy = np.cumsum(correct[order]) / np.arange(1, len(labels) + 1)
    return {
        "coverage": (np.arange(1, len(labels) + 1) / len(labels)).tolist(),
        "risk": (1.0 - cumulative_accuracy).tolist(),
    }
