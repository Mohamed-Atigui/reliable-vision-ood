from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, coverage: float = 0.90) -> float:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("scores cannot be empty")
    level = min(1.0, np.ceil((values.size + 1) * coverage) / values.size)
    return float(np.quantile(values, level, method="higher"))


def calibrate_aps(probabilities: np.ndarray, labels: np.ndarray, coverage: float = 0.90) -> float:
    """Calibrate adaptive prediction sets using cumulative sorted probability."""
    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    rank = np.argmax(order == labels[:, None], axis=1)
    scores = cumulative[np.arange(len(labels)), rank]
    return conformal_quantile(scores, coverage)


def prediction_sets(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative_before = np.cumsum(sorted_probabilities, axis=1) - sorted_probabilities
    sorted_membership = cumulative_before < threshold
    membership = np.zeros_like(sorted_membership, dtype=bool)
    np.put_along_axis(membership, order, sorted_membership, axis=1)
    return membership


def set_metrics(sets: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "coverage": float(sets[np.arange(len(labels)), labels].mean()),
        "mean_set_size": float(sets.sum(axis=1).mean()),
        "singleton_rate": float((sets.sum(axis=1) == 1).mean()),
    }
