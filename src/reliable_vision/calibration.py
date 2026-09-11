from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    probabilities = softmax(logits, temperature)
    chosen = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    grid_size: int = 400,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit one scalar temperature on held-out calibration logits."""
    candidates = np.geomspace(bounds[0], bounds[1], grid_size)
    losses = [negative_log_likelihood(logits, labels, value) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    target = np.eye(probabilities.shape[1])[labels]
    return float(np.square(probabilities - target).sum(axis=1).mean())
