from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from .calibration import brier_score, expected_calibration_error


def collect_logits(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for inputs, target in loader:
            logits.append(model(inputs.to(device)).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, probabilities.argmax(axis=1))),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
        "ece_15_bins": expected_calibration_error(probabilities, labels, n_bins=15),
        "brier": brier_score(probabilities, labels),
    }
