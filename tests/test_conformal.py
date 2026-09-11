import numpy as np

from reliable_vision.conformal import calibrate_aps, conformal_quantile, prediction_sets, set_metrics


def test_finite_sample_quantile_is_conservative():
    scores = np.arange(1, 11) / 10
    assert conformal_quantile(scores, coverage=0.90) == 1.0


def test_prediction_sets_are_never_empty():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])
    sets = prediction_sets(probabilities, threshold=0.8)
    assert np.all(sets.sum(axis=1) >= 1)


def test_aps_pipeline_returns_valid_metrics():
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
    labels = np.array([0, 1, 2])
    threshold = calibrate_aps(probabilities, labels, coverage=0.8)
    metrics = set_metrics(prediction_sets(probabilities, threshold), labels)
    assert 0 <= metrics["coverage"] <= 1
    assert 1 <= metrics["mean_set_size"] <= 3
