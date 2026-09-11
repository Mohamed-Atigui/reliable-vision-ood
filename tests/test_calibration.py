import numpy as np

from reliable_vision.calibration import (
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1.0, 2.0], [-2.0, 4.0]]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_fitting_does_not_worsen_calibration_nll():
    logits = np.array([[8.0, 0.0], [7.0, 1.0], [6.0, 2.0], [5.0, 3.0]])
    labels = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels, grid_size=200)
    assert negative_log_likelihood(logits, labels, temperature) <= negative_log_likelihood(logits, labels)


def test_calibration_metrics_are_bounded():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    labels = np.array([0, 1])
    assert 0 <= expected_calibration_error(probabilities, labels) <= 1
    assert 0 <= brier_score(probabilities, labels) <= 2
