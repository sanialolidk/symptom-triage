import numpy as np

from src.evaluation import (
    expected_calibration_error,
    modality_disagreement,
    top_k_accuracy,
    tune_abstention_threshold,
)


def test_top_k_accuracy_perfect():
    probs = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    y = np.array([1, 0])
    assert top_k_accuracy(y, probs, k=1) == 1.0


def test_modality_disagreement_identical():
    p = np.array([0.2, 0.5, 0.3])
    assert modality_disagreement(p, p) == 0.0


def test_ece_bounded():
    probs = np.eye(3)
    y = np.array([0, 1, 2])
    assert 0.0 <= expected_calibration_error(y, probs) <= 1.0


def test_abstention_tuning_returns_threshold():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.8, 0.1, 0.1]])
    y = np.array([0, 1, 0])
    threshold, stats = tune_abstention_threshold(y, probs, target_abstain_rate=0.33)
    assert 0.0 < threshold < 1.0
    assert "abstain_rate" in stats