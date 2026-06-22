"""Metrics for eval."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([int(y in row) for y, row in zip(y_true, topk)]))


def mean_reciprocal_rank(y_true: np.ndarray, probs: np.ndarray) -> float:
    ranks = []
    for y, row in zip(y_true, probs):
        order = np.argsort(row)[::-1]
        rank = int(np.where(order == y)[0][0]) + 1
        ranks.append(1.0 / rank)
    return float(np.mean(ranks))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def tune_abstention_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_abstain_rate: float = 0.12,
) -> tuple[float, dict]:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    candidates = np.unique(np.round(confidences, 4))
    if len(candidates) < 5:
        candidates = np.linspace(0.15, 0.95, 40)

    best_threshold = 0.38
    best_score = -1.0
    best_stats: dict = {}

    for threshold in candidates:
        active = confidences >= threshold
        abstain_rate = float(1.0 - active.mean())
        if active.sum() == 0:
            continue
        selective_acc = float(accuracy_score(y_true[active], preds[active]))
        selective_f1 = float(f1_score(y_true[active], preds[active], average="macro", zero_division=0))
        penalty = abs(abstain_rate - target_abstain_rate)
        score = selective_acc - 0.35 * penalty
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_stats = {
                "threshold": best_threshold,
                "abstain_rate": round(abstain_rate, 4),
                "selective_accuracy": round(selective_acc, 4),
                "selective_macro_f1": round(selective_f1, 4),
                "coverage": round(float(active.mean()), 4),
            }
    return best_threshold, best_stats


def modality_disagreement(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """Symmetric KL-derived disagreement in [0, 1]."""
    eps = 1e-8
    pa = np.clip(probs_a, eps, 1.0)
    pb = np.clip(probs_b, eps, 1.0)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    kl = 0.5 * (np.sum(pa * np.log(pa / pb)) + np.sum(pb * np.log(pb / pa)))
    return float(min(1.0, kl / 2.0))


def summarize_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    abstain_threshold: float,
) -> dict:
    preds = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)
    abstain_mask = max_prob < abstain_threshold
    active = ~abstain_mask
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
        "top3_accuracy": round(top_k_accuracy(y_true, probs, k=3), 4),
        "mrr": round(mean_reciprocal_rank(y_true, probs), 4),
        "ece": round(expected_calibration_error(y_true, probs), 4),
        "abstain_rate": round(float(abstain_mask.mean()), 4),
        "abstain_threshold": abstain_threshold,
        "accuracy_when_not_abstaining": round(
            float(accuracy_score(y_true[active], preds[active])) if active.any() else 0.0, 4
        ),
    }


def cross_validate_structured(X: np.ndarray, y: np.ndarray, model, folds: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return {
        "folds": folds,
        "macro_f1_mean": round(float(scores.mean()), 4),
        "macro_f1_std": round(float(scores.std()), 4),
        "per_fold": [round(float(s), 4) for s in scores],
    }