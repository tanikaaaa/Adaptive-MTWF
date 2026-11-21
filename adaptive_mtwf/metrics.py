"""Domain-independent multilabel metrics over binary targets and probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_THRESHOLD = 0.5
CALIBRATION_GRID = tuple(value / 100 for value in range(10, 91))


def _inputs(targets: ArrayLike, probabilities: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    targets, scores = np.asarray(targets), np.asarray(probabilities)
    if targets.ndim != 2 or scores.shape != targets.shape or not targets.size:
        raise ValueError("Targets and probabilities must share a nonempty (samples, classes) shape.")
    if targets.dtype.kind not in "biuf" or not np.isin(targets, (0, 1)).all():
        raise ValueError("Targets must be binary.")
    if (scores.dtype.kind not in "biuf" or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return targets.astype(bool, copy=False), scores


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)


def _threshold_summary(targets: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    hits = (targets & prediction).sum(axis=0)
    positives, calls = targets.sum(axis=0), prediction.sum(axis=0)
    return {
        "micro_precision": float(_ratio(hits.sum(), calls.sum())),
        "micro_recall": float(_ratio(hits.sum(), positives.sum())),
        "micro_f1": float(_ratio(2 * hits.sum(), calls.sum() + positives.sum())),
        "macro_precision": float(_ratio(hits, calls).mean()),
        "macro_recall": float(_ratio(hits, positives).mean()),
        "macro_f1": float(_ratio(2 * hits, calls + positives).mean()),
        "subset_accuracy": float(np.all(targets == prediction, axis=1).mean()),
        "mean_predicted_cardinality": float(prediction.sum(axis=1).mean()),
        "mean_absolute_cardinality_error": float(np.abs(prediction.sum(axis=1) - targets.sum(axis=1)).mean()),
    }


def threshold_metrics(targets: ArrayLike, probabilities: ArrayLike, *,
                      threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    """Use p >= threshold. Pool counts for micro scores and average classes for macro.

    Precision is TP / predictions, recall is TP / positives, and F1 is
    2TP / (positives + predictions). Undefined per-class scores contribute
    zero. Exact-match accuracy counts only rows with identical label sets.
    Cardinality error averages the absolute difference in label counts.
    """
    truth, scores = _inputs(targets, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    return _threshold_summary(truth, scores >= threshold)


def _binary_auc(truth: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(truth.sum())
    negatives = truth.size - positives
    if not positives or not negatives:
        return None
    _, score_groups = np.unique(scores, return_inverse=True)
    positive_at_score = np.bincount(score_groups[truth], minlength=score_groups.max() + 1)
    negative_at_score = np.bincount(score_groups[~truth], minlength=positive_at_score.size)
    lower_or_half_tied = np.cumsum(negative_at_score) - 0.5 * negative_at_score
    return float(np.dot(positive_at_score, lower_or_half_tied) / (positives * negatives))


def ranking_metrics(targets: ArrayLike, probabilities: ArrayLike, *, k: Optional[int] = None) -> dict:
    """Compute partial-credit A@true-K, P@K, MAP@K, and ROC AUC.

    A@true-K averages top-|Y| hits / |Y| over rows with nonempty true label
    sets Y. P@K averages hits / K. AP@K sums the precision at each relevant
    rank up to K and divides by min(K, |Y|). MAP is the mean of row APs.
    With fixed K, empty targets contribute zero P and AP. Without fixed K,
    P and MAP use |Y| and skip empty rows. Score ties use class-index order.
    ROC AUC counts positive-negative score wins with ties worth half. Macro
    AUC averages defined classes and micro AUC pools all decisions.
    """
    truth, scores = _inputs(targets, probabilities)
    if k is not None and (type(k) is not int or not 1 <= k <= scores.shape[1]):
        raise ValueError("K must be an integer from 1 through the number of classes.")
    ranked = np.take_along_axis(truth, np.argsort(-scores.astype(float), axis=1, kind="stable"), axis=1)
    true_cardinalities = truth.sum(axis=1)
    accuracies, precisions, aps = [], [], []
    for row, positive_count in zip(ranked, true_cardinalities):
        if positive_count:
            accuracies.append(float(row[:positive_count].mean()))
        limit = positive_count if k is None else k
        if limit:
            relevant = row[:limit]
            precisions.append(float(relevant.mean()))
            denominator = min(limit, positive_count)
            aps.append(float((relevant * np.cumsum(relevant) / np.arange(1, limit + 1)).sum()
                             / denominator) if denominator else 0.0)
    aucs = [_binary_auc(column, score) for column, score in zip(truth.T, scores.T)]
    defined = [value for value in aucs if value is not None]
    return {
        "a_at_true_k": float(np.mean(accuracies)) if accuracies else None,
        "precision_at_k": float(np.mean(precisions)) if precisions else None,
        "map_at_k": float(np.mean(aps)) if aps else None,
        "macro_roc_auc": float(np.mean(defined)) if defined else None,
        "micro_roc_auc": _binary_auc(truth.ravel(), scores.ravel()),
        "per_class_roc_auc": aucs,
        "auc_defined_classes": len(defined),
    }


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    macro_f1: float


def calibrate_threshold(validation_targets: ArrayLike, validation_probabilities: ArrayLike) -> ThresholdCalibration:
    """Maximize validation macro-F1 on 0.10–0.90 in steps of 0.01.

    Resolve ties by distance from 0.50, then by the lower threshold. Pass
    validation arrays only. The result is JSON-ready via dataclasses.asdict.
    """
    truth, scores = _inputs(validation_targets, validation_probabilities)
    thresholds = (ThresholdCalibration(value, _threshold_summary(truth, scores >= value)["macro_f1"])
                  for value in CALIBRATION_GRID)
    return max(thresholds, key=lambda value: (value.macro_f1, -abs(round(value.threshold * 100) - 50), -value.threshold))
