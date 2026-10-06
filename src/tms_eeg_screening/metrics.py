from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def _weighted_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="weighted", zero_division=0))


def aggregate_subject_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = []
    for subject in sorted(set(map(str, subject_ids))):
        mask = np.asarray([str(value) == subject for value in subject_ids])
        subject_labels = np.unique(labels[mask])
        if len(subject_labels) != 1:
            raise ValueError(f"subject {subject} has inconsistent labels")
        rows.append((subject, int(subject_labels[0]), float(np.mean(probabilities[mask]))))
    return {
        "subject_ids": np.asarray([row[0] for row in rows]),
        "labels": np.asarray([row[1] for row in rows], dtype=int),
        "probabilities": np.asarray([row[2] for row in rows], dtype=float),
    }


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else math.nan
    specificity = float(tn / (tn + fp)) if tn + fp else math.nan
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probabilities))
        balanced_accuracy = 0.5 * (sensitivity + specificity)
    else:
        auc = math.nan
        balanced_accuracy = math.nan
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": _weighted_f1(labels, predictions),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "npv": float(tn / max(tn + fn, 1)),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def paired_bootstrap(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    n_bootstrap: int = 20000,
    seed: int = 20260610,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline_probabilities, dtype=float)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    baseline_predictions = (baseline >= 0.5).astype(int)
    candidate_predictions = (candidate >= 0.5).astype(int)
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(labels), len(labels))
        base_metric = _weighted_f1(labels[indices], baseline_predictions[indices])
        candidate_metric = _weighted_f1(labels[indices], candidate_predictions[indices])
        deltas[iteration] = candidate_metric - base_metric
    observed = _weighted_f1(labels, candidate_predictions) - _weighted_f1(
        labels, baseline_predictions
    )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_subjects": len(labels),
        "n_bootstrap": int(n_bootstrap),
        "observed_weighted_f1_delta": observed,
        "ci_2_5": float(low),
        "ci_97_5": float(high),
        "seed": int(seed),
    }
