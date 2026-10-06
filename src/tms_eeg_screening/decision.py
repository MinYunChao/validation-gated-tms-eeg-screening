from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import aggregate_subject_probabilities, classification_metrics


def tangent_features(waveforms: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    """Compute stabilized log-covariance upper-triangle features per epoch."""
    output = []
    for epoch in np.asarray(waveforms, dtype=np.float64):
        centered = epoch - epoch.mean(axis=1, keepdims=True)
        covariance = centered @ centered.T / max(epoch.shape[1] - 1, 1)
        scale = np.trace(covariance) / covariance.shape[0]
        covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(
            covariance.shape[0]
        )
        values, vectors = eigh(covariance)
        matrix_log = (vectors * np.log(np.clip(values, 1e-8, None))) @ vectors.T
        rows, cols = np.triu_indices_from(matrix_log)
        feature = matrix_log[rows, cols]
        feature[rows != cols] *= np.sqrt(2.0)
        output.append(feature)
    return np.asarray(output, dtype=np.float32)


@dataclass
class ResidualBranch:
    estimator: object
    residual_lambda: float
    shrinkage: float

    def epoch_probabilities(self, aligned_waveforms: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(tangent_features(aligned_waveforms, self.shrinkage))[
            :, 1
        ]

    def combine(
        self, clean_probabilities: np.ndarray, residual_probabilities: np.ndarray
    ) -> np.ndarray:
        clean_logits = logit(np.clip(clean_probabilities, 1e-6, 1 - 1e-6))
        residual_logits = logit(np.clip(residual_probabilities, 1e-6, 1 - 1e-6))
        return expit(clean_logits + self.residual_lambda * residual_logits)


def fit_residual_branch(
    train_waveforms: np.ndarray,
    train_labels: np.ndarray,
    validation_waveforms: np.ndarray,
    validation_labels: np.ndarray,
    validation_subject_ids: np.ndarray,
    clean_validation_probabilities: np.ndarray,
    lambda_candidates: list[float],
    shrinkage: float,
    seed: int,
) -> tuple[ResidualBranch, list[dict[str, float]]]:
    estimator = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
    )
    estimator.fit(tangent_features(train_waveforms, shrinkage), train_labels)
    residual_validation = estimator.predict_proba(
        tangent_features(validation_waveforms, shrinkage)
    )[:, 1]
    rows = []
    best_lambda = 0.0
    best_key = None
    for value in lambda_candidates:
        temporary = ResidualBranch(estimator, float(value), shrinkage)
        combined = temporary.combine(clean_validation_probabilities, residual_validation)
        subject = aggregate_subject_probabilities(
            validation_labels, combined, validation_subject_ids
        )
        metrics = classification_metrics(subject["labels"], subject["probabilities"])
        row = {"lambda": float(value), **metrics}
        rows.append(row)
        key = (metrics["weighted_f1"], metrics["balanced_accuracy"], -abs(float(value)))
        if best_key is None or key > best_key:
            best_key, best_lambda = key, float(value)
    return ResidualBranch(estimator, best_lambda, shrinkage), rows


def apply_metadata_policy(
    clean_probabilities: np.ndarray,
    metadata_probabilities: np.ndarray,
    policy: str,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean_probabilities, dtype=float)
    metadata = np.asarray(metadata_probabilities, dtype=float)
    if policy == "clean":
        active = np.zeros(len(clean), dtype=bool)
    elif policy == "specificity_guard":
        active = (clean >= 0.5) & (metadata <= 0.5 - confidence)
    elif policy == "sensitivity_guard":
        active = (clean < 0.5) & (metadata >= 0.5 + confidence)
    elif policy == "bidirectional_guard":
        active = ((clean >= 0.5) & (metadata <= 0.5 - confidence)) | (
            (clean < 0.5) & (metadata >= 0.5 + confidence)
        )
    else:
        raise ValueError(f"Unknown metadata policy: {policy}")
    return np.where(active, metadata, clean), active


@dataclass
class MetadataGate:
    estimator: object
    policy: str
    confidence: float
    feature_names: list[str]

    def apply(self, clean_probabilities: np.ndarray, metadata: np.ndarray):
        metadata_probabilities = self.estimator.predict_proba(metadata)[:, 1]
        return apply_metadata_policy(
            clean_probabilities, metadata_probabilities, self.policy, self.confidence
        )


def fit_metadata_gate(
    train_metadata: np.ndarray,
    train_labels: np.ndarray,
    validation_metadata: np.ndarray,
    validation_labels: np.ndarray,
    clean_validation_probabilities: np.ndarray,
    feature_names: list[str],
    c_values: list[float],
    confidence_values: list[float],
    policies: list[str],
    seed: int,
) -> tuple[MetadataGate, list[dict[str, float]]]:
    candidates = []
    models = {}
    for c_value in c_values:
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(C=c_value, max_iter=2000, random_state=seed)
        )
        estimator.fit(train_metadata, train_labels)
        models[c_value] = estimator
        metadata_probabilities = estimator.predict_proba(validation_metadata)[:, 1]
        for policy in policies:
            values = [0.0] if policy == "clean" else confidence_values
            for confidence in values:
                fused, active = apply_metadata_policy(
                    clean_validation_probabilities, metadata_probabilities, policy, confidence
                )
                metrics = classification_metrics(validation_labels, fused)
                candidates.append(
                    {
                        "C": float(c_value),
                        "policy": policy,
                        "confidence": float(confidence),
                        "activation_rate": float(active.mean()),
                        **metrics,
                    }
                )
    best = max(
        candidates,
        key=lambda row: (
            row["weighted_f1"],
            row["balanced_accuracy"],
            row["specificity"],
            row["sensitivity"],
            -row["activation_rate"],
            row["policy"] == "clean",
        ),
    )
    return (
        MetadataGate(models[best["C"]], best["policy"], best["confidence"], feature_names),
        candidates,
    )
