import numpy as np

from tms_eeg_screening.decision import (
    ResidualBranch,
    apply_metadata_policy,
    tangent_features,
)
from tms_eeg_screening.metrics import classification_metrics, paired_bootstrap


def test_tangent_features_are_finite():
    rng = np.random.default_rng(3)
    waveforms = rng.normal(size=(6, 8, 128)).astype(np.float32)
    features = tangent_features(waveforms)
    assert features.shape == (6, 36)
    assert np.isfinite(features).all()


def test_residual_lambda_changes_probabilities():
    clean = np.asarray([0.4, 0.6])
    residual = np.asarray([0.8, 0.2])
    branch_zero = ResidualBranch(None, 0.0, 0.2)
    branch_nonzero = ResidualBranch(None, 0.1, 0.2)
    assert np.allclose(branch_zero.combine(clean, residual), clean)
    assert not np.allclose(branch_nonzero.combine(clean, residual), clean)


def test_metadata_guard_only_changes_eligible_rows():
    clean = np.asarray([0.7, 0.8, 0.3])
    metadata = np.asarray([0.2, 0.49, 0.9])
    fused, active = apply_metadata_policy(clean, metadata, "specificity_guard", 0.1)
    assert active.tolist() == [True, False, False]
    assert np.allclose(fused, [0.2, 0.8, 0.3])


def test_single_class_bootstrap_resamples_are_supported():
    labels = np.asarray([0, 0, 1])
    clean = np.asarray([0.1, 0.7, 0.6])
    candidate = np.asarray([0.1, 0.4, 0.7])
    result = paired_bootstrap(labels, clean, candidate, n_bootstrap=50, seed=2)
    assert result["n_bootstrap"] == 50
    single_class = classification_metrics(np.asarray([0, 0]), np.asarray([0.1, 0.2]))
    assert np.isnan(single_class["auc"])
    assert np.isnan(single_class["balanced_accuracy"])
