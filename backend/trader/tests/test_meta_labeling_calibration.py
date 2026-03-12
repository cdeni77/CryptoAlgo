import numpy as np
import pandas as pd

from core.meta_labeling import calibrator_predict, fit_calibrator, train_meta_classifier


def test_fit_calibrator_all_ones_falls_back_to_none():
    scores = np.array([0.6, 0.7, 0.8, 0.9], dtype=float)
    labels = pd.Series([1, 1, 1, 1], dtype=int)

    calibrator, calibrator_type = fit_calibrator('platt', scores, labels)

    assert calibrator is None
    assert calibrator_type == 'none_single_class_fallback'
    np.testing.assert_allclose(calibrator_predict(calibrator, scores), scores)


def test_fit_calibrator_all_zeros_falls_back_to_none():
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    labels = pd.Series([0, 0, 0, 0], dtype=int)

    calibrator, calibrator_type = fit_calibrator('platt', scores, labels)

    assert calibrator is None
    assert calibrator_type == 'none_single_class_fallback'
    np.testing.assert_allclose(calibrator_predict(calibrator, scores), scores)


def test_train_meta_classifier_uses_single_class_calibration_fallback():
    n_train = 80
    n_val = 80

    X_train = pd.DataFrame({'f1': np.linspace(0.0, 1.0, n_train), 'f2': np.linspace(1.0, 2.0, n_train)})
    y_train = pd.Series(([0, 1] * (n_train // 2)), dtype=int)

    X_val = pd.DataFrame({'f1': np.linspace(0.2, 1.2, n_val), 'f2': np.linspace(1.2, 2.2, n_val)})
    y_val = pd.Series(([1] * 60) + ([0] * 20), dtype=int)
    primary_val_mask = pd.Series([True] * n_val)

    artifacts = train_meta_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        primary_val_mask,
        primary_threshold=0.6,
        meta_threshold=0.7,
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=5,
        calibration_strategy='platt',
    )

    assert artifacts.model is not None
    assert artifacts.calibrator is None
    assert artifacts.calibrator_type == 'none_single_class_fallback'


def test_fit_calibrator_binary_labels_fits_platt():
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=float)
    labels = pd.Series([0, 0, 0, 1, 1, 1], dtype=int)

    calibrator, calibrator_type = fit_calibrator('platt', scores, labels)

    assert calibrator is not None
    assert calibrator_type == 'platt'
    calibrated = calibrator_predict(calibrator, scores)
    assert calibrated.shape == scores.shape
