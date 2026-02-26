import pytest
import numpy as np

from scripts.train_model import MLSystem


def test_compute_member_pair_correlation_valid_aligned():
    left = {
        "val_probs": [0.1, 0.2, 0.4, 0.8],
        "val_index": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
            "2024-01-01T03:00:00+00:00",
        ],
    }
    right = {
        "val_probs": [0.2, 0.4, 0.8, 0.9],
        "val_index": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
            "2024-01-01T03:00:00+00:00",
        ],
    }

    result = MLSystem.compute_member_pair_correlation(left, right)
    assert result["status"] == "ok"
    assert result["na_reason"] is None
    assert result["n_overlap"] == 4
    assert np.isfinite(result["corr"])


def test_compute_member_pair_correlation_no_overlap():
    left = {
        "val_probs": [0.1, 0.2],
        "val_index": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
    }
    right = {
        "val_probs": [0.3, 0.4],
        "val_index": ["2024-01-02T00:00:00+00:00", "2024-01-02T01:00:00+00:00"],
    }

    result = MLSystem.compute_member_pair_correlation(left, right)
    assert result["status"] == "na"
    assert result["na_reason"] == "no_timestamp_overlap"
    assert result["n_overlap"] == 0


def test_compute_member_pair_correlation_zero_variance_with_fallback_metric():
    left = {
        "val_probs": [0.5, 0.5, 0.5],
        "val_index": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
        ],
    }
    right = {
        "val_probs": [0.7, 0.7, 0.7],
        "val_index": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
        ],
    }

    result = MLSystem.compute_member_pair_correlation(left, right)
    assert result["status"] == "na"
    assert result["na_reason"] == "zero_variance_both"
    assert result["mean_abs_diff"] == pytest.approx(0.2)
    assert result["max_abs_diff"] == pytest.approx(0.2)


def test_compute_member_pair_correlation_nonfinite_filtered_path():
    left = {
        "val_probs": [np.nan, np.inf],
        "val_index": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
    }
    right = {
        "val_probs": [0.1, 0.2],
        "val_index": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
    }

    result = MLSystem.compute_member_pair_correlation(left, right)
    assert result["status"] == "na"
    assert result["na_reason"] == "all_nonfinite_filtered"
    assert result["n_overlap"] == 2
    assert result["n_valid_after_filter"] == 0


def test_compute_member_pair_correlation_length_mismatch_with_overlap():
    left = {
        "val_probs": [0.1, 0.2, 0.3, 0.4],
        "val_index": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
            "2024-01-01T03:00:00+00:00",
        ],
    }
    right = {
        "val_probs": [0.15, 0.25],
        "val_index": ["2024-01-01T01:00:00+00:00", "2024-01-01T02:00:00+00:00"],
    }

    result = MLSystem.compute_member_pair_correlation(left, right)
    assert result["status"] == "ok"
    assert result["na_reason"] is None
    assert result["n_left_raw"] == 4
    assert result["n_right_raw"] == 2
    assert result["n_overlap"] == 2
