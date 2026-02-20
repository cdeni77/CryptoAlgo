from scripts.validate_robustness import check_cv_consistency


def test_check_cv_consistency_boundary_variance_threshold() -> None:
    metrics = {
        'mean_oos_sharpe': 0.2,
        'min_oos_sharpe': -0.1,
        'std_oos_sharpe': 0.3,
        'n_folds': 3,
    }

    result = check_cv_consistency(metrics)

    assert result['valid'] is True
    assert result['checks']['low_variance'] is False
    assert result['checks_passed'] == 3
    assert result['consistent'] is True


def test_check_cv_consistency_boundary_min_fold_sharpe_threshold() -> None:
    metrics = {
        'mean_oos_sharpe': 0.15,
        'min_oos_sharpe': -0.3,
        'std_oos_sharpe': 0.2,
        'n_folds': 4,
    }

    result = check_cv_consistency(metrics)

    assert result['valid'] is True
    assert result['checks']['min_fold_acceptable'] is False
    assert result['checks_passed'] == 3
    assert result['consistent'] is True


def test_check_cv_consistency_prefers_fold_distribution_metrics() -> None:
    metrics = {
        'mean_oos_sharpe': 5.0,
        'min_oos_sharpe': 5.0,
        'std_oos_sharpe': 0.0,
        'n_folds': 99,
        'fold_metrics': [
            {'fold_idx': 0, 'sharpe': 0.10},
            {'fold_idx': 1, 'sharpe': -0.29},
            {'fold_idx': 2, 'sharpe': 0.20},
        ],
    }

    result = check_cv_consistency(metrics)

    assert result['used_fold_metrics'] is True
    assert result['n_folds'] == 3
    assert result['mean_oos_sharpe'] < 0.1
    assert result['min_oos_sharpe'] == -0.29
