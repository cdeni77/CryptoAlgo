"""The classifier, which predicts a *correction* to the baseline.

The architecture is one decision and it makes everything else honest: the
baseline's logit enters LightGBM as an `init_score`, so the model fits the
residual. An untrained model reproduces the baseline exactly, every tree it
grows is incremental skill by construction, and "did it beat the baseline"
stops being a comparison of two numbers computed by different code paths and
becomes a property of a single fitted object.

Three things follow from that choice, and they are the reason for it:

* **The objective is the right one.** Cross-entropy on a binary label, not a
  regression on a return that is then thresholded. The previous incarnation of
  this project regressed net return and took its sign, which counted every flat
  bar as a miss and optimised magnitude accuracy nobody was paid for. Here the
  quantity being fitted is exactly the quantity being traded: a probability.

* **Abstention is native.** A calibrated probability compares directly against
  the price, so "no trade" is not a separate model or a tuned threshold on a
  score of unknown units — it is `edge below the gate`, in probability points,
  against a break-even the fee schedule defines.

* **Overconfidence is measurable in one number.** `residual_scale` is a single
  coefficient fitted on held-out training rows: how much of the correction the
  model claims actually survives out of sample. The last version of this repo
  discovered its predictions were thirty-four times too confident only by
  regressing realised on predicted after the fact. Here it is `alpha`, it is
  fitted before anything is traded, and a value near zero says the model found
  nothing however good its in-sample loss looked.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from core.baseline import BarrierBaseline, clip_prob, expit, log_loss, logit
from core.config import Config, DEFAULT_CONFIG
from core.features import CONTROL_GROUPS, feature_columns

logger = logging.getLogger(__name__)

BASELINE_LOGIT = 'baseline_probability_logit'
# Fraction of the training windows held back to fit `residual_scale` and to
# early-stop on. Taken from the *end* of training, so it is the most recent
# data and the shrinkage is measured on the regime nearest the test block.
INNER_VALIDATION_FRACTION = 0.2


def _feature_matrix(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f'feature columns absent from the table: {missing}')
    return table.loc[:, list(columns)].to_numpy(dtype=float)


@dataclass
class ForecastModel:
    """A fitted correction to a fitted baseline. One object, one provenance."""

    booster: Any
    features: list[str]
    baseline: BarrierBaseline
    # Everything else the live path needs: the fitted volatility models and
    # seasonality factors. Without these the artifact can be evaluated and not
    # deployed, which is a distinction nothing surfaced until the live path
    # tried to score a window.
    scoring: Any = None
    residual_scale: float = 1.0
    groups: tuple[str, ...] = ()
    n_train_rows: int = 0
    n_train_windows: int = 0
    empty_features: tuple[str, ...] = ()
    best_iteration: Optional[int] = None
    train_log_loss: float = float('nan')
    inner_log_loss: float = float('nan')
    inner_baseline_log_loss: float = float('nan')
    config_provenance: dict = field(default_factory=dict)

    # ---- prediction -----------------------------------------------------
    def raw_correction(self, table: pd.DataFrame) -> np.ndarray:
        """The model's logit correction, before shrinkage."""
        matrix = _feature_matrix(table, self.features)
        return np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float)

    def predict(self, table: pd.DataFrame, *, shrink: bool = True) -> np.ndarray:
        """Calibrated probability that the window settles above its strike."""
        if BASELINE_LOGIT not in table.columns:
            raise ValueError(
                f'{BASELINE_LOGIT} is missing — score through '
                f'core.dataset.apply_fold, which attaches it from the fold\'s own '
                f'baseline. Recomputing it here would silently use a different one.'
            )
        base = table[BASELINE_LOGIT].to_numpy(dtype=float)
        alpha = self.residual_scale if shrink else 1.0
        return clip_prob(expit(base + alpha * self.raw_correction(table)))

    def predict_baseline(self, table: pd.DataFrame) -> np.ndarray:
        return clip_prob(table['baseline_probability'].to_numpy(dtype=float))

    # ---- provenance -----------------------------------------------------
    def importance(self, kind: str = 'gain') -> pd.DataFrame:
        values = self.booster.feature_importance(importance_type=kind)
        control = {c for g in CONTROL_GROUPS for c in feature_columns([g])}
        frame = pd.DataFrame({'feature': self.features, kind: values})
        frame['is_control'] = frame['feature'].isin(control)
        total = frame[kind].sum()
        frame['share'] = frame[kind] / total if total else np.nan
        return frame.sort_values(kind, ascending=False, ignore_index=True)

    @property
    def control_importance_share(self) -> float:
        """Share of total gain taken by the control group.

        Hour-of-day cannot forecast direction. A large share here does not mean
        the clock works; it means the measurement is picking up something that
        is not a forecast, and the last version of this project had its 27-cell
        survey won by exactly this group.
        """
        frame = self.importance()
        return float(frame.loc[frame['is_control'], 'share'].sum())

    def provenance(self) -> dict:
        return {
            'features': list(self.features),
            'n_features': len(self.features),
            # `features` is already the populated list, so subtracting the empty
            # ones again reported 28 for 35. It goes into every provenance record
            # the trial count is read from.
            'n_features_populated': len(self.features),
            'empty_features': list(self.empty_features),
            'groups': list(self.groups),
            'residual_scale': self.residual_scale,
            'n_train_rows': self.n_train_rows,
            'n_train_windows': self.n_train_windows,
            'best_iteration': self.best_iteration,
            'train_log_loss': self.train_log_loss,
            'inner_log_loss': self.inner_log_loss,
            'inner_baseline_log_loss': self.inner_baseline_log_loss,
            'inner_log_loss_skill': self.inner_baseline_log_loss - self.inner_log_loss,
            'control_importance_share': self.control_importance_share,
            'baseline': self.baseline.provenance(),
            'deployable': self.scoring is not None,
            'config': self.config_provenance,
        }

    @property
    def deployable(self) -> bool:
        """Can this artifact score a window it has never seen?"""
        return self.scoring is not None

    def summary(self) -> str:
        skill = self.inner_baseline_log_loss - self.inner_log_loss
        return (
            f'model: {len(self.features)} features '
            f'({len(self.empty_features)} empty), {self.best_iteration} trees, '
            f'alpha={self.residual_scale:.3f} | inner log loss {self.inner_log_loss:.5f} '
            f'vs baseline {self.inner_baseline_log_loss:.5f} '
            f'(skill {skill:+.5f}) | control gain share '
            f'{self.control_importance_share:.1%}'
        )

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix('.provenance.json').write_text(
            json.dumps(self.provenance(), indent=2, default=str))
        return path

    @staticmethod
    def load(path: str | Path) -> 'ForecastModel':
        import joblib
        return joblib.load(Path(path))


def _fit_residual_scale(
    baseline_logit: np.ndarray,
    correction: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """One coefficient: how much of the claimed correction survives.

    Fitted by minimising held-out log loss over a single scalar, which is the
    smallest recalibration that can exist and therefore the one least able to
    manufacture skill. Clipped to [0, 2]: a negative alpha would mean the model
    is anti-predictive and the honest response is to report zero skill rather
    than to invert it, which is curve-fitting on the validation split.
    """
    from scipy import optimize

    # Refuse non-finite input rather than optimising over it. A single NaN makes
    # the objective NaN at every alpha, `minimize_scalar` gives up, and it
    # returns its golden-section bracket seed 0.7639320225 with
    # `success=False`. That value was never checked, so the *overfitting
    # detector* returned a search constant that sails past its own
    # `residual_scale >= 0.25` gate. On a five-year BTC walk-forward four of six
    # folds returned exactly 0.7639320225, and those four folds carried the
    # reported skill.
    finite = (np.isfinite(baseline_logit) & np.isfinite(correction)
              & np.isfinite(np.asarray(outcome, dtype=float)))
    if not finite.all():
        raise ValueError(
            f'{int((~finite).sum())} of {finite.size} validation rows are '
            f'non-finite, so the shrinkage cannot be fitted. Scoring them would '
            f'silently return scipy\'s bracket seed and pass the residual_scale '
            f'gate on an unfitted constant. Fix the upstream data hole.'
        )

    def objective(alpha: float) -> float:
        return log_loss(outcome, expit(baseline_logit + float(alpha) * correction))

    result = optimize.minimize_scalar(objective, bounds=(0.0, 2.0), method='bounded')
    if not result.success or not np.isfinite(result.x):
        raise ValueError(
            f'the shrinkage fit did not converge ({getattr(result, "message", "")!r}). '
            f'Reporting its abandoned bracket point as a fitted alpha is how an '
            f'unfitted constant reaches a gate.'
        )
    return float(np.clip(result.x, 0.0, 2.0))


def fit_model(
    train: pd.DataFrame,
    baseline: BarrierBaseline,
    config: Config = DEFAULT_CONFIG,
    *,
    groups: Optional[Sequence[str]] = None,
    weights: Optional[np.ndarray] = None,
    scoring: Any = None,
) -> ForecastModel:
    """Fit the residual classifier on a training slice.

    The inner validation split is taken from the *end* of training — the most
    recent windows — and is used for both early stopping and the shrinkage
    coefficient. Splitting at random would put four offsets of the same window
    on both sides and the early-stopping signal would be measuring memorisation.
    """
    import lightgbm as lgb

    columns = feature_columns(groups)
    if BASELINE_LOGIT not in train.columns:
        raise ValueError(f'{BASELINE_LOGIT} is missing from the training table')

    populated = [
        c for c in columns
        if c in train.columns and np.isfinite(train[c].to_numpy(dtype=float)).any()
    ]
    empty = tuple(c for c in columns if c not in populated)
    if empty:
        logger.warning(
            'these features are entirely NaN and carry no information: %s. '
            'A group that produced nothing arrives with the same shape as one '
            'that worked, so this is the only place it is visible.', list(empty))
    if not populated:
        raise ValueError('no populated features')

    windows = pd.DatetimeIndex(sorted(train['window_open'].unique()))
    if len(windows) < 20:
        raise ValueError(f'{len(windows)} training windows is not enough to fit')
    cut = windows[int(len(windows) * (1.0 - INNER_VALIDATION_FRACTION))]
    inner_train = train.loc[train['window_open'] < cut]
    inner_valid = train.loc[train['window_open'] >= cut]
    if inner_valid.empty or inner_train.empty:
        raise ValueError('inner validation split is empty')

    def dataset(frame: pd.DataFrame, reference=None):
        w = None
        if weights is not None:
            w = np.asarray(weights, dtype=float)[frame.index.to_numpy()]
        return lgb.Dataset(
            _feature_matrix(frame, populated),
            label=frame['outcome'].to_numpy(dtype=float),
            init_score=frame[BASELINE_LOGIT].to_numpy(dtype=float),
            weight=w, feature_name=list(populated), reference=reference,
            free_raw_data=False,
        )

    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
        'learning_rate': config.learning_rate, 'num_leaves': config.num_leaves,
        'max_depth': config.max_depth, 'min_child_samples': config.min_child_samples,
        'bagging_fraction': config.subsample, 'bagging_freq': 1,
        'feature_fraction': config.colsample_bytree, 'lambda_l2': config.reg_lambda,
        'deterministic': True, 'seed': 17,
    }
    train_set = dataset(inner_train)
    valid_set = dataset(inner_valid, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=config.n_estimators,
        valid_sets=[valid_set], valid_names=['inner'],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )

    valid_matrix = _feature_matrix(inner_valid, populated)
    correction = np.asarray(booster.predict(valid_matrix, raw_score=True), dtype=float)
    base_logit = inner_valid[BASELINE_LOGIT].to_numpy(dtype=float)
    outcome = inner_valid['outcome'].to_numpy(dtype=float)
    alpha = _fit_residual_scale(base_logit, correction, outcome)

    model = ForecastModel(
        booster=booster, features=list(populated), baseline=baseline,
        scoring=scoring, residual_scale=alpha,
        groups=tuple(groups) if groups else (),
        n_train_rows=len(train), n_train_windows=len(windows),
        empty_features=empty, best_iteration=booster.best_iteration,
        train_log_loss=log_loss(
            inner_train['outcome'].to_numpy(dtype=float),
            expit(inner_train[BASELINE_LOGIT].to_numpy(dtype=float)
                  + alpha * np.asarray(
                      booster.predict(_feature_matrix(inner_train, populated), raw_score=True),
                      dtype=float))),
        inner_log_loss=log_loss(outcome, expit(base_logit + alpha * correction)),
        inner_baseline_log_loss=log_loss(outcome, expit(base_logit)),
        config_provenance=config.provenance(),
    )
    logger.info(model.summary())
    return model
