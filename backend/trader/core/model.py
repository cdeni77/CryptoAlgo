"""The classifier, which predicts a correction to an init forecaster.

The init forecaster's logit (the baseline, or the market) is LightGBM's
`init_score`, so the model fits the residual. An untrained model reproduces the
init forecaster exactly, and every tree is incremental skill by construction.

* The objective is cross-entropy on the binary label: a probability, the
  quantity being traded.
* Abstention is native: a calibrated probability compares directly against the
  price and the fee-defined break-even.
* `residual_scale` (alpha) is one coefficient fitted on held-out training rows:
  how much of the claimed correction survives out of sample.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from core.baseline import BarrierBaseline, clip_prob, expit, log_loss, logit
from core.config import Config, DEFAULT_CONFIG
from core.features import (
    BOOK_GROUPS, CONTROL_GROUPS, FEATURE_GROUPS, feature_columns,
)

logger = logging.getLogger(__name__)

BASELINE_LOGIT = 'baseline_probability_logit'
MARKET_LOGIT = 'market_probability_logit'

# Which forecaster the correction is fitted on top of, selected by
# `Config.init_score_source`.
#
# 'market' fits the residual `logit(truth) - logit(price)`: how the price is wrong.
# An untrained market-init model reproduces the price, so nothing trades by
# default. An untrained baseline-init model disagrees with the price and trades.
# 'market' needs enough recorded quotes; until then fitting refuses clearly.
INIT_SCORE_COLUMNS = {'baseline': BASELINE_LOGIT, 'market': MARKET_LOGIT}


def attach_market_logit(table: pd.DataFrame,
                        column: str = 'market_probability') -> pd.DataFrame:
    """Attach `market_probability_logit` for a market-initialised model.

    Rows with no quote stay NaN. Falling back to the baseline would silently
    turn a market-init model into a baseline-init one.
    """
    out = table.copy()
    if column not in out.columns:
        raise ValueError(
            f'{column} is missing. A market-initialised model can only be scored '
            f'where a real quote was recorded; a backtest has no book, so this is '
            f'the expected failure there rather than something to work around.'
        )
    values = out[column].to_numpy(dtype=float)
    out[MARKET_LOGIT] = np.where(np.isfinite(values), logit(clip_prob(values)),
                                 np.nan)
    return out
# Fewest windows an inner validation block may hold and still be split in two.
# Below this, early stopping and the shrinkage fit share rows and alpha reads high.
MIN_INNER_BLOCK_WINDOWS = 200

# Fewest scoreable rows the shrinkage may be fitted on: enough that alpha is not
# noise, few enough that a data outage does not stop an evaluation.
MIN_SHRINKAGE_ROWS = 500

# Fraction of training windows held back, from the end of training, for early
# stopping and the shrinkage fit.
INNER_VALIDATION_FRACTION = 0.2

# Standardized stack-layer inputs are clipped here so one outlier cannot dominate.
LOGISTIC_CLIP = 5.0


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
    # The fitted volatility models and seasonality factors live scoring needs.
    # Without them the artifact can be evaluated but not deployed.
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
    # Which forecaster the correction sits on top of. Scoring a correction on
    # the other logit gives a well-formed probability to a different question.
    init_score_source: str = 'baseline'
    config_provenance: dict = field(default_factory=dict)
    # The final layer is p = expit(init_coefficient*init + residual_scale*corr
    # + intercept). A one-stage fit sets 1 and 0. Older artifacts lack the
    # fields, so readers use getattr.
    init_coefficient: float = 1.0
    intercept: float = 0.0
    # Every booster when ensembled (booster is members[0]); empty for one.
    members: tuple = ()

    @property
    def init_score_column(self) -> str:
        try:
            return INIT_SCORE_COLUMNS[self.init_score_source]
        except KeyError:                     # pragma: no cover - guarded on fit
            raise ValueError(
                f'unknown init_score_source {self.init_score_source!r}; '
                f'expected one of {sorted(INIT_SCORE_COLUMNS)}') from None

    # ---- prediction -----------------------------------------------------
    @property
    def boosters(self) -> list:
        return list(getattr(self, 'members', ()) or ()) or [self.booster]

    @property
    def ensemble_size(self) -> int:
        return len(self.boosters)

    def member_corrections(self, table: pd.DataFrame) -> np.ndarray:
        """Each member's raw logit correction, shape (members, rows)."""
        matrix = _feature_matrix(table, self.features)
        return np.vstack([np.asarray(b.predict(matrix, raw_score=True), dtype=float)
                          for b in self.boosters])

    def raw_correction(self, table: pd.DataFrame) -> np.ndarray:
        """The model's logit correction, before shrinkage (the member mean)."""
        if self.ensemble_size == 1:
            matrix = _feature_matrix(table, self.features)
            return np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float)
        return self.member_corrections(table).mean(axis=0)

    def _init_logit(self, table: pd.DataFrame) -> np.ndarray:
        column = self.init_score_column
        if column not in table.columns:
            raise ValueError(
                f'{column} is missing — score through core.dataset.apply_fold, '
                f'which attaches it from the fold\'s own baseline, or '
                f'attach_market_logit for a market-initialised model. Recomputing '
                f'it here would silently use a different one.'
            )
        return table[column].to_numpy(dtype=float)

    def _final(self, base: np.ndarray, correction: np.ndarray, *,
               shrink: bool = True) -> np.ndarray:
        if not shrink:
            return clip_prob(expit(base + correction))
        a = float(getattr(self, 'init_coefficient', 1.0))
        c = float(getattr(self, 'intercept', 0.0))
        return clip_prob(expit(a * base + self.residual_scale * correction + c))

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        """`model_probability`, plus the members' spread when ensembled.

        Member probabilities go through the same final layer as the mean, so
        `model_probability` always lies between `_min` and `_max`.
        """
        out = pd.DataFrame(index=table.index)
        out['model_probability'] = self.predict(table)
        if self.ensemble_size > 1:
            base = self._init_logit(table)
            members = np.vstack([self._final(base, c)
                                 for c in self.member_corrections(table)])
            members = np.where(np.isfinite(base), members, np.nan)
            with np.errstate(invalid='ignore'):
                out['model_probability_sd'] = members.std(axis=0)
                out['model_probability_min'] = members.min(axis=0)
                out['model_probability_max'] = members.max(axis=0)
        return out

    def predict(self, table: pd.DataFrame, *, shrink: bool = True) -> np.ndarray:
        """Calibrated probability that the window settles at or above its strike.

        A row whose init logit is NaN returns NaN, never the baseline: a
        market-init model with no quote must abstain. `decide()` abstains on a
        non-finite probability, and the `non_finite_share` gate counts it.
        """
        base = self._init_logit(table)
        out = self._final(base, self.raw_correction(table), shrink=shrink)
        # Redundant today, and kept: NaN already propagates through `expit` and
        # `np.clip`. This guarantees "no quote" never becomes a confident 0.5 if
        # `clip_prob` changes. Pinned by
        # `test_clip_prob_propagates_nan_which_this_relies_on`.
        return np.where(np.isfinite(base), out, np.nan)

    # ---- provenance -----------------------------------------------------
    def importance(self, kind: str = 'gain') -> pd.DataFrame:
        values = np.mean([np.asarray(b.feature_importance(importance_type=kind),
                                     dtype=float) for b in self.boosters], axis=0) \
            if self.ensemble_size > 1 else \
            self.booster.feature_importance(importance_type=kind)
        control = {c for g in CONTROL_GROUPS for c in feature_columns([g])}
        frame = pd.DataFrame({'feature': self.features, kind: values})
        frame['is_control'] = frame['feature'].isin(control)
        total = frame[kind].sum()
        frame['share'] = frame[kind] / total if total else np.nan
        return frame.sort_values(kind, ascending=False, ignore_index=True)

    @property
    def control_importance_share(self) -> float:
        """Share of total gain taken by the control group.

        The clock cannot forecast direction, so a large share means the
        measurement picks up something that is not a forecast.
        """
        frame = self.importance()
        return float(frame.loc[frame['is_control'], 'share'].sum())

    def provenance(self) -> dict:
        return {
            'features': list(self.features),
            'n_features': len(self.features),
            # `features` already lists only populated columns.
            'n_features_populated': len(self.features),
            'empty_features': list(self.empty_features),
            'groups': list(self.groups),
            'residual_scale': self.residual_scale,
            'init_coefficient': float(getattr(self, 'init_coefficient', 1.0)),
            'intercept': float(getattr(self, 'intercept', 0.0)),
            'ensemble_size': self.ensemble_size,
            # Which forecaster this corrects; the skill number depends on it.
            'init_score_source': self.init_score_source,
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
            f'({len(self.empty_features)} empty), {self.best_iteration} trees'
            f'{f" x{self.ensemble_size}" if self.ensemble_size > 1 else ""}, '
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

    def integrity(self) -> dict:
        """A snapshot of the artifact's feature shape, for inspection.

        Not the safety check: `verify()` raises on a mismatch. `load()` logs this
        at debug level, so a mismatch is visible even when no config was given.
        """
        names = list(self.booster.feature_name())
        return {
            'features': list(self.features),
            'booster_features': names,
            'n_features': self.booster.num_feature(),
        }

    def verify(self, config: Optional[Config] = None) -> None:
        """Raise unless this artifact can be scored as it stands.

        Called at load. Compares the booster's feature list, the init source, and
        the config fields that change an answer.
        """
        names = list(self.booster.feature_name())
        if names != list(self.features):
            raise ValueError(
                f'the booster was trained on {len(names)} columns and the artifact '
                f'lists {len(self.features)}; the first disagreement is at '
                f'{next((i for i, (a, b) in enumerate(zip(names, self.features)) if a != b), 0)}. '
                f'The feature matrix is built by name from the artifact\'s list, so '
                f'this would score a well-formed matrix of the wrong columns.'
            )
        if self.init_score_source not in INIT_SCORE_COLUMNS:
            raise ValueError(
                f'this artifact records init_score_source='
                f'{self.init_score_source!r}, which nothing can score')
        if config is None:
            return
        wanted = getattr(config, 'init_score_source', 'baseline')
        if wanted != self.init_score_source:
            raise ValueError(
                f'this artifact was fitted on the {self.init_score_source} logit '
                f'and the running configuration asks for {wanted}. The correction '
                f'is a residual of one specific forecaster, so adding it to the '
                f'other one produces a well-formed probability that answers a '
                f'different question — which is exactly how this would go '
                f'unnoticed.'
            )
        self._verify_config_drift(config)

    def _verify_config_drift(self, config: Config) -> None:
        stored = dict(self.config_provenance or {})
        if not stored:
            logger.warning('this artifact records no config provenance, so nothing '
                           'can be checked against the running configuration')
            return
        current = config.provenance()
        # Only the fields that change a number. Paths, versions and CLI notes
        # differ legitimately between the fitting run and the scoring run.
        material = ('window_minutes', 'decision_offsets', 'vol_lookbacks_minutes',
                    'embargo_minutes', 'fee_rate', 'maker_fee_multiplier', 'assume_maker',
                    'half_spread_cents', 'min_traded_price', 'max_traded_price')
        drift = {
            key: (stored.get(key), current.get(key))
            for key in material
            if key in stored and key in current and stored[key] != current[key]
        }
        if drift:
            detail = '; '.join(f'{k}: fitted {a!r}, running {b!r}'
                              for k, (a, b) in sorted(drift.items()))
            raise ValueError(
                f'this artifact was fitted under a different configuration — '
                f'{detail}. Scoring it here would answer a different question '
                f'than the one the gates were evaluated on.'
            )

    @staticmethod
    def load(path: str | Path, config: Optional[Config] = None) -> 'ForecastModel':
        import joblib
        model = joblib.load(Path(path))
        # Verified at load, not at first use, so a bad artifact fails where the
        # operator is looking.
        model.verify(config)
        logger.debug('artifact integrity: %s', model.integrity())
        return model


def _fit_residual_scale(
    baseline_logit: np.ndarray,
    correction: np.ndarray,
    outcome: np.ndarray,
    *,
    focus_quantile: float = 0.0,
    max_scale: float = 2.0,
) -> float:
    """One coefficient: how much of the claimed correction survives.

    Fitted by minimising held-out log loss over one scalar, the smallest
    recalibration there is. Clipped to [0, max_scale]: a negative alpha would
    invert an anti-predictive model, which is curve-fitting.

    `focus_quantile` = q fits only rows whose `|correction|` is in the top 1-q:
    a proxy for the rows that become trades, where the model is selected for
    disagreeing with the price. The ranking does not depend on alpha, so there
    is no fixed point. If too few rows remain (`MIN_SHRINKAGE_ROWS`), it falls
    back to all rows.
    """
    from scipy import optimize

    # Drop rows with no forecast (a NaN here makes the objective NaN at every
    # alpha). Require enough rows to remain, and raise if the fit does not
    # converge: an abandoned search point must never be reported as alpha.
    finite = (np.isfinite(baseline_logit) & np.isfinite(correction)
              & np.isfinite(np.asarray(outcome, dtype=float)))
    dropped = int((~finite).sum())
    if dropped:
        logger.info(
            '%d of %d shrinkage rows carry no forecast (a NaN sigma, usually the '
            'tail of a data outage) and are excluded', dropped, finite.size)
    baseline_logit = np.asarray(baseline_logit)[finite]
    correction = np.asarray(correction)[finite]
    outcome = np.asarray(outcome, dtype=float)[finite]
    if outcome.size < MIN_SHRINKAGE_ROWS:
        raise ValueError(
            f'only {outcome.size} scoreable rows remain of {finite.size}, which '
            f'is under the {MIN_SHRINKAGE_ROWS} needed to fit a shrinkage that '
            f'means anything. That is a data problem, not a model one.'
        )

    # Restrict to the likely-traded rows, if asked. After the minimum-rows check,
    # so a focus that leaves too little backs off to the full sample.
    if focus_quantile and 0.0 < focus_quantile < 1.0:
        cut = float(np.quantile(np.abs(correction), focus_quantile))
        keep = np.abs(correction) >= cut
        if int(keep.sum()) >= MIN_SHRINKAGE_ROWS:
            logger.info(
                'fitting the shrinkage on the %d row(s) above the %.0fth '
                'percentile of |correction| — the subset that becomes trades, '
                'where the model is selected for disagreeing with the price',
                int(keep.sum()), 100 * focus_quantile)
            baseline_logit = baseline_logit[keep]
            correction = correction[keep]
            outcome = outcome[keep]
        else:
            logger.warning(
                'the top %.0f%% of |correction| is only %d row(s), under the %d '
                'minimum, so the shrinkage is fitted on all %d instead',
                100 * (1 - focus_quantile), int(keep.sum()),
                MIN_SHRINKAGE_ROWS, outcome.size)

    def objective(alpha: float) -> float:
        return log_loss(outcome, expit(baseline_logit + float(alpha) * correction))

    # The ceiling, `max_residual_scale`, defines alpha: a correction may be
    # discounted but not inflated. Alpha above 1 means the validation split asks
    # to amplify the correction, which is an overfitting signature.
    result = optimize.minimize_scalar(objective, bounds=(0.0, max_scale),
                                      method='bounded')
    if not result.success or not np.isfinite(result.x):
        raise ValueError(
            f'the shrinkage fit did not converge ({getattr(result, "message", "")!r}). '
            f'Reporting its abandoned bracket point as a fitted alpha is how an '
            f'unfitted constant reaches a gate.'
        )
    return float(np.clip(result.x, 0.0, max_scale))


def _finite_log_loss(outcome: np.ndarray, base_logit: np.ndarray,
                     correction: np.ndarray, alpha: float) -> float:
    """Log loss over the rows that carry a forecast, so one NaN row cannot
    make the reported figure NaN."""
    outcome = np.asarray(outcome, dtype=float)
    keep = (np.isfinite(outcome) & np.isfinite(base_logit) & np.isfinite(correction))
    if not keep.any():
        return float('nan')
    return log_loss(outcome[keep],
                    expit(base_logit[keep] + alpha * correction[keep]))


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

    The inner validation split is the most recent windows, for early stopping
    and the shrinkage. A random split would put one window's offsets on both
    sides, and early stopping would measure memorisation.
    """
    import lightgbm as lgb

    columns = feature_columns(groups)
    source = getattr(config, 'init_score_source', 'baseline')
    if source not in INIT_SCORE_COLUMNS:
        raise ValueError(f'init_score_source={source!r}; expected one of '
                         f'{sorted(INIT_SCORE_COLUMNS)}')
    init_column = INIT_SCORE_COLUMNS[source]
    if init_column not in train.columns:
        extra = ''
        if source == 'market':
            # The likeliest cause: a backtest built from bars has no book, so a
            # market-init model can only fit on live-recorded quotes. Do not
            # attach the baseline column to make the error go away.
            extra = (' A backtest has no order book, so a market-initialised '
                     'model can only be fitted and scored on live-recorded '
                     'quotes. Substituting the baseline here would make "beat '
                     'the price" and "beat the baseline" the same question '
                     'answered twice with the same number.')
        raise ValueError(f'{init_column} is missing from the training table '
                         f'(init_score_source={source!r}).{extra}')
    if source == 'market':
        # A row with no quote has no market residual, so drop it. Its NaN init
        # score would reflect our recording uptime, not the market.
        # The baseline path drops nothing: its rare non-finite rows are real
        # data holes that `non_finite_share` gates.
        finite = np.isfinite(train[init_column].to_numpy(dtype=float))
        dropped = int((~finite).sum())
        if dropped:
            logger.warning(
                'dropping %d of %d training rows with no recorded quote: a '
                'market-initialised correction has nothing to correct there',
                dropped, len(train))
        train = train.loc[finite].copy()
        if train.empty:
            raise ValueError(
                f'every training row has a non-finite {init_column}, so there is '
                f'no market to fit a residual against')

    if getattr(config, 'label_source', 'proxy') == 'venue':
        # Everything downstream reads `outcome`, so the label is swapped once, here.
        proxy = train['outcome'].to_numpy(dtype=float)
        venue = (pd.to_numeric(train['venue_outcome'], errors='coerce').to_numpy(dtype=float)
                 if 'venue_outcome' in train.columns else np.full(len(train), np.nan))
        has_venue = np.isfinite(venue)
        logger.info(
            'label_source=venue: %d of %d training rows carry the venue label '
            '(%d disagree with the proxy); %d fall back to the proxy',
            int(has_venue.sum()), len(train),
            int((has_venue & np.isfinite(proxy) & (venue != proxy)).sum()),
            int((~has_venue).sum()))
        train = train.assign(outcome=np.where(has_venue, venue, proxy))

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

    # Purge the inner split like the outer one. `best_iteration` and
    # `residual_scale` are fitted here; a leak would ship a shrinkage that is
    # too weak and make its gate too easy.
    embargo = pd.Timedelta(minutes=int(config.embargo_minutes))
    inner_train = train.loc[train['window_open'] < cut - embargo]
    if inner_train.empty:
        # A short slice cannot afford the full embargo. Fall back to the
        # label's own horizon, which must never be skipped, and warn.
        fallback = pd.Timedelta(minutes=int(config.window_minutes))
        logger.warning(
            'the %d-minute inner embargo leaves no training rows in a slice of '
            '%d windows; falling back to %s. The shrinkage this fits will be '
            'optimistic.', config.embargo_minutes, len(windows), fallback)
        inner_train = train.loc[train['window_open'] < cut - fallback]

    # Early stopping and the shrinkage must not share rows: alpha measured on
    # the rows that chose the tree count reads high even on a null.
    # Split the validation block in two.
    valid_windows = windows[windows >= cut]
    holdout = train.loc[train['window_open'] >= cut]
    inner_valid, alpha_rows = holdout, holdout
    if len(valid_windows) >= 2 * MIN_INNER_BLOCK_WINDOWS:
        mid = valid_windows[len(valid_windows) // 2]
        stop_block = train.loc[(train['window_open'] >= cut)
                               & (train['window_open'] < mid - embargo)]
        alpha_block = train.loc[train['window_open'] >= mid]
        if not stop_block.empty and not alpha_block.empty:
            inner_valid, alpha_rows = stop_block, alpha_block
    if inner_valid is alpha_rows:
        # **Measured, not merely feared.** On the 2026-10-18 promotion this
        # fired on 3 of 28 folds and ALL THREE returned alpha at the 1.000 cap,
        # against 0.637 mean across the 27 folds whose blocks were separated.
        # So a shared block does not nudge alpha high, it pins it — and those
        # folds then inflate the `residual_scale` aggregate the gate reads.
        #
        # It does not explain everything: 10 properly-split folds also hit the
        # cap and 4 returned exactly 0.000, so alpha is genuinely unstable on
        # this data. That is a finding about the correction, not a bug here.
        logger.warning(
            'the inner validation block (%d windows) is too small to separate '
            'early stopping from the shrinkage fit; alpha will be measured on '
            'the rows the tree count was chosen for and will read high. '
            'Measured: every fold this fired on returned alpha at the cap.',
            len(valid_windows))

    if inner_valid.empty or inner_train.empty or alpha_rows.empty:
        raise ValueError('inner validation split is empty')

    def frame_weights(frame: pd.DataFrame) -> Optional[np.ndarray]:
        if weights is None:
            return None
        return np.asarray(weights, dtype=float)[frame.index.to_numpy()]

    fit_rows, fit_weights = inner_train, frame_weights(inner_train)

    def dataset(frame: pd.DataFrame, w: Optional[np.ndarray], reference=None):
        return lgb.Dataset(
            _feature_matrix(frame, populated),
            label=frame['outcome'].to_numpy(dtype=float),
            init_score=frame[init_column].to_numpy(dtype=float),
            weight=w, feature_name=list(populated), reference=reference,
            free_raw_data=False,
        )

    features = list(populated)
    members = []
    for k in range(max(int(getattr(config, 'ensemble_size', 1) or 1), 1)):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'learning_rate': config.learning_rate, 'num_leaves': config.num_leaves,
            'max_depth': config.max_depth,
            'min_child_samples': config.min_child_samples,
            'bagging_fraction': config.subsample, 'bagging_freq': 1,
            'feature_fraction': config.colsample_bytree,
            'lambda_l2': config.reg_lambda,
            'deterministic': True,
            'seed': int(getattr(config, 'model_seed', 17)) + k,
        }
        # Fresh datasets per member: a constructed Dataset pins seed-derived
        # parameters, and members must differ in their bagging draws.
        train_set = dataset(fit_rows, fit_weights)
        valid_set = dataset(inner_valid, frame_weights(inner_valid),
                            reference=train_set)
        members.append(lgb.train(
            params, train_set, num_boost_round=config.n_estimators,
            valid_sets=[valid_set], valid_names=['inner'],
            callbacks=[lgb.early_stopping(config.early_stopping_rounds,
                                          verbose=False)],
        ))
    booster = members[0]

    def mean_correction(frame: pd.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, features)
        return np.mean([np.asarray(m.predict(matrix, raw_score=True), dtype=float)
                        for m in members], axis=0)

    correction = mean_correction(alpha_rows)
    base_logit = alpha_rows[init_column].to_numpy(dtype=float)
    outcome = alpha_rows['outcome'].to_numpy(dtype=float)
    alpha = _fit_residual_scale(
        base_logit, correction, outcome,
        focus_quantile=float(getattr(config, 'shrinkage_focus_quantile', 0.0) or 0.0),
        max_scale=float(getattr(config, 'max_residual_scale', 2.0) or 2.0))

    model = ForecastModel(
        booster=booster, features=features, baseline=baseline,
        scoring=scoring, residual_scale=alpha,
        groups=tuple(groups) if groups else (),
        n_train_rows=len(train), n_train_windows=len(windows),
        empty_features=empty, best_iteration=booster.best_iteration,
        init_score_source=source,
        train_log_loss=_finite_log_loss(
            inner_train['outcome'].to_numpy(dtype=float),
            inner_train[init_column].to_numpy(dtype=float),
            mean_correction(inner_train), alpha),
        inner_log_loss=_finite_log_loss(outcome, base_logit, correction, alpha),
        inner_baseline_log_loss=_finite_log_loss(outcome, base_logit, correction, 0.0),
        config_provenance=config.provenance(),
        members=tuple(members) if len(members) > 1 else (),
    )
    logger.info(model.summary())
    return model


# ---- two-stage stacking -------------------------------------------------------
#
# Stage 1 learns spot -> probability on the long, clean bar history. Stage 2
# learns, on live-recorded quotes only and with two or three parameters, how
# much of stage 1's disagreement with the price survives.
STACK_INIT_BOUNDS = (0.5, 1.5)
STACK_INTERCEPT_BOUNDS = (-0.5, 0.5)


def stack_feature_groups(config: Config) -> tuple[str, ...]:
    """Feature groups a stacked model needs: stage 1's spot groups plus every
    book group holding a stage-2 column, so live fills those columns."""
    groups = list(getattr(config, 'stack_spot_groups', ()) or ())
    for column in getattr(config, 'stack_book_features', ()) or ():
        for name in BOOK_GROUPS:
            if column in FEATURE_GROUPS.get(name, ()) and name not in groups:
                groups.append(name)
    return tuple(groups)


def table_groups(config: Config, groups: Optional[Sequence[str]] = None
                 ) -> Optional[tuple[str, ...]]:
    """The groups a fold's feature table is built with under this config."""
    if getattr(config, 'stack_mode', 'off') != 'off':
        if groups:
            logger.warning('--groups %s is ignored in stack mode; stage 1 reads '
                           'stack_spot_groups %s', list(groups),
                           list(config.stack_spot_groups))
        return stack_feature_groups(config)
    return tuple(groups) if groups else None


def _standardize(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (np.asarray(matrix, dtype=float) - mean) / scale
    z = np.clip(z, -LOGISTIC_CLIP, LOGISTIC_CLIP)
    return np.where(np.isfinite(z), z, 0.0)


def _fit_stack(market_logit: np.ndarray, spot_logit: np.ndarray,
               book: np.ndarray, outcome: np.ndarray, *,
               weights: Optional[np.ndarray] = None, l2: float = 1.0,
               max_b: float = 1.0, intercept: bool = False
               ) -> tuple[float, float, float, np.ndarray]:
    """Fit logit p = a*m + b*(s - m) [+ c] [+ z @ beta] by penalised log loss.

    `book` is already standardized. Only `beta` is penalised: `a` and `b` are
    the quantities being measured, and their bounds constrain them.
    """
    from scipy import optimize

    m = np.asarray(market_logit, dtype=float)
    d = np.asarray(spot_logit, dtype=float) - m
    y = np.asarray(outcome, dtype=float)
    z = np.asarray(book, dtype=float).reshape(len(y), -1)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    n, k = len(y), z.shape[1]
    first = 3 if intercept else 2

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = theta[0], theta[1]
        c = theta[2] if intercept else 0.0
        beta = theta[first:]
        eta = a * m + b * d + c + z @ beta
        loss = float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))
        r = w * (expit(eta) - y)
        head = [float(r @ m), float(r @ d)] + ([float(r.sum())] if intercept else [])
        grad = np.concatenate([np.asarray(head), z.T @ r + l2 * beta / n])
        return loss + 0.5 * l2 * float(beta @ beta) / n, grad

    bounds = [STACK_INIT_BOUNDS, (0.0, float(max_b))]
    start = [1.0, 0.0]
    if intercept:
        bounds.append(STACK_INTERCEPT_BOUNDS)
        start.append(0.0)
    bounds += [(None, None)] * k
    start += [0.0] * k
    result = optimize.minimize(objective, np.asarray(start, dtype=float), jac=True,
                               method='L-BFGS-B', bounds=bounds)
    if not np.all(np.isfinite(result.x)):
        raise ValueError(f'the stage-2 fit did not converge ({result.message!r})')
    if not result.success:
        logger.warning('the stage-2 fit stopped early (%s); using its last point',
                       result.message)
    a, b = float(result.x[0]), float(result.x[1])
    c = float(result.x[2]) if intercept else 0.0
    return a, b, c, np.asarray(result.x[first:], dtype=float)


@dataclass
class StackedForecastModel(ForecastModel):
    """Stage 1 (`spot`, a baseline-init ForecastModel) under a stage-2 blend.

    `init_coefficient` is a, `residual_scale` is b and `intercept` is c, so the
    gates read b where they read alpha for a one-stage model. It reports
    `init_score_source='market'` because it needs the price to score, which
    makes the live path attach `market_probability_logit`.
    """

    spot: Any = None
    stack_features: tuple[str, ...] = ()
    stack_mean: Any = None
    stack_scale: Any = None
    stack_coef: Any = None
    stack_l2: float = 1.0
    n_stack_rows: int = 0
    stack_fitted: bool = False
    stack_inner_log_loss: float = float('nan')
    stack_inner_market_log_loss: float = float('nan')
    crossfit: tuple = ()

    # ---- prediction -----------------------------------------------------
    @property
    def boosters(self) -> list:
        return self.spot.boosters

    def _book_matrix(self, table: pd.DataFrame) -> np.ndarray:
        names = list(self.stack_features)
        if not names:
            return np.zeros((len(table), 0))
        raw = np.column_stack([
            pd.to_numeric(table[c], errors='coerce').to_numpy(dtype=float)
            if c in table.columns else np.full(len(table), np.nan)
            for c in names])
        return _standardize(raw, self.stack_mean, self.stack_scale)

    def _stage2(self, market_logit: np.ndarray, spot_probability: np.ndarray,
                book: np.ndarray) -> np.ndarray:
        market_logit = np.asarray(market_logit, dtype=float)
        spot_probability = np.asarray(spot_probability, dtype=float)
        with np.errstate(invalid='ignore'):
            eta = (self.init_coefficient * market_logit
                   + self.residual_scale * (logit(spot_probability) - market_logit)
                   + self.intercept
                   + book @ np.asarray(self.stack_coef, dtype=float))
        ok = np.isfinite(market_logit) & np.isfinite(spot_probability)
        return np.where(ok, clip_prob(expit(eta)), np.nan)

    def spot_probability(self, table: pd.DataFrame) -> np.ndarray:
        return self.spot.predict(table)

    def predict(self, table: pd.DataFrame, *, shrink: bool = True) -> np.ndarray:
        """NaN wherever there is no market logit: stage 2 is a statement about
        the price, and without a price `decide()` must abstain."""
        market = self._init_logit(table)
        if not shrink:
            return np.where(np.isfinite(market), self.spot_probability(table), np.nan)
        return self._stage2(market, self.spot_probability(table),
                            self._book_matrix(table))

    def raw_correction(self, table: pd.DataFrame) -> np.ndarray:
        """The stack's logit correction to the market."""
        return logit(self.predict(table)) - self._init_logit(table)

    def _member_probabilities(self, table: pd.DataFrame) -> list[np.ndarray]:
        market = self._init_logit(table)
        base = self.spot._init_logit(table)
        book = self._book_matrix(table)
        out = []
        for correction in self.spot.member_corrections(table):
            spot = np.where(np.isfinite(base), self.spot._final(base, correction),
                            np.nan)
            out.append(self._stage2(market, spot, book))
        return out

    def member_corrections(self, table: pd.DataFrame) -> np.ndarray:
        market = self._init_logit(table)
        return np.vstack([logit(p) - market
                          for p in self._member_probabilities(table)])

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=table.index)
        out['model_probability'] = self.predict(table)
        # Stage 1 alone, to show how far the spot model sat from the price.
        out['spot_probability'] = self.spot_probability(table)
        if self.ensemble_size > 1:
            members = np.vstack(self._member_probabilities(table))
            with np.errstate(invalid='ignore'):
                out['model_probability_sd'] = members.std(axis=0)
                out['model_probability_min'] = members.min(axis=0)
                out['model_probability_max'] = members.max(axis=0)
        return out

    # ---- provenance -----------------------------------------------------
    def importance(self, kind: str = 'gain') -> pd.DataFrame:
        return self.spot.importance(kind)

    def stack_summary(self) -> dict:
        return {
            'a': float(self.init_coefficient), 'b': float(self.residual_scale),
            'c': float(self.intercept),
            'book_features': list(self.stack_features),
            'book_coef': [float(v) for v in np.asarray(self.stack_coef, dtype=float)],
            'l2': float(self.stack_l2), 'n_rows': int(self.n_stack_rows),
            'fitted': bool(self.stack_fitted),
            'inner_log_loss': self.stack_inner_log_loss,
            'inner_market_log_loss': self.stack_inner_market_log_loss,
            'crossfit': list(self.crossfit),
        }

    def provenance(self) -> dict:
        out = super().provenance()
        out['stack'] = self.stack_summary()
        out['spot'] = self.spot.provenance()
        return out

    def summary(self) -> str:
        s = self.stack_summary()
        skill = s['inner_market_log_loss'] - s['inner_log_loss']
        book = (f', {len(s["book_features"])} book term(s)'
                if s['book_features'] else '')
        fitted = '' if s['fitted'] else ' (NOT fitted: too few rows)'
        return (f'stack: a={s["a"]:.3f} b={s["b"]:.3f} c={s["c"]:+.3f} on '
                f'{s["n_rows"]:,} quoted rows{fitted}{book} | out-of-fold skill '
                f'vs market {skill:+.5f}\n  stage 1 {self.spot.summary()}')

    def integrity(self) -> dict:
        out = self.spot.integrity()
        out['stack_features'] = list(self.stack_features)
        return out

    def verify(self, config: Optional[Config] = None) -> None:
        if self.spot is None:
            raise ValueError('this stacked artifact carries no stage-1 model')
        self.spot.verify(None)
        expected = list(self.spot.features) + [
            c for c in self.stack_features if c not in self.spot.features]
        if list(self.features) != expected:
            raise ValueError(
                f'the stacked artifact lists {len(self.features)} features but its '
                f'stages need {len(expected)}; it was assembled inconsistently')
        k = len(self.stack_features)
        for name in ('stack_mean', 'stack_scale', 'stack_coef'):
            if np.asarray(getattr(self, name), dtype=float).shape != (k,):
                raise ValueError(f'{name} does not match the {k} stage-2 book '
                                 f'feature(s)')
        if self.init_score_source != 'market':
            raise ValueError('a stacked artifact must be scored on the market logit')
        if config is None:
            return
        # The running config either asks for the market logit (live adopts it
        # from the artifact) or is itself a stack run.
        if (getattr(config, 'init_score_source', 'baseline') != 'market'
                and getattr(config, 'stack_mode', 'off') == 'off'):
            raise ValueError(
                'this stacked artifact needs the market logit, and the running '
                'configuration neither asks for it nor runs in stack mode')
        self._verify_config_drift(config)


def _training_label(train: pd.DataFrame, config: Config) -> np.ndarray:
    """The label stage 2 fits, swapped exactly as `fit_model` swaps stage 1's."""
    proxy = train['outcome'].to_numpy(dtype=float)
    if (getattr(config, 'label_source', 'proxy') != 'venue'
            or 'venue_outcome' not in train.columns):
        return proxy
    venue = pd.to_numeric(train['venue_outcome'], errors='coerce').to_numpy(dtype=float)
    return np.where(np.isfinite(venue), venue, proxy)


def crossfit_blocks(window_opens: pd.Series, mask: np.ndarray,
                    n_blocks: int) -> list[pd.DatetimeIndex]:
    """The stage-2 windows cut into at most `n_blocks` contiguous blocks."""
    opens = pd.DatetimeIndex(sorted(pd.unique(window_opens[mask])))
    if not len(opens):
        return []
    chunks = np.array_split(np.arange(len(opens)), min(int(n_blocks), len(opens)))
    return [opens[c] for c in chunks if len(c)]


def crossfit_spot(train: pd.DataFrame, stage2: np.ndarray, fitter, *,
                  n_blocks: int, embargo_minutes: int
                  ) -> tuple[np.ndarray, list[dict]]:
    """Out-of-sample stage-1 probabilities on the stage-2 rows.

    Each block is predicted by `fitter` trained on every other training row of
    the fold, minus the block widened by the embargo on both sides, as in the
    outer folds. NaN off `stage2` and on any block whose fit failed.
    """
    out = np.full(len(train), np.nan)
    records: list[dict] = []
    embargo = pd.Timedelta(minutes=int(embargo_minutes))
    opens = pd.to_datetime(train['window_open'], utc=True)
    for i, block in enumerate(crossfit_blocks(opens, stage2, n_blocks)):
        lo, hi = block.min() - embargo, block.max() + embargo
        fit_rows = train.loc[((opens < lo) | (opens > hi)).to_numpy()]
        target = stage2 & opens.isin(block).to_numpy()
        record = {'block': i, 'start': str(block.min()), 'end': str(block.max()),
                  'rows': int(target.sum()), 'fit_rows': int(len(fit_rows))}
        try:
            model = fitter(fit_rows)
            out[target] = np.asarray(model.predict(train.loc[target]), dtype=float)
            record['alpha'] = float(getattr(model, 'residual_scale', float('nan')))
        except ValueError as exc:
            logger.warning('stack cross-fit block %d (%s .. %s) could not be '
                           'fitted (%s); its %d row(s) leave stage 2', i,
                           block.min(), block.max(), str(exc)[:120],
                           int(target.sum()))
            record['error'] = str(exc)[:200]
        records.append(record)
    return out, records


def fit_stacked_model(
    train: pd.DataFrame,
    baseline: BarrierBaseline,
    config: Config = DEFAULT_CONFIG,
    *,
    weights: Optional[np.ndarray] = None,
    scoring: Any = None,
    _spot_fitter=None,
) -> StackedForecastModel:
    """Stage 1 on every training row, stage 2 on the quoted ones, out of fold.

    `_spot_fitter(rows) -> object with .predict(table)` replaces the stage-1
    fit inside the cross-fit only; tests use it to plant an overfit.
    """
    train = train.reset_index(drop=True)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(train):
            raise ValueError('weights do not align with the training table')
    if MARKET_LOGIT not in train.columns:
        if 'market_probability' not in train.columns:
            raise ValueError(
                'stack mode needs a market quote on at least some training rows; '
                'this table carries no market_probability at all')
        train = attach_market_logit(train)
    spot_groups = tuple(config.stack_spot_groups)
    # Stage 1 is an ordinary baseline-init fit; ensemble and label options
    # reach it unchanged through `fit_model`.
    spot_config = replace(config, init_score_source='baseline', stack_mode='off')

    def spot_fit(rows: pd.DataFrame, *, final: bool = False) -> ForecastModel:
        return fit_model(rows, baseline, spot_config, groups=spot_groups,
                         weights=weights, scoring=scoring if final else None)

    market = train[MARKET_LOGIT].to_numpy(dtype=float)
    label = _training_label(train, config)
    stage2 = (np.isfinite(market) & np.isfinite(label)
              & np.isfinite(train[BASELINE_LOGIT].to_numpy(dtype=float)))
    logger.info('stack: %d of %d training rows carry a usable quote for stage 2',
                int(stage2.sum()), len(train))

    oof, records = crossfit_spot(
        train, stage2, _spot_fitter or spot_fit,
        n_blocks=int(config.stack_crossfit_blocks),
        embargo_minutes=int(config.embargo_minutes))
    # Refit on every training row for test-time scoring.
    spot = spot_fit(train, final=True)

    book_names = tuple(getattr(config, 'stack_book_features', ()) or ())
    for name in book_names:
        if not any(name in FEATURE_GROUPS.get(g, ()) for g in BOOK_GROUPS):
            logger.warning('stack book feature %s is in no book feature group, so '
                           'the live path cannot be shown to supply it', name)
    raw_book = (np.column_stack([
        pd.to_numeric(train[c], errors='coerce').to_numpy(dtype=float)
        if c in train.columns else np.full(len(train), np.nan)
        for c in book_names]) if book_names else np.zeros((len(train), 0)))
    rows = stage2 & np.isfinite(oof)
    mean, scale = np.zeros(len(book_names)), np.ones(len(book_names))
    if book_names and rows.any():
        with warnings.catch_warnings():
            # An all-NaN column standardizes to zero and contributes nothing.
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nan_to_num(np.nanmean(raw_book[rows], axis=0))
            scale = np.nan_to_num(np.nanstd(raw_book[rows], axis=0))
        scale = np.where(scale > 0, scale, 1.0)
    z = _standardize(raw_book, mean, scale)

    a, b, c, coef = 1.0, 0.0, 0.0, np.zeros(len(book_names))
    fitted = int(rows.sum()) >= MIN_SHRINKAGE_ROWS
    if fitted:
        a, b, c, coef = _fit_stack(
            market[rows], logit(oof[rows]), z[rows], label[rows],
            weights=None if weights is None else weights[rows],
            l2=float(config.stack_l2),
            max_b=float(getattr(config, 'max_residual_scale', 1.0) or 1.0),
            intercept=bool(config.stack_intercept))
    else:
        # Too few rows: keep the prior that the price is right (a=1, b=0).
        logger.warning('stack: only %d stage-2 row(s) carry an out-of-fold spot '
                       'prediction, under the %d minimum; stage 2 stays at a=1, '
                       'b=0 (the market)', int(rows.sum()), MIN_SHRINKAGE_ROWS)

    features = list(spot.features) + [n for n in book_names if n not in spot.features]
    model = StackedForecastModel(
        booster=spot.booster, features=features, baseline=baseline,
        scoring=scoring, residual_scale=b, groups=stack_feature_groups(config),
        n_train_rows=spot.n_train_rows, n_train_windows=spot.n_train_windows,
        empty_features=spot.empty_features, best_iteration=spot.best_iteration,
        train_log_loss=spot.train_log_loss, inner_log_loss=spot.inner_log_loss,
        inner_baseline_log_loss=spot.inner_baseline_log_loss,
        init_score_source='market', config_provenance=config.provenance(),
        init_coefficient=a, intercept=c,
        spot=spot, stack_features=book_names, stack_mean=mean,
        stack_scale=scale, stack_coef=coef, stack_l2=float(config.stack_l2),
        n_stack_rows=int(rows.sum()), stack_fitted=fitted,
        crossfit=tuple(records),
    )
    if rows.any():
        p = model._stage2(market[rows], oof[rows], z[rows])
        model.stack_inner_log_loss = log_loss(label[rows], p)
        model.stack_inner_market_log_loss = log_loss(label[rows],
                                                     expit(market[rows]))
    logger.info(model.summary())
    return model


def fit_for_config(
    train: pd.DataFrame,
    baseline: BarrierBaseline,
    config: Config = DEFAULT_CONFIG,
    *,
    groups: Optional[Sequence[str]] = None,
    weights: Optional[np.ndarray] = None,
    scoring: Any = None,
) -> ForecastModel:
    """`fit_model`, or the two-stage stack when `config.stack_mode` asks."""
    if getattr(config, 'stack_mode', 'off') == 'spot_then_market':
        return fit_stacked_model(train, baseline, config, weights=weights,
                                 scoring=scoring)
    return fit_model(train, baseline, config, groups=groups, weights=weights,
                     scoring=scoring)
