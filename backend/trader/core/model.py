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
MARKET_LOGIT = 'market_probability_logit'

# Which forecaster the correction is fitted on top of. `Config.init_score_source`
# selects it; this maps the name to the column.
#
# **The baseline is the weaker of the two, measurably.** Over 1,109 live-recorded
# rows and 285 symbol-windows, log loss was 0.331 for the market's de-spread mid,
# 0.428 for `F(x/sigma)` and 0.430 for the model — the same sign on all three
# symbols and all four offsets, and on the 108 rows actually traded the model came
# in *worse* than its own baseline. So a model initialised on the baseline spends
# its capacity correcting a forecaster that is already 0.10 nats behind the price
# it has to trade against, and "beat the baseline" stops implying anything about
# whether the trade pays.
#
# Initialised on the market instead, three things change, and the second is the
# reason to want it:
#
# * The residual being fitted is `logit(truth) - logit(price)` — how the price is
#   wrong, which is exactly the quantity the money depends on.
# * **The null inverts, in the right direction.** An untrained baseline-init model
#   reproduces `F(x/sigma)`, which disagrees with the price by 5.79pp on average —
#   a large apparent edge that is mostly noise, and it is *live by default*. An
#   untrained market-init model reproduces the price, so the edge is identically
#   zero and nothing trades. The default state becomes "the price is right",
#   which is the honest prior.
# * `decide()`'s edge becomes the model's own output rather than a difference
#   between two independently-fitted things.
#
# It is not trainable yet, and the gap is not close: 285 symbol-windows of
# recorded quotes exist against a `windows_evaluated >= 20,000` gate — roughly
# seventy days of recording at ~285 a day. The mechanism is here, refusing
# clearly, so that the day the data exists this is a config change and not a
# rewrite.
INIT_SCORE_COLUMNS = {'baseline': BASELINE_LOGIT, 'market': MARKET_LOGIT}


def attach_market_logit(table: pd.DataFrame,
                        column: str = 'market_probability') -> pd.DataFrame:
    """Attach `market_probability_logit` for a market-initialised model.

    Rows with no quote stay NaN rather than borrowing the baseline. A market-init
    model that silently fell back to the baseline on a missing quote would be a
    baseline-init model wearing the other one's provenance, which is the precise
    failure this whole arrangement exists to avoid — and it would be invisible,
    because the two produce identically well-formed numbers.
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
# Fraction of the training windows held back to fit `residual_scale` and to
# early-stop on. Taken from the *end* of training, so it is the most recent
# data and the shrinkage is measured on the regime nearest the test block.
# Fewest windows an inner block may hold and still be split in two. Below
# this, early stopping and the shrinkage fit share rows and alpha reads high.
MIN_INNER_BLOCK_WINDOWS = 200

# Fewest scoreable rows the shrinkage may be fitted on. Small enough that an
# outage does not stop an evaluation, large enough that alpha is not noise.
MIN_SHRINKAGE_ROWS = 500

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
    # Which forecaster the correction sits on top of. Stored, because scoring a
    # market-init correction on a baseline logit (or the reverse) produces a
    # perfectly well-formed probability that answers a different question.
    init_score_source: str = 'baseline'
    config_provenance: dict = field(default_factory=dict)

    @property
    def init_score_column(self) -> str:
        try:
            return INIT_SCORE_COLUMNS[self.init_score_source]
        except KeyError:                     # pragma: no cover - guarded on fit
            raise ValueError(
                f'unknown init_score_source {self.init_score_source!r}; '
                f'expected one of {sorted(INIT_SCORE_COLUMNS)}') from None

    # ---- prediction -----------------------------------------------------
    def raw_correction(self, table: pd.DataFrame) -> np.ndarray:
        """The model's logit correction, before shrinkage."""
        matrix = _feature_matrix(table, self.features)
        return np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float)

    def predict(self, table: pd.DataFrame, *, shrink: bool = True) -> np.ndarray:
        """Calibrated probability that the window settles above its strike.

        A row whose init column is NaN comes back NaN. That is a market-init model
        meeting a window with no quote, and it must not fall through to the
        baseline: `decide()` abstains on a non-finite probability, and the
        `non_finite_share` gate counts them, so the abstention is both safe and
        visible. Substituting the baseline would be silent and would put the
        wrong provenance on a real trade.
        """
        column = self.init_score_column
        if column not in table.columns:
            raise ValueError(
                f'{column} is missing — score through core.dataset.apply_fold, '
                f'which attaches it from the fold\'s own baseline, or '
                f'attach_market_logit for a market-initialised model. Recomputing '
                f'it here would silently use a different one.'
            )
        base = table[column].to_numpy(dtype=float)
        alpha = self.residual_scale if shrink else 1.0
        out = clip_prob(expit(base + alpha * self.raw_correction(table)))
        # Redundant today and kept deliberately. NaN already propagates through
        # `expit` and `np.clip`, so mutating this line away changes nothing and
        # mutation testing cannot tell the difference — which is exactly why the
        # guarantee is written down here and pinned by
        # `test_clip_prob_propagates_nan_which_this_relies_on`. A `clip_prob` that
        # one day filled NaN with 0.5 would turn "no quote" into "a coin flip,
        # confidently asserted" on a real-money path, silently.
        return np.where(np.isfinite(base), out, np.nan)

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
            # Which forecaster this is a correction to. A promoted artifact that
            # does not say is an artifact whose skill number means one of two
            # different things.
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

    def integrity(self) -> dict:
        """A snapshot of the artifact's feature shape, for a human or a script.

        **This is not the safety check** — `verify()` below is. `verify()` is
        what `load()` calls and what actually raises on a mismatched feature
        list; this docstring used to describe that same failure mode as if
        `integrity()` were the thing guarding against it, while nothing called
        it and it never raised anything. It is introspection: `load()` logs it
        at debug level so a mismatch that `verify()` allowed through (no config
        passed, so no comparison to make) is still visible in the log rather
        than only inferable from wrong numbers downstream.
        """
        names = list(self.booster.feature_name())
        return {
            'features': list(self.features),
            'booster_features': names,
            'n_features': self.booster.num_feature(),
        }

    def verify(self, config: Optional[Config] = None) -> None:
        """Raise unless this artifact can be scored as it stands.

        Called at load. The fields compared against `config` are the ones that
        change an answer rather than a path: a model fitted at offsets (3,6,9,12)
        scored at (1,2) is being asked a question it was never fitted for, and
        `core/dataset.py` only warned. `scripts/live.py` did not read
        `config_provenance` at all.
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
        stored = dict(self.config_provenance or {})
        if not stored:
            logger.warning('this artifact records no config provenance, so nothing '
                           'can be checked against the running configuration')
            return
        current = config.provenance()
        # Only the fields that change a number. Paths, versions and CLI notes
        # differ legitimately between the run that fitted and the run that scores.
        material = ('window_minutes', 'decision_offsets', 'vol_lookbacks_minutes',
                    'embargo_minutes', 'fee_rate', 'maker_fee_rate', 'assume_maker',
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
        # Verified at load, not at first use. An artifact that cannot be scored
        # should fail where the operator is looking, not several minutes into a
        # cycle with a quote in hand.
        model.verify(config)
        logger.debug('artifact integrity: %s', model.integrity())
        return model


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

    # Exclude rows that carry no forecast, rather than optimising over them.
    #
    # A single NaN makes the objective NaN at every alpha, `minimize_scalar` gives
    # up, and it returns its golden-section bracket seed 0.7639320225 with
    # `success=False` — which was never checked, so the *overfitting detector*
    # returned a search constant that sails past its own `residual_scale >= 0.25`
    # gate. On a five-year BTC walk-forward four of six folds returned exactly
    # that number, and those four folds carried the reported skill.
    #
    # Refusing outright was the first fix here and it was wrong: a 6.5-hour
    # Coinbase outage leaves ~83 rows in 26,488 without a volatility estimate, and
    # that killed the whole evaluation. The defect was never the NaN, it was
    # reporting an abandoned search as a fitted value. So drop the unscoreable
    # rows, insist enough remain to mean anything, and still raise if the fit
    # itself does not converge.
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


def _finite_log_loss(outcome: np.ndarray, base_logit: np.ndarray,
                     correction: np.ndarray, alpha: float) -> float:
    """Log loss over the rows that carry a forecast.

    These three numbers go into the artifact's provenance and its `summary()`
    line, and they were computed over unfiltered arrays — so a single row with a
    NaN sigma printed `inner log loss nan vs baseline nan (skill +nan)` for the
    whole model. Observed on real bars, where one venue outage is enough.
    """
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

    The inner validation split is taken from the *end* of training — the most
    recent windows — and is used for both early stopping and the shrinkage
    coefficient. Splitting at random would put four offsets of the same window
    on both sides and the early-stopping signal would be measuring memorisation.
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
            # The likeliest way to arrive here, and it is not a bug to route
            # around: `walk_forward` builds its tables from bars and has no book,
            # so a market-initialised model cannot be backtested at all. Saying
            # that here is the difference between a clear refusal and someone
            # attaching the baseline column to make the error go away.
            extra = (' A backtest has no order book, so a market-initialised '
                     'model can only be fitted and scored on live-recorded '
                     'quotes. Substituting the baseline here would make "beat '
                     'the price" and "beat the baseline" the same question '
                     'answered twice with the same number.')
        raise ValueError(f'{init_column} is missing from the training table '
                         f'(init_score_source={source!r}).{extra}')
    if source == 'market':
        # Only on this path, and it is not a tolerance dial. A row with no quote
        # cannot carry a market residual at all — there is nothing to be a
        # residual of — so it is dropped rather than tolerated. Keeping it would
        # hand LightGBM an init score that is NaN precisely where our recording
        # failed, and "did we have a quote" is a property of our uptime, not of
        # the market.
        #
        # A blanket coverage floor across both sources was tried first and was
        # wrong: the baseline logit is legitimately non-finite on a small share of
        # rows (a venue outage leaves a real hole), test fixtures sit at 98.5%,
        # and `non_finite_share` already gates that end. Two earlier fixes in this
        # repo failed the same way — a `non_finite_rows == 0` gate and a shrinkage
        # guard that raised on any NaN — so the baseline path is left exactly as
        # it was.
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

    # Purge the inner split the same way the outer one is purged.
    #
    # This boundary used to be hard: the last inner-training window was fifteen
    # minutes from the first inner-validation window, against the 1,440 the outer
    # CV insists on for exactly the same stated reason. Two things are fitted
    # here — LightGBM's `best_iteration` and `residual_scale` — and
    # `residual_scale` is the single number guarding against overconfidence, with
    # its own gate. Leaking into it does not inflate the reported out-of-sample
    # skill, which is measured on the properly embargoed outer fold; it ships a
    # model whose shrinkage is too weak and makes its gate easier to pass than
    # intended.
    embargo = pd.Timedelta(minutes=int(config.embargo_minutes))
    inner_train = train.loc[train['window_open'] < cut - embargo]
    if inner_train.empty:
        # A short training slice cannot afford the full embargo. Say so rather
        # than silently dropping it — the window is still purged by the label's
        # own horizon, which is the part that must not be skipped.
        fallback = pd.Timedelta(minutes=int(config.window_minutes))
        logger.warning(
            'the %d-minute inner embargo leaves no training rows in a slice of '
            '%d windows; falling back to %s. The shrinkage this fits will be '
            'optimistic.', config.embargo_minutes, len(windows), fallback)
        inner_train = train.loc[train['window_open'] < cut - fallback]

    # Early stopping and the shrinkage must not share rows. `residual_scale`
    # answers "how much of the claimed correction survives out of sample", and
    # measured on the same rows that chose the tree count it answers "how much
    # survives on the rows the tree count was selected for" — it read 0.902 on a
    # provably zero-signal null. Split the validation block in two.
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
        logger.warning(
            'the inner validation block (%d windows) is too small to separate '
            'early stopping from the shrinkage fit; alpha will be measured on '
            'the rows the tree count was chosen for and will read high.',
            len(valid_windows))

    if inner_valid.empty or inner_train.empty or alpha_rows.empty:
        raise ValueError('inner validation split is empty')

    def dataset(frame: pd.DataFrame, reference=None):
        w = None
        if weights is not None:
            w = np.asarray(weights, dtype=float)[frame.index.to_numpy()]
        return lgb.Dataset(
            _feature_matrix(frame, populated),
            label=frame['outcome'].to_numpy(dtype=float),
            init_score=frame[init_column].to_numpy(dtype=float),
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

    alpha_matrix = _feature_matrix(alpha_rows, populated)
    correction = np.asarray(booster.predict(alpha_matrix, raw_score=True), dtype=float)
    base_logit = alpha_rows[init_column].to_numpy(dtype=float)
    outcome = alpha_rows['outcome'].to_numpy(dtype=float)
    alpha = _fit_residual_scale(base_logit, correction, outcome)

    model = ForecastModel(
        booster=booster, features=list(populated), baseline=baseline,
        scoring=scoring, residual_scale=alpha,
        groups=tuple(groups) if groups else (),
        n_train_rows=len(train), n_train_windows=len(windows),
        empty_features=empty, best_iteration=booster.best_iteration,
        init_score_source=source,
        train_log_loss=_finite_log_loss(
            inner_train['outcome'].to_numpy(dtype=float),
            inner_train[init_column].to_numpy(dtype=float),
            np.asarray(booster.predict(_feature_matrix(inner_train, populated),
                                       raw_score=True), dtype=float),
            alpha),
        inner_log_loss=_finite_log_loss(outcome, base_logit, correction, alpha),
        inner_baseline_log_loss=_finite_log_loss(outcome, base_logit, correction, 0.0),
        config_provenance=config.provenance(),
    )
    logger.info(model.summary())
    return model
