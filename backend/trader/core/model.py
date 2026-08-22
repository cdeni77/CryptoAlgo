"""The forecast model: expected net return, decomposed, with a risk estimate.

Three heads, because they are three different problems:

    price       expected price return over the horizon. Barely predictable at an
                hourly frequency, and honest R-squared here is a few thousandths.
    carry       expected funding accrual. Published and strongly persistent, so
                genuinely predictable — often the only head that works.
    dispersion  expected absolute error of the price forecast. Not a return
                prediction at all; it is what turns a forecast into a position
                size.

Keeping them apart is the point. A single net-return head would report one
number and hide whether the edge came from carry (plausible, mechanical) or from
directional prediction (hard, and usually noise). When the price head is worthless
and the carry head is not, that is a working carry harvester and should be
recognised as one rather than averaged into mush.

Everything is pooled across instruments with the symbol as a categorical
feature. A 120-day hourly window at a 96-hour horizon carries roughly 30
independent observations per instrument (`core.cv.effective_sample_size`); against
77 features no per-coin fit is identifiable, and cross-sectional standardisation
in `core.features` is what makes rows from different instruments comparable
enough to pool.

Two invariants are enforced rather than documented: folds split on time so no
timestamp straddles train and test, and scalers are fitted inside each fold.

Regression changes how this is scored. AUC does not apply. What matters is rank
information coefficient — the Spearman correlation between forecast and outcome —
and after that, the economics: does acting on the forecast make money net of
cost. IC near 0.03 is a real signal at this frequency; R-squared will look like
nothing and that is expected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

from core.config import Config
from core.cv import CVFold, FoldPreprocessor, effective_sample_size, purged_walk_forward, sample_weights
from core.metrics import PathDistribution, summarise_paths
from core.profiles import CoinProfile

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
SYMBOL_COLUMN = '__symbol'
ARTIFACT_VERSION = 'forecast-v1'

# Fewer rows than this and a fold is describing noise; both the fit and its
# score would be meaningless.
MIN_FOLD_ROWS = 200

# Whether to give the model the instrument's identity as a feature.
#
# Off by default, and the reason is measured rather than theoretical: a forecast
# that knows only each symbol's in-sample average forward return — no timing
# skill at all — scores a rank IC of +0.23 on a five-instrument panel of pure
# random walks. Identity lets a tree memorise "this instrument drifted up in the
# training sample", which is indistinguishable from skill in-sample and worthless
# out of it. Legitimate per-instrument differences reach the model through actual
# characteristics: cost, volatility, liquidity, carry level.
#
# `symbol_identity_ic` in the CV report measures exactly how much IC identity
# alone buys on a given panel. Any model's IC is only interesting above it.
USE_SYMBOL_IDENTITY = False

# Price IC this close to the hindsight identity ceiling suggests the model is
# ranking by instrument level rather than by timing.
MEMORISATION_WARNING_RATIO = 0.7

# E|e| for a Gaussian is sigma * sqrt(2/pi). The dispersion head predicts mean
# absolute error, so this converts it to a standard deviation for sizing.
MAE_TO_SIGMA = float(np.sqrt(np.pi / 2.0))

HEADS = ('price', 'carry', 'dispersion')


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadSpec:
    """Hyperparameters for one head.

    Shallow by design. With a few hundred independent observations, depth is the
    fastest available route to fitting noise.
    """

    objective: str = 'regression'
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.03
    min_child_samples: int = 60
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    reg_lambda: float = 5.0
    seed: int = 7

    def to_params(self) -> dict[str, Any]:
        return {
            'objective': self.objective,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'subsample_freq': 1,
            'colsample_bytree': self.colsample_bytree,
            'reg_lambda': self.reg_lambda,
            'random_state': self.seed,
            'n_jobs': 1,
            'verbose': -1,
        }


def default_head_specs() -> dict[str, HeadSpec]:
    """Per-head defaults.

    The price head is regularised hardest because it is the one most likely to
    be fitting noise. Carry gets more capacity because there is real structure
    to find. Dispersion uses an L1 objective, since it is predicting a magnitude
    and squared error would let a few large moves dominate.
    """
    return {
        'price': HeadSpec(reg_lambda=10.0, colsample_bytree=0.5, seed=7),
        'carry': HeadSpec(n_estimators=400, max_depth=5, reg_lambda=1.0, seed=17),
        'dispersion': HeadSpec(objective='regression_l1', reg_lambda=5.0, seed=27),
    }


# ---------------------------------------------------------------------------
# Panel plumbing
# ---------------------------------------------------------------------------


def align_panel(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows present and resolved in both frames."""
    if features.empty or targets.empty:
        return features.iloc[:0], targets.iloc[:0]

    resolved = targets.dropna(subset=['price'])
    shared = features.index.intersection(resolved.index)
    if shared.empty:
        return features.iloc[:0], targets.iloc[:0]

    x = features.loc[shared].replace([np.inf, -np.inf], np.nan)
    usable = x.notna().any(axis=1)
    return x.loc[usable], resolved.loc[shared].loc[usable]


def add_symbol_feature(
    features: pd.DataFrame,
    categories: Optional[Sequence[str]] = None,
    *,
    enabled: bool = USE_SYMBOL_IDENTITY,
) -> pd.DataFrame:
    """Attach the instrument as a categorical column, if identity is enabled.

    See `USE_SYMBOL_IDENTITY`. When disabled this is a no-op, and the model sees
    instruments only through their measured characteristics.
    """
    if not enabled:
        return features.copy()
    out = features.copy()
    symbols = out.index.get_level_values('symbol')
    out[SYMBOL_COLUMN] = pd.Categorical(
        symbols, categories=list(categories) if categories is not None else None
    )
    return out


def categorical_features(columns: Sequence[str]) -> list[str]:
    """The categorical column list LightGBM should be told about."""
    return [SYMBOL_COLUMN] if SYMBOL_COLUMN in columns else []


def panel_sample_weights(
    index: pd.MultiIndex,
    *,
    horizon_bars: int,
    half_life_days: float = 0.0,
) -> np.ndarray:
    """Uniqueness weights per instrument, reassembled.

    Overlap is a property of one instrument's own label windows. Computing it
    across the pooled index would treat two instruments' simultaneous outcomes
    as overlapping, when they are the cross-section this model exists to use.
    """
    weights = pd.Series(0.0, index=index)
    for symbol in index.get_level_values('symbol').unique():
        mask = index.get_level_values('symbol') == symbol
        times = pd.DatetimeIndex(index[mask].get_level_values('event_time'))
        weights.loc[mask] = sample_weights(
            times, horizon_bars=horizon_bars, half_life_days=half_life_days
        )
    mean = weights.mean()
    return (weights / mean).to_numpy() if mean > 0 else np.ones(len(index))


def time_folds(panel_index: pd.MultiIndex, folds: Sequence[CVFold]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Translate time-indexed folds into panel row positions.

    Every instrument at a timestamp lands on the same side. Splitting rows
    instead would put SOL at 14:00 in train and BTC at 14:00 in test, and those
    two rows share the market move that decides both outcomes.
    """
    times = pd.DatetimeIndex(panel_index.get_level_values('event_time'))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        train = np.flatnonzero(times.isin(fold.train_idx))
        test = np.flatnonzero(times.isin(fold.test_idx))
        if train.size >= MIN_FOLD_ROWS and test.size > 0:
            out.append((train, test))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def information_coefficient(prediction: np.ndarray, outcome: np.ndarray) -> float:
    """Spearman rank correlation between forecast and realisation.

    The standard measure for a return forecast, and robust to the fat tails that
    make R-squared uninformative here. At an hourly horizon, 0.02-0.05 is a real
    signal; 0.2 means something is leaking.
    """
    p = np.asarray(prediction, dtype=float)
    o = np.asarray(outcome, dtype=float)
    finite = np.isfinite(p) & np.isfinite(o)
    if finite.sum() < 30 or np.unique(p[finite]).size < 2:
        return float('nan')
    return float(spearmanr(p[finite], o[finite]).statistic)


def cross_sectional_ic(
    prediction: np.ndarray,
    outcome: np.ndarray,
    index: pd.MultiIndex,
    *,
    min_universe: int = 3,
) -> float:
    """Mean per-timestamp rank correlation across instruments.

    The standard cross-sectional IC: rank instruments at each bar, correlate with
    what they went on to do, average over bars. It answers "can this pick the
    best instrument right now", which is the question a cross-sectional strategy
    actually asks. Pooled Spearman instead mixes that with "can it pick the good
    hours", and is dominated by whichever varies more.
    """
    frame = pd.DataFrame({'p': prediction, 'a': outcome}, index=index)
    scores: list[float] = []
    for _, group in frame.groupby(level='event_time'):
        if len(group) < min_universe or group['p'].nunique() < 2:
            continue
        statistic = spearmanr(group['p'], group['a']).statistic
        if np.isfinite(statistic):
            scores.append(float(statistic))
    return float(np.mean(scores)) if scores else float('nan')


def identity_ceiling_ic(outcome: pd.Series) -> float:
    """How much rank information is just cross-instrument level differences.

    Computed by ranking every row by its own instrument's mean outcome *in this
    same window*. That uses the outcomes it is being scored against, so it is a
    hindsight ceiling rather than a benchmark a model competes with — it cannot
    be beaten fairly and a model is not "behind" for scoring below it.

    What it is for: sizing the memorisation hazard. On a five-instrument panel of
    random walks it reaches +0.54, because realised drift differs by sample and
    ranking by it sorts the outcomes well. A model whose IC approaches that
    number is probably exploiting instrument level rather than timing, and
    `MEMORISATION_WARNING_RATIO` flags it.
    """
    if outcome.empty:
        return float('nan')
    means = outcome.groupby(level='symbol').transform('mean')
    return information_coefficient(means.to_numpy(), outcome.to_numpy())


def out_of_sample_r2(prediction: np.ndarray, outcome: np.ndarray) -> float:
    """R-squared against a zero forecast, not against the outcome's own mean.

    Predicting the in-sample mean return is not a benchmark a trader can use;
    zero is. This is the number that will look like nothing, and should.
    """
    p = np.asarray(prediction, dtype=float)
    o = np.asarray(outcome, dtype=float)
    finite = np.isfinite(p) & np.isfinite(o)
    if finite.sum() < 30:
        return float('nan')
    residual = np.sum((o[finite] - p[finite]) ** 2)
    total = np.sum(o[finite] ** 2)
    return float(1.0 - residual / total) if total > 0 else float('nan')


# ---------------------------------------------------------------------------
# Trained artifact
# ---------------------------------------------------------------------------


@dataclass
class ForecastModel:
    """Trained heads plus the provenance needed to trust them."""

    heads: dict[str, Any]
    feature_columns: tuple[str, ...]
    symbol_categories: tuple[str, ...]
    feature_set_hash: str
    horizon_bars: int
    cost_config_version: str = 'unknown'
    trained_at: str = ''
    data_as_of: Optional[str] = None
    train_rows: int = 0
    effective_observations: float = 0.0
    # False when the train/validation split could not be purged for want of
    # history, which makes the recorded ic/r2 optimistic. Reported rather
    # than assumed, because the fallback fires exactly when data is scarce.
    validation_purged: bool = True
    # Columns the panel declared but never populated. `build_panel` reindexes to
    # the canonical column list so a saved model always scores against the same
    # matrix, which means a feature group that produced nothing arrives as an
    # all-NaN column rather than an absent one. `feature_set_hash` is over column
    # *names*, so without this a model fit with the cross-venue group and one fit
    # without it are indistinguishable — the likely case for a US operator, whose
    # reference venue answers 451.
    empty_features: tuple[str, ...] = ()
    # Symbols whose funding came from a venue other than the traded one, from
    # `Dataset.proxy_funding_symbols`. Funding feeds the `carry` component of the
    # target, so this is the difference between a carry edge measured on the cash
    # flow this account receives and one measured on somebody else's.
    proxy_funding_symbols: tuple[str, ...] = ()
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_version: str = ARTIFACT_VERSION

    # -- design matrix ------------------------------------------------------

    @property
    def uses_symbol_identity(self) -> bool:
        return SYMBOL_COLUMN in self.feature_columns

    def _design(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reindex to the trained column order, restoring the symbol category.

        Reindexing rather than trusting the caller: a column-order mismatch
        between training and scoring is silent and produces confident garbage.
        """
        x = features.drop(columns=[SYMBOL_COLUMN], errors='ignore')
        x = add_symbol_feature(
            x, categories=self.symbol_categories, enabled=self.uses_symbol_identity
        )
        return x.reindex(columns=list(self.feature_columns))

    # -- forecasts ----------------------------------------------------------

    def predict_component(self, features: pd.DataFrame, head: str) -> np.ndarray:
        model = self.heads.get(head)
        if model is None:
            return np.zeros(len(features))
        return model.predict(self._design(features))

    def predict(self, features: pd.DataFrame, cost: np.ndarray | float) -> pd.DataFrame:
        """Expected net return per side, plus the risk estimate and sizing ratio.

        `cost` is known at decision time and passed in rather than predicted.
        Both sides pay it, which is why `net_long + net_short == -2 * cost` and
        at most one side can be worth taking.
        """
        price = self.predict_component(features, 'price')
        carry = self.predict_component(features, 'carry')
        cost_array = np.broadcast_to(np.asarray(cost, dtype=float), price.shape)

        # The dispersion head predicts mean absolute error; convert to a sigma.
        sigma = np.maximum(self.predict_component(features, 'dispersion'), 0.0) * MAE_TO_SIGMA

        net_long = price + carry - cost_array
        net_short = -price - carry - cost_array
        take_long = net_long >= net_short
        best_net = np.where(take_long, net_long, net_short)
        side = np.where(take_long, 1.0, -1.0)
        side = np.where(best_net > 0, side, 0.0)

        out = pd.DataFrame(
            {
                'price': price,
                'carry': carry,
                'cost': cost_array,
                'sigma': sigma,
                'net_long': net_long,
                'net_short': net_short,
                'side': side,
                'expected_net': np.where(side != 0, best_net, 0.0),
            },
            index=features.index,
        )
        # Forecast per unit of risk: what sizing should scale with. Guarded
        # because a zero sigma would otherwise imply infinite conviction.
        out['edge_to_risk'] = np.where(
            out['sigma'] > 1e-9, out['expected_net'] / out['sigma'], 0.0
        )
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write atomically, so a crashed save cannot leave half a model live."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.tmp')
        joblib.dump(self, temporary)
        os.replace(temporary, target)
        logger.info('saved forecast model %s (%s)', target, self.feature_set_hash)
        return target

    @staticmethod
    def load(path: str | Path) -> 'ForecastModel':
        model = joblib.load(Path(path))
        if not isinstance(model, ForecastModel):
            raise TypeError(f'{path} does not contain a ForecastModel')
        return model

    def provenance(self) -> dict[str, Any]:
        """What this model was trained on. Belongs in every report."""
        return {
            'artifact_version': self.artifact_version,
            'feature_set_hash': self.feature_set_hash,
            'n_features': len(self.feature_columns),
            'heads': sorted(self.heads),
            'uses_symbol_identity': self.uses_symbol_identity,
            'horizon_bars': self.horizon_bars,
            'cost_config_version': self.cost_config_version,
            'trained_at': self.trained_at,
            'data_as_of': self.data_as_of,
            'train_rows': self.train_rows,
            'effective_observations': round(self.effective_observations, 1),
            'validation_purged': self.validation_purged,
            'n_features_populated': len(self.feature_columns) - len(self.empty_features),
            'empty_features': list(self.empty_features),
            'proxy_funding_symbols': list(self.proxy_funding_symbols),
            'train_start': str(self.train_start) if self.train_start is not None else None,
            'train_end': str(self.train_end) if self.train_end is not None else None,
            'symbols': list(self.symbol_categories),
        }

    def assert_compatible(self, features: pd.DataFrame) -> None:
        """Refuse to score a feature matrix this model never saw."""
        expected = set(self.feature_columns) - {SYMBOL_COLUMN}
        missing = sorted(expected - set(features.columns))
        if missing:
            raise ValueError(
                f'model {self.feature_set_hash} needs {len(missing)} absent features: '
                f'{missing[:6]}{"..." if len(missing) > 6 else ""}'
            )

    def in_sample_rows(self, features: pd.DataFrame) -> int:
        """How many of these rows fall inside the training window.

        Non-zero means any backtest over them is scoring the model's own memory.
        Measured, not assumed: trading in-sample forecasts on driftless random
        walks produced a mean price PnL of +95,000 with a t-statistic of +7
        across six independent seeds. There is no edge in a driftless random
        walk, so that number was entirely the model recognising bars it had
        already been shown.
        """
        if self.train_end is None or features.empty:
            return 0
        times = pd.DatetimeIndex(features.index.get_level_values('event_time'))
        return int((times <= self.train_end).sum())


def feature_set_hash(columns: Sequence[str]) -> str:
    payload = json.dumps(list(map(str, columns)), separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _fit_head(spec: HeadSpec, x: pd.DataFrame, y: pd.Series, weights: np.ndarray):
    model = lgb.LGBMRegressor(**spec.to_params())
    model.fit(x, y, sample_weight=weights,
              categorical_feature=categorical_features(x.columns) or 'auto')
    return model


def train_forecast_model(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    config: Optional[Config] = None,
    profile: Optional[CoinProfile] = None,
    head_specs: Optional[dict[str, HeadSpec]] = None,
    validation_fraction: float = 0.2,
    data_as_of: Optional[str] = None,
    horizon_bars: Optional[int] = None,
    proxy_funding_symbols: Sequence[str] = (),
) -> Optional[ForecastModel]:
    """Fit the three heads on a (event_time, symbol) panel.

    The validation split is chronological and purged by one horizon. A random
    split would place a row's overlapping neighbours on the other side and report
    scores that cannot survive contact with live data.

    `horizon_bars` must be the horizon the *targets were built at*. It used to be
    read from the config unconditionally, which silently disagreed with the data
    whenever `--horizon` overrode the profile: the targets were built at 8h and
    the model purged at 96h and recorded 96h. Too wide is merely wasteful, but the
    same bug in the other direction — a horizon longer than the profile's — purges
    less than one label span and leaks.
    """
    config = config or Config()
    head_specs = head_specs or default_head_specs()

    x, y = align_panel(features, targets)
    if len(x) < MIN_FOLD_ROWS:
        logger.warning('not enough resolved rows to train: %d', len(x))
        return None

    x = add_symbol_feature(x)
    categories = (
        tuple(map(str, x[SYMBOL_COLUMN].cat.categories))
        if SYMBOL_COLUMN in x.columns
        else tuple(sorted(map(str, x.index.get_level_values('symbol').unique())))
    )
    horizon = int(horizon_bars) if horizon_bars else config.label_horizon_hours(profile)

    times = pd.DatetimeIndex(x.index.get_level_values('event_time'))
    unique_times = times.unique().sort_values()
    boundary = unique_times[int(len(unique_times) * (1 - validation_fraction))]

    purged = times < (boundary - pd.Timedelta(hours=horizon))
    purge_disabled = False
    if purged.sum() < MIN_FOLD_ROWS:
        # Falling back to an unpurged split means training labels overlap the
        # validation window, so the `ic` and `r2` reported below are optimistic.
        # This used to happen silently — and it happens precisely when history is
        # scarce, which is this system's normal condition, so the artifact carried
        # leaked validation figures with nothing to distinguish them.
        purge_disabled = True
        logger.warning(
            'purging %dh leaves only %d training rows (min %d): falling back to an '
            'UNPURGED split. The validation ic/r2 below overlap the training '
            'labels and are optimistic; recorded on the artifact as '
            'validation_purged=False.',
            horizon, int(purged.sum()), MIN_FOLD_ROWS,
        )
        purged = times < boundary
    train_mask, val_mask = purged, times >= boundary

    if train_mask.sum() < MIN_FOLD_ROWS or val_mask.sum() < 50:
        logger.warning('train/validation split too small: %d/%d',
                       train_mask.sum(), val_mask.sum())
        return None

    weights = panel_sample_weights(
        x.index[train_mask], horizon_bars=horizon,
        half_life_days=config.recency_half_life_days,
    )
    x_train, x_val = x[train_mask], x[val_mask]

    heads: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    for head, column in (('price', 'price'), ('carry', 'carry')):
        target = y[column]
        if target.loc[x_train.index].std() <= 0:
            continue          # nothing to learn: constant target
        model = _fit_head(head_specs[head], x_train, target.loc[x_train.index], weights)
        heads[head] = model

        forecast = model.predict(x_val)
        realised = target.loc[x_val.index].to_numpy()
        metrics[head] = {
            'ic': information_coefficient(forecast, realised),
            'r2': out_of_sample_r2(forecast, realised),
            'forecast_std_bps': float(np.std(forecast) * 10_000),
            'realised_std_bps': float(np.nanstd(realised) * 10_000),
        }

    # The dispersion head learns how large the price head's error will be, and it
    # must learn that from errors the price model has not already fitted.
    # Training it on in-sample residuals underestimated risk by a factor of 3.6,
    # which would size positions 3.6 times too large — the failure mode that
    # turns a marginal edge into a blow-up.
    if 'price' in heads:
        heads['dispersion'], dispersion_metrics = _fit_dispersion_head(
            head_specs['dispersion'], head_specs['price'],
            x_train, y['price'].loc[x_train.index], weights,
        )
        if heads['dispersion'] is None:
            heads.pop('dispersion')
        else:
            predicted = heads['dispersion'].predict(x_val)
            actual = np.abs(
                y['price'].loc[x_val.index].to_numpy() - heads['price'].predict(x_val)
            )
            metrics['dispersion'] = {
                **dispersion_metrics,
                'ic': information_coefficient(predicted, actual),
                'mean_predicted_bps': float(np.mean(predicted) * 10_000),
                'mean_actual_bps': float(np.nanmean(actual) * 10_000),
                'calibration_ratio': float(
                    np.mean(predicted) / np.nanmean(actual)
                ) if np.nanmean(actual) > 0 else float('nan'),
            }

    if not heads:
        return None

    empty = tuple(
        column for column in x.columns
        if column != SYMBOL_COLUMN and x[column].isna().all()
    )
    if proxy_funding_symbols:
        logger.warning(
            'carry trained on proxy funding for %d symbol(s): %s. Funding feeds '
            'the carry component of the target, so this measures a cash flow the '
            'traded venue does not pay',
            len(proxy_funding_symbols), ', '.join(sorted(proxy_funding_symbols)),
        )

    if empty:
        logger.warning(
            'trained on %d of %d declared features: %s carried no data at all. '
            'A feature group that produced nothing still arrives as an all-NaN '
            'column, and the feature-set hash cannot see the difference',
            len(x.columns) - len(empty), len(x.columns), ', '.join(empty),
        )

    return ForecastModel(
        heads=heads,
        feature_columns=tuple(x.columns),
        symbol_categories=categories,
        feature_set_hash=feature_set_hash(x.columns),
        horizon_bars=horizon,
        cost_config_version=config.cost_config_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        data_as_of=data_as_of,
        train_rows=int(train_mask.sum()),
        effective_observations=_panel_effective_observations(x.index[train_mask], horizon),
        validation_purged=not purge_disabled,
        empty_features=empty,
        proxy_funding_symbols=tuple(proxy_funding_symbols),
        train_start=pd.Timestamp(times[train_mask].min()) if train_mask.any() else None,
        train_end=pd.Timestamp(times[train_mask].max()) if train_mask.any() else None,
        metrics=metrics,
    )


def _fit_dispersion_head(
    dispersion_spec: HeadSpec,
    price_spec: HeadSpec,
    x: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    *,
    n_splits: int = 3,
) -> tuple[Optional[Any], dict[str, Any]]:
    """Fit the risk head on out-of-sample price residuals.

    Walk forward through the training window, and for each block predict it from
    everything before. Those residuals are the errors the price head actually
    makes on data it has not seen, which is what live sizing has to survive.
    In-sample residuals are smaller by a large and variable factor, so a model
    fitted to them reports confidence it has not earned.
    """
    times = pd.DatetimeIndex(x.index.get_level_values('event_time'))
    unique_times = times.unique().sort_values()
    if len(unique_times) < n_splits * 4:
        return None, {'reason': 'not_enough_history_for_oos_residuals'}

    edges = np.linspace(0, len(unique_times), n_splits + 2).astype(int)[1:]
    residuals = pd.Series(np.nan, index=x.index)

    for start, stop in zip(edges[:-1], edges[1:]):
        boundary = unique_times[start]
        block_end = unique_times[stop - 1]
        fit_mask = times < boundary
        block_mask = (times >= boundary) & (times <= block_end)
        if fit_mask.sum() < MIN_FOLD_ROWS or block_mask.sum() == 0:
            continue
        interim = _fit_head(price_spec, x[fit_mask], y[fit_mask], weights[fit_mask])
        residuals[block_mask] = np.abs(
            y[block_mask].to_numpy() - interim.predict(x[block_mask])
        )

    trained_on = residuals.notna()
    if trained_on.sum() < MIN_FOLD_ROWS:
        return None, {'reason': 'too_few_oos_residuals'}

    model = _fit_head(
        dispersion_spec, x[trained_on.to_numpy()],
        residuals[trained_on], weights[trained_on.to_numpy()],
    )
    in_sample = np.abs(y.to_numpy() - _fit_head(price_spec, x, y, weights).predict(x))
    return model, {
        'oos_residual_rows': int(trained_on.sum()),
        'mean_oos_residual_bps': float(residuals[trained_on].mean() * 10_000),
        'mean_in_sample_residual_bps': float(np.mean(in_sample) * 10_000),
        'in_sample_understatement': float(
            residuals[trained_on].mean() / np.mean(in_sample)
        ) if np.mean(in_sample) > 0 else float('nan'),
    }


def _panel_effective_observations(index: pd.MultiIndex, horizon_bars: int) -> float:
    total = 0.0
    for symbol in index.get_level_values('symbol').unique():
        mask = index.get_level_values('symbol') == symbol
        times = pd.DatetimeIndex(index[mask].get_level_values('event_time'))
        total += effective_sample_size(times, horizon_bars)
    return float(total)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """One fold's out-of-sample scores, per head."""

    fold: int
    train_rows: int
    test_rows: int
    effective_observations: float
    price_ic: float
    carry_ic: float
    net_ic: float
    price_ic_xs: float
    net_ic_xs: float
    # What net IC a forecast with NO skill scores on this fold. `expected_net`
    # and `realised_net` both carry the same `-cost` term, so the correlation
    # between them is positive before any prediction happens. Carried per fold
    # so net IC can be read against it rather than against zero.
    net_ic_cost_only: float
    net_ic_xs_cost_only: float
    identity_ceiling: float
    mean_expected_net_bps: float
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class CVReport:
    """Fold results plus the distributions the promotion gates read."""

    folds: list[FoldResult]
    price_ic: PathDistribution
    carry_ic: PathDistribution
    net_ic: PathDistribution
    price_ic_xs: PathDistribution
    net_ic_xs: PathDistribution
    net_ic_cost_only: PathDistribution
    net_ic_xs_cost_only: PathDistribution
    identity_ceiling: PathDistribution
    total_effective_observations: float

    @property
    def identity_ratio(self) -> float:
        """Model price IC as a fraction of the hindsight identity ceiling.

        Not a score. A diagnostic: as this approaches 1 the model's ranking looks
        increasingly like a ranking by instrument level, which does not persist.
        Low is uninformative on its own — it can mean either "no memorisation" or
        "no skill", and `price_ic` is what distinguishes those.
        """
        ceiling = abs(self.identity_ceiling.median)
        return float(abs(self.price_ic.median) / ceiling) if ceiling > 1e-9 else float('nan')

    @property
    def net_ic_skill(self) -> float:
        """Net IC above what a zero-skill forecast scores on the same folds.

        `expected_net` and `realised_net` share the `-cost` term, so their
        correlation starts positive: on this store at h=4h a forecast predicting
        price = 0 and carry = 0 scores +0.0714 pooled and +0.1094
        cross-sectionally, with every fold positive. Read against zero, that
        looks like a stable signal; it is the fee schedule on both sides of a
        correlation.

        This is the difference, and it is what net IC was being read as. A
        `volatility,trend,market_factor` set at h=4h scored net IC +0.0461
        against a +0.0714 floor — negative skill reported as six-of-six positive
        folds.
        """
        return float(self.net_ic.median - self.net_ic_cost_only.median)

    @property
    def net_ic_is_cost_only(self) -> bool:
        """True when net IC does not beat a forecast that predicts nothing."""
        return bool(np.isfinite(self.net_ic_skill) and self.net_ic_skill <= 0.0)

    @property
    def memorisation_suspected(self) -> bool:
        return bool(np.isfinite(self.identity_ratio)
                    and self.identity_ratio >= MEMORISATION_WARNING_RATIO)

    @property
    def carry_share_of_signal(self) -> float:
        """How much of the forecast quality is carry rather than direction.

        The single most useful diagnostic here. Near 1 means a carry harvester
        that happens to be dressed as a return forecast; near 0 means the system
        is betting on direction, and the honest prior for that is much worse.
        """
        price = abs(self.price_ic.median)
        carry = abs(self.carry_ic.median)
        return float(carry / (price + carry)) if (price + carry) > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            'n_folds': len(self.folds),
            'price_ic': self.price_ic.as_dict(),
            'carry_ic': self.carry_ic.as_dict(),
            'net_ic': self.net_ic.as_dict(),
            'price_ic_cross_sectional': self.price_ic_xs.as_dict(),
            'net_ic_cross_sectional': self.net_ic_xs.as_dict(),
            'net_ic_cost_only': self.net_ic_cost_only.as_dict(),
            'net_ic_cross_sectional_cost_only': self.net_ic_xs_cost_only.as_dict(),
            'net_ic_skill': round(self.net_ic_skill, 4),
            'net_ic_is_cost_only': self.net_ic_is_cost_only,
            'identity_ceiling_ic': self.identity_ceiling.as_dict(),
            'identity_ratio': round(self.identity_ratio, 3),
            'memorisation_suspected': self.memorisation_suspected,
            'carry_share_of_signal': round(self.carry_share_of_signal, 3),
            'effective_observations': round(self.total_effective_observations, 1),
        }

    def __str__(self) -> str:
        warning = '  MEMORISATION SUSPECTED' if self.memorisation_suspected else ''
        if self.net_ic_is_cost_only:
            warning += '  NET IC IS THE COST TERM'
        return (
            f"{len(self.folds)} folds | "
            f"price IC {self.price_ic.median:+.4f} (xs {self.price_ic_xs.median:+.4f}) | "
            f"carry IC {self.carry_ic.median:+.4f} | "
            f"net IC {self.net_ic.median:+.4f} "
            f"(cost-only floor {self.net_ic_cost_only.median:+.4f}, "
            f"skill {self.net_ic_skill:+.4f}) | "
            f"carry share {self.carry_share_of_signal:.0%} | "
            f"identity ceiling {self.identity_ceiling.median:+.4f} "
            f"(ratio {self.identity_ratio:.2f}){warning} | "
            f"{self.total_effective_observations:.0f} effective obs"
        )


def cross_validate_forecast(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    config: Optional[Config] = None,
    profile: Optional[CoinProfile] = None,
    head_specs: Optional[dict[str, HeadSpec]] = None,
    folds: Optional[Sequence[CVFold]] = None,
    n_folds: int = 6,
    scale: bool = True,
    horizon_bars: Optional[int] = None,
) -> CVReport:
    """Refit per fold and score out of sample, returning distributions.

    Each fold refits from scratch and scales inside the fold, so every number
    that comes back is out of sample by construction.

    `horizon_bars` is the horizon the targets were built at, and it sets the purge
    width. Getting it from the config instead would purge the profile's horizon
    even when the targets used a different one.
    """
    config = config or Config()
    head_specs = head_specs or default_head_specs()

    x, y = align_panel(features, targets)
    empty = summarise_paths([])
    if len(x) < MIN_FOLD_ROWS:
        return CVReport([], empty, empty, empty, empty, empty, empty, empty,
                        empty, 0.0)

    x = add_symbol_feature(x)
    categories = (
        x[SYMBOL_COLUMN].cat.categories if SYMBOL_COLUMN in x.columns else None
    )
    horizon = int(horizon_bars) if horizon_bars else config.label_horizon_hours(profile)

    times = pd.DatetimeIndex(x.index.get_level_values('event_time'))
    unique_times = times.unique().sort_values()
    if folds is None:
        folds = purged_walk_forward(
            unique_times, n_folds=n_folds,
            min_train_bars=max(len(unique_times) // 4, 1),
            purge_bars=horizon, embargo_bars=horizon,
        )

    numeric = [c for c in x.columns if c != SYMBOL_COLUMN]
    results: list[FoldResult] = []

    for number, (train_rows, test_rows) in enumerate(time_folds(x.index, folds)):
        x_train, x_test = x.iloc[train_rows], x.iloc[test_rows]
        y_train, y_test = y.iloc[train_rows], y.iloc[test_rows]

        if scale:
            pre = FoldPreprocessor(scaler_factory=RobustScaler)
            scaled_train = pre.fit_transform(x_train[numeric])
            scaled_test = pre.transform(x_test[numeric])
            if categories is not None:
                # Re-wrap: assigning the raw array back yields object dtype,
                # which LightGBM rejects outright.
                scaled_train[SYMBOL_COLUMN] = pd.Categorical(
                    x_train[SYMBOL_COLUMN], categories=categories)
                scaled_test[SYMBOL_COLUMN] = pd.Categorical(
                    x_test[SYMBOL_COLUMN], categories=categories)
            x_train, x_test = scaled_train[x.columns], scaled_test[x.columns]

        weights = panel_sample_weights(
            x_train.index, horizon_bars=horizon,
            half_life_days=config.recency_half_life_days,
        )

        forecasts: dict[str, np.ndarray] = {}
        for head in ('price', 'carry'):
            target = y_train[head]
            if target.std() <= 0:
                forecasts[head] = np.zeros(len(x_test))
                continue
            model = _fit_head(head_specs[head], x_train, target, weights)
            forecasts[head] = model.predict(x_test)

        cost = y_test['cost'].to_numpy()
        expected_net = forecasts['price'] + forecasts['carry'] - cost
        realised_net = y_test['price'].to_numpy() + y_test['carry'].to_numpy() - cost
        # The same quantity for a forecast that predicts nothing. Cost is known
        # at decision time, not forecast, so this is the floor net IC has to
        # clear before any of it is skill. Measured on this store at h=4h it is
        # +0.0714 pooled and +0.1094 cross-sectionally, which is larger than
        # every net IC any feature set has produced here.
        zero_skill = -cost

        results.append(FoldResult(
            fold=number,
            train_rows=int(len(x_train)),
            test_rows=int(len(x_test)),
            effective_observations=_panel_effective_observations(x_test.index, horizon),
            price_ic=information_coefficient(forecasts['price'], y_test['price'].to_numpy()),
            carry_ic=information_coefficient(forecasts['carry'], y_test['carry'].to_numpy()),
            net_ic=information_coefficient(expected_net, realised_net),
            net_ic_cost_only=information_coefficient(zero_skill, realised_net),
            net_ic_xs_cost_only=cross_sectional_ic(zero_skill, realised_net, x_test.index),
            price_ic_xs=cross_sectional_ic(
                forecasts['price'], y_test['price'].to_numpy(), x_test.index),
            net_ic_xs=cross_sectional_ic(expected_net, realised_net, x_test.index),
            identity_ceiling=identity_ceiling_ic(y_test['price']),
            mean_expected_net_bps=float(np.nanmean(expected_net) * 10_000),
            predictions=pd.DataFrame(
                {'expected_net': expected_net, 'realised_net': realised_net},
                index=x_test.index,
            ),
        ))

    def distribution(attribute: str) -> PathDistribution:
        return summarise_paths([
            getattr(r, attribute) for r in results if np.isfinite(getattr(r, attribute))
        ])

    return CVReport(
        folds=results,
        price_ic=distribution('price_ic'),
        carry_ic=distribution('carry_ic'),
        net_ic=distribution('net_ic'),
        price_ic_xs=distribution('price_ic_xs'),
        net_ic_xs=distribution('net_ic_xs'),
        net_ic_cost_only=distribution('net_ic_cost_only'),
        net_ic_xs_cost_only=distribution('net_ic_xs_cost_only'),
        identity_ceiling=distribution('identity_ceiling'),
        total_effective_observations=sum(r.effective_observations for r in results),
    )
