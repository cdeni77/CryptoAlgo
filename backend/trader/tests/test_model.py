"""Forecast model invariants.

The tests that matter here are the ones that catch a model reporting skill it
does not have. Two mechanisms produce that, and both are checked:

* Instrument identity. A forecast that knows only each instrument's average
  outcome — zero timing skill — scores a rank IC above +0.2 on a panel of random
  walks, because realised drift differs by sample.
* In-sample risk estimates. Fitting the dispersion head on residuals the price
  head has already minimised understates risk by a factor of two or more, and
  position size scales inversely with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.model import (
    MAE_TO_SIGMA,
    MIN_FOLD_ROWS,
    SYMBOL_COLUMN,
    ForecastModel,
    add_symbol_feature,
    align_panel,
    cross_sectional_ic,
    cross_validate_forecast,
    identity_ceiling_ic,
    information_coefficient,
    panel_sample_weights,
    time_folds,
    train_forecast_model,
)
from core.cv import purged_walk_forward
from core.targets import build_target_panel

N_BARS = 1_800
SYMBOLS = ('BIP', 'ETP', 'SLP', 'XPP', 'DOP')
CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


def _bars(price: float, seed: int, n: int = N_BARS) -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(1e-4, 0.012, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {'open': open_,
         'high': np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n))),
         'low': np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n))),
         'close': close, 'volume': rng.lognormal(8, 0.6, n)},
        index=index,
    )


def _persistent_funding(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """An AR(1), which is what real funding looks like: mean-reverting, sticky."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 2e-5, len(index))
    rate = np.zeros(len(index))
    for i in range(1, len(index)):
        rate[i] = 0.985 * rate[i - 1] + shocks[i]
    return pd.DataFrame({'rate': rate + 3e-5}, index=index)


@pytest.fixture(scope='module')
def panel(config):
    """A feature panel and matching targets, built the way production does."""
    from core.features import SymbolInputs, build_panel

    market = _bars(60_000, 1)
    inputs, bars_by, funding_by = [], {}, {}
    for i, (symbol, price) in enumerate(
        zip(SYMBOLS, (60_000, 3_000, 150, 2.2, 0.35))
    ):
        bars = _bars(price, i + 1)
        index = bars.index
        rng = np.random.default_rng(i + 50)
        funding = _persistent_funding(index, i + 90)
        reference = bars.copy()
        reference['close'] = bars['close'] * (1 + np.cumsum(rng.normal(2e-5, 3e-4, len(index))))
        bars_by[symbol] = bars
        funding_by[symbol] = funding
        inputs.append(SymbolInputs(
            symbol, bars, funding,
            pd.DataFrame({'oi_contracts': np.abs(np.cumsum(rng.normal(0, 50, len(index))) + 5e4)},
                         index=index),
            reference, market,
        ))

    features = build_panel(inputs, config=config)
    targets = build_target_panel(
        bars_by, funding_by_symbol=funding_by, config=config, horizon_bars=48,
        index_by_symbol={s: features.xs(s, level='symbol').index for s in SYMBOLS},
    )
    return features, targets


# ---------------------------------------------------------------------------
# Memorisation
# ---------------------------------------------------------------------------


def test_identity_alone_scores_a_high_ic():
    """Quantifies the hazard the identity ceiling exists to measure.

    Built to match the real target: an overlapping multi-bar forward return on a
    random walk. Overlap is what makes this bite — adjacent rows share almost all
    of their outcome, so each instrument's realised drift dominates its whole
    column, and ranking by that drift sorts the outcomes well even though the
    forecast has no timing skill at all.
    """
    horizon, n = 96, 1_200
    index = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')

    frames = []
    for i, symbol in enumerate(SYMBOLS):
        rng = np.random.default_rng(i + 1)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(1e-4, 0.012, n))), index=index)
        forward = (close.shift(-horizon) / close) - 1.0
        frames.append(pd.DataFrame({'symbol': symbol, 'event_time': index,
                                    'price': forward.to_numpy()}))
    outcome = (pd.concat(frames).set_index(['event_time', 'symbol'])
               .sort_index()['price'].dropna())

    ceiling = identity_ceiling_ic(outcome)

    assert ceiling > 0.15, f'expected identity to rank well, got {ceiling}'


def test_identity_ceiling_collapses_without_overlap():
    """The contrast that shows what drives the hazard.

    Same panel shape, non-overlapping i.i.d. outcomes: the ceiling falls to near
    zero, because per-instrument mean differences are then tiny beside the noise.
    So the hazard is a property of overlapping targets rather than of panels in
    general, and a shorter horizon is one real mitigation.
    """
    index = pd.date_range('2026-01-01', periods=1_200, freq='1h', tz='UTC')
    frames = []
    for i, symbol in enumerate(SYMBOLS):
        rng = np.random.default_rng(i + 1)
        frames.append(pd.DataFrame({
            'symbol': symbol, 'event_time': index,
            'price': rng.normal(1e-4 * (i - 2), 0.02, len(index)),
        }))
    outcome = (pd.concat(frames).set_index(['event_time', 'symbol'])
               .sort_index()['price'])

    assert identity_ceiling_ic(outcome) < 0.1
def test_symbol_identity_is_off_by_default(panel, config):
    """Because it lets a tree memorise per-instrument drift."""
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)

    assert model is not None
    assert not model.uses_symbol_identity
    assert SYMBOL_COLUMN not in model.feature_columns


def test_add_symbol_feature_respects_the_switch():
    index = pd.MultiIndex.from_product(
        [pd.date_range('2026-01-01', periods=3, freq='1h', tz='UTC'), ['BIP', 'ETP']],
        names=['event_time', 'symbol'],
    )
    frame = pd.DataFrame({'x': np.arange(6, dtype=float)}, index=index)

    assert SYMBOL_COLUMN not in add_symbol_feature(frame, enabled=False).columns
    enabled = add_symbol_feature(frame, enabled=True)
    assert str(enabled[SYMBOL_COLUMN].dtype) == 'category'


# ---------------------------------------------------------------------------
# Risk estimation
# ---------------------------------------------------------------------------


def test_dispersion_head_is_calibrated_out_of_sample(panel, config):
    """Position size scales inversely with this, so understating it over-levers.

    Trained on in-sample residuals the head understated risk by a factor above
    two. Trained on walk-forward residuals it should land near one.
    """
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)
    dispersion = model.metrics.get('dispersion')

    assert dispersion is not None, 'dispersion head did not fit'
    assert dispersion['in_sample_understatement'] > 1.2, (
        'in-sample residuals should be visibly smaller; if not, this test no '
        'longer demonstrates the hazard it was written for'
    )
    assert 0.6 < dispersion['calibration_ratio'] < 1.6, (
        f"risk estimate is off by {dispersion['calibration_ratio']:.2f}x"
    )


def test_sigma_conversion_is_the_gaussian_constant():
    """The head predicts mean absolute error; sizing needs a standard deviation."""
    assert MAE_TO_SIGMA == pytest.approx(np.sqrt(np.pi / 2))


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_folds_never_split_a_timestamp(panel, config):
    """Every instrument at one bar goes to the same side.

    Splitting rows instead would put SOL at 14:00 in train and BTC at 14:00 in
    test, and those rows share the market move that decides both outcomes.
    """
    features, targets = panel
    x, _ = align_panel(features, targets)
    times = pd.DatetimeIndex(x.index.get_level_values('event_time'))
    unique = times.unique().sort_values()

    folds = purged_walk_forward(
        unique, n_folds=5, min_train_bars=len(unique) // 4,
        purge_bars=48, embargo_bars=48,
    )

    for train_rows, test_rows in time_folds(x.index, folds):
        assert not set(times[train_rows]) & set(times[test_rows])
        # and the test side holds every row at its timestamps
        at_test_times = np.flatnonzero(times.isin(list(set(times[test_rows]))))
        assert set(at_test_times) == set(test_rows)


def test_weights_are_per_instrument(panel):
    """Overlap belongs to one instrument's own label windows.

    Pooling the calculation would treat two instruments' simultaneous outcomes
    as overlapping, when they are the cross-section the model exists to use.
    """
    features, targets = panel
    x, _ = align_panel(features, targets)

    weights = panel_sample_weights(x.index, horizon_bars=48, half_life_days=50)

    assert weights.mean() == pytest.approx(1.0)
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------


def test_forecast_preserves_the_cost_identity(panel, config):
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)

    resolved = features.loc[targets.dropna(subset=['price']).index]
    cost = targets['cost'].reindex(resolved.index).to_numpy()
    forecast = model.predict(resolved, cost=cost)

    assert np.allclose(forecast['net_long'] + forecast['net_short'], -2 * cost)
    assert not ((forecast['net_long'] > 0) & (forecast['net_short'] > 0)).any()


def test_stand_aside_when_neither_side_clears_cost(panel, config):
    """A forecast below cost must produce no position, not a small one."""
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)
    resolved = features.loc[targets.dropna(subset=['price']).index]

    # An absurd cost no forecast can clear.
    forecast = model.predict(resolved, cost=1.0)

    assert (forecast['side'] == 0).all()
    assert (forecast['expected_net'] == 0).all()
    assert (forecast['edge_to_risk'] == 0).all()


def test_edge_to_risk_is_guarded_against_zero_sigma(panel, config):
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)
    resolved = features.loc[targets.dropna(subset=['price']).index]

    forecast = model.predict(resolved, cost=0.0005)

    assert np.isfinite(forecast['edge_to_risk']).all()


def test_carry_head_beats_the_price_head_when_funding_is_persistent(panel, config):
    """The decomposition's whole purpose.

    Funding here is an AR(1) and price is a random walk, so a working system must
    report that the predictable component is carry. A single net-return head
    would report one number and hide which half it came from.
    """
    features, targets = panel
    report = cross_validate_forecast(features, targets, config=config, n_folds=5)

    assert report.folds
    assert report.carry_ic.median > report.price_ic.median
    assert report.carry_share_of_signal > 0.5


def test_cross_sectional_ic_needs_a_universe():
    index = pd.MultiIndex.from_product(
        [pd.date_range('2026-01-01', periods=50, freq='1h', tz='UTC'), ['BIP', 'ETP']],
        names=['event_time', 'symbol'],
    )
    prediction = np.arange(100, dtype=float)
    outcome = np.arange(100, dtype=float)

    # Two instruments per timestamp is below min_universe, so nothing is scored.
    assert np.isnan(cross_sectional_ic(prediction, outcome, index, min_universe=3))


def test_information_coefficient_rejects_tiny_samples():
    assert np.isnan(information_coefficient(np.arange(5.0), np.arange(5.0)))
    assert information_coefficient(np.arange(50.0), np.arange(50.0)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_model_records_what_it_trained_on(panel, config):
    features, targets = panel
    model = train_forecast_model(features, targets, config=config, data_as_of='2026-06-01')

    provenance = model.provenance()

    assert provenance['cost_config_version'] == 'coinbase_us_perps_cde_v202602'
    assert provenance['data_as_of'] == '2026-06-01'
    assert provenance['feature_set_hash']
    assert provenance['effective_observations'] > 0
    assert set(provenance['heads']) >= {'price', 'carry'}


def test_stale_feature_set_is_rejected(panel, config):
    """A silent column mismatch produces confident garbage."""
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)

    with pytest.raises(ValueError, match='absent'):
        model.assert_compatible(features.drop(columns=features.columns[:5]))


def test_model_round_trips_through_disk(panel, config, tmp_path):
    features, targets = panel
    model = train_forecast_model(features, targets, config=config)
    resolved = features.loc[targets.dropna(subset=['price']).index]
    before = model.predict(resolved, cost=0.0005)

    path = model.save(tmp_path / 'forecast.joblib')
    after = ForecastModel.load(path).predict(resolved, cost=0.0005)

    pd.testing.assert_frame_equal(before, after)


def test_refuses_to_train_on_too_little_data(config):
    index = pd.MultiIndex.from_product(
        [pd.date_range('2026-01-01', periods=10, freq='1h', tz='UTC'), ['BIP']],
        names=['event_time', 'symbol'],
    )
    features = pd.DataFrame({'x': np.arange(10, dtype=float)}, index=index)
    targets = pd.DataFrame(
        {'price': np.arange(10, dtype=float), 'carry': 0.0, 'cost': 0.0005},
        index=index,
    )

    assert train_forecast_model(features, targets, config=config) is None


# ---------------------------------------------------------------------------
# The horizon has to come from the data
# ---------------------------------------------------------------------------


def test_the_model_records_the_horizon_its_targets_were_built_at(panel, config):
    """`--horizon` overrides the profile, and the model has to follow.

    The horizon was read from the config unconditionally, so a run with
    `--horizon 8` built targets at 8h while the model purged its validation split
    at the profile's 96h and recorded 96h in its provenance. Too wide is merely
    wasteful — it discards training rows — but the same bug with a horizon
    *longer* than the profile's purges less than one label span, and that leaks.

    The recorded value is not cosmetic: it drives `effective_observations`, which
    is the number every significance claim downstream rests on. At 96 instead of 8
    it understated the effective sample twelvefold.
    """
    features, targets = panel

    eight = train_forecast_model(features, targets, config=config, horizon_bars=8)
    default = train_forecast_model(features, targets, config=config)

    assert eight is not None and default is not None
    assert eight.horizon_bars == 8, 'the model ignored the horizon it was given'
    assert default.horizon_bars == config.label_horizon_hours(None), (
        'without an explicit horizon the config default should still apply'
    )

    # A shorter horizon means less overlap, so more independent observations from
    # the same rows. Roughly proportional, and certainly not equal.
    assert eight.effective_observations > default.effective_observations, (
        f'a shorter horizon should raise the effective sample: '
        f'{eight.effective_observations:.0f} vs {default.effective_observations:.0f}'
    )


def test_the_walk_forward_backtest_trains_at_the_dataset_horizon(panel, config):
    """The retrain inside `walk_forward_backtest` must agree with the targets too."""
    from core.backtest import generate_walk_forward_forecasts

    features, targets = panel
    generated = generate_walk_forward_forecasts(
        features, targets, config=config, n_periods=2, horizon_bars=8,
    )

    assert generated.models, 'no models were fitted'
    assert all(m.horizon_bars == 8 for m in generated.models), (
        f'walk-forward models recorded '
        f'{sorted({m.horizon_bars for m in generated.models})} instead of 8'
    )


def test_a_perfect_feature_is_recovered_end_to_end():
    """The alignment canary: inject the answer and the pipeline must find it.

    Every "why is the IC zero" investigation has to rule this out first, and
    nothing in the suite did. A one-bar shift between the feature panel and the
    target — either direction — destroys signal silently and produces exactly the
    reading this repo spent months interpreting: IC indistinguishable from zero,
    on every feature set, at every horizon.

    Lookahead tests cannot catch it. They assert the model does not see the
    future; this asserts it *does* see the present, which is the opposite failure
    and has no other guard. If the panel, the targets, the fold splitter, the
    purge or the per-fold scaler misaligns anything, a feature that is literally
    the realised outcome stops scoring near 1.0 and this fails.

    Deliberately not `1.0`: `purged_walk_forward` drops the bars around each fold
    boundary, LightGBM is a step function over a shallow tree, and the injected
    column is scaled inside the fold. 0.9 clears all of that while leaving no room
    for a misalignment.
    """
    import numpy as np
    import pandas as pd

    from core.model import cross_validate_forecast

    horizon = 4
    times = pd.date_range('2026-01-01', periods=1_400, freq='h', tz='UTC')
    symbols = ('BIP', 'ETP', 'XPP', 'SLP')
    index = pd.MultiIndex.from_product([times, symbols], names=['event_time', 'symbol'])

    rng = np.random.default_rng(19)
    realised_price = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    targets = pd.DataFrame({
        'price': realised_price,
        'carry': 0.0,
        'cost': 0.0027,
        'net_long': realised_price - 0.0027,
        'net_short': -realised_price - 0.0027,
    }, index=index)

    # One informative column plus noise, so the fit has something to ignore.
    features = pd.DataFrame({
        'oracle': realised_price,
        'noise_a': rng.normal(0, 1, len(index)),
        'noise_b': rng.normal(0, 1, len(index)),
    }, index=index)

    report = cross_validate_forecast(
        features, targets, n_folds=4, horizon_bars=horizon,
    )

    assert report.folds, 'no folds were evaluated'
    assert report.price_ic.median > 0.9, (
        f'a feature equal to the realised outcome scored price IC '
        f'{report.price_ic.median:+.4f}. The panel and the targets are not '
        f'aligned on (event_time, symbol), or a fold boundary is shifting one '
        f'of them.'
    )
    assert report.price_ic.positive_fraction == 1.0, (
        f'the oracle feature failed in some folds: {[f.price_ic for f in report.folds]}'
    )


def test_the_canary_fails_when_the_target_is_shifted():
    """Mutation check: the canary above must actually be able to fail.

    A test that passes on a broken pipeline is worse than no test. Shift the
    target by one bar per symbol — the exact defect the canary exists to catch —
    and the oracle feature must stop working.
    """
    import numpy as np
    import pandas as pd

    from core.model import cross_validate_forecast

    times = pd.date_range('2026-01-01', periods=1_400, freq='h', tz='UTC')
    symbols = ('BIP', 'ETP', 'XPP', 'SLP')
    index = pd.MultiIndex.from_product([times, symbols], names=['event_time', 'symbol'])

    rng = np.random.default_rng(23)
    truth = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    # The feature sees the outcome one bar later than the target records it.
    shifted = truth.groupby(level='symbol').shift(1).fillna(0.0)

    targets = pd.DataFrame({'price': truth, 'carry': 0.0, 'cost': 0.0027},
                           index=index)
    features = pd.DataFrame({'oracle': shifted,
                             'noise': rng.normal(0, 1, len(index))}, index=index)

    report = cross_validate_forecast(features, targets, n_folds=4, horizon_bars=4)

    assert abs(report.price_ic.median) < 0.15, (
        f'a one-bar misalignment still scored {report.price_ic.median:+.4f} — the '
        f'canary cannot distinguish aligned from misaligned data'
    )
