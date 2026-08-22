"""Execution, decision and backtest invariants.

The first test in this file is the most important one in the suite. A backtest
that reports a profit on a driftless random walk has a lookahead, and every other
number the system produces is then meaningless. It is checked directly rather
than reasoned about, because reasoning is how the bug got in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.execution import (
    MAINTENANCE_MARGIN_FRACTION,
    ExitReason,
    Position,
    accrue_funding,
    barrier_prices,
    fill_price,
    fractional_kelly,
    participation_rate,
    resolve_bar,
    size_from_forecast,
    slippage_bps,
)
from core.features import SymbolInputs, build_panel
from core.model import train_forecast_model
from core.profiles import COIN_PROFILES
from core.signal import Gate, GateCounter, Decision, DecisionContext, decide
from core.targets import build_target_panel
from core.backtest import backtest_from_model, walk_forward_backtest

CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'
UNIVERSE = {'BIP': 60_000.0, 'ETP': 3_000.0, 'SLP': 150.0, 'XPP': 2.2, 'DOP': 0.35}
PROFILE_FOR = {'BIP': 'BTC', 'ETP': 'ETH', 'SLP': 'SOL', 'XPP': 'XRP', 'DOP': 'DOGE'}
N_BARS = 1_400

# Enough independent draws for a t-statistic to mean something. The lookahead
# this guards against produced t = +7, so six seeds separate it from noise with
# room to spare while keeping the test inside a tolerable runtime.
SEEDS_FOR_SIGNIFICANCE = 6


def _t_statistic(values) -> float:
    """One-sample t against zero. Returns 0 when the sample cannot support one."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    error = float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return float(np.mean(values) / error) if error > 0 else 0.0


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


def _panel(seed_offset: int, *, drift: float = 0.0, funding_mean: float = 0.0):
    """A synthetic universe. `drift=0` is the case that exposes lookahead."""
    def bars(price: float, seed: int) -> pd.DataFrame:
        index = pd.date_range('2026-01-01', periods=N_BARS, freq='1h', tz='UTC')
        rng = np.random.default_rng(seed * 1_000 + seed_offset)
        close = price * np.exp(np.cumsum(rng.normal(drift, 0.012, N_BARS)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        return pd.DataFrame(
            {'open': open_,
             'high': np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, N_BARS))),
             'low': np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, N_BARS))),
             'close': close, 'volume': rng.lognormal(11, 0.6, N_BARS)},
            index=index,
        )

    market = bars(60_000, 1)
    inputs, bars_by, funding_by = [], {}, {}
    for i, (symbol, price) in enumerate(UNIVERSE.items()):
        frame = bars(price, i + 1)
        index = frame.index
        rng = np.random.default_rng((i + 50) * 1_000 + seed_offset)
        shocks = rng.normal(0, 2e-5, N_BARS)
        rate = np.zeros(N_BARS)
        for k in range(1, N_BARS):
            rate[k] = 0.985 * rate[k - 1] + shocks[k]
        funding = pd.DataFrame({'rate': rate + funding_mean}, index=index)
        reference = frame.copy()
        reference['close'] = frame['close'] * (
            1 + np.cumsum(rng.normal(2e-5, 3e-4, N_BARS))
        )
        bars_by[symbol] = frame
        funding_by[symbol] = funding
        inputs.append(SymbolInputs(
            symbol, frame, funding,
            pd.DataFrame({'oi_contracts': np.abs(np.cumsum(rng.normal(0, 50, N_BARS)) + 5e4)},
                         index=index),
            reference, market,
        ))
    return inputs, bars_by, funding_by


def _features_targets(config, seed_offset: int, **kwargs):
    inputs, bars_by, funding_by = _panel(seed_offset, **kwargs)
    features = build_panel(inputs, config=config)
    profiles = {k: COIN_PROFILES[v] for k, v in PROFILE_FOR.items()}
    targets = build_target_panel(
        bars_by, profiles=profiles, funding_by_symbol=funding_by, config=config,
        horizon_bars=48,
        index_by_symbol={s: features.xs(s, level='symbol').index for s in UNIVERSE},
    )
    return features, targets, bars_by, funding_by, profiles


# ---------------------------------------------------------------------------
# The lookahead test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_walk_forward_finds_no_edge_in_driftless_random_walks(config):
    """There is nothing to find in a driftless random walk except the fee bill.

    Trading in-sample forecasts produced a mean price PnL of +95,000 with a
    t-statistic of +7 across six seeds — the model recognising bars it had been
    shown. This is the regression test for that, and the assertion has to be
    stated as a significance test rather than a sign test: a driftless walk has
    zero *expected* price PnL, not negative, so demanding a loss on the mean of
    a few seeds fails whenever the walks happen to go the model's way. A single
    levered run has enormous per-trade variance, and a t-statistic is the only
    honest way to separate a lookahead from luck.
    """
    nets, prices = [], []
    for seed in range(SEEDS_FOR_SIGNIFICANCE):
        features, targets, bars_by, funding_by, profiles = _features_targets(
            config, seed, drift=0.0, funding_mean=0.0
        )
        result, generated = walk_forward_backtest(
            features, targets, bars_by_symbol=bars_by, funding_by_symbol=funding_by,
            config=config, profiles=profiles, n_periods=3,
        )
        assert generated.periods, 'walk-forward produced no periods'
        # Costs are a certainty on every path, whatever the price did.
        assert result.net_pnl < result.price_pnl, (
            f'seed {seed}: net {result.net_pnl:+,.0f} exceeds gross price PnL '
            f'{result.price_pnl:+,.0f} — costs are not being charged'
        )
        nets.append(result.net_pnl)
        prices.append(result.price_pnl)

    assert _t_statistic(prices) < 2.0, (
        f'price PnL t-statistic of {_t_statistic(prices):+.2f} on driftless '
        f'random walks indicates a lookahead (mean {np.mean(prices):+,.0f})'
    )
    assert _t_statistic(nets) < 2.0, (
        f'net PnL t-statistic of {_t_statistic(nets):+.2f} — a strategy that '
        f'profits from noise is a bug, not an edge'
    )


def test_backtesting_in_sample_rows_is_refused(config):
    """The guard that stops the lookahead recurring."""
    features, targets, bars_by, funding_by, profiles = _features_targets(config, 0)
    model = train_forecast_model(features, targets, config=config)
    resolved = features.loc[targets.dropna(subset=['price']).index]

    assert model.in_sample_rows(resolved) > 0

    with pytest.raises(ValueError, match='training window'):
        backtest_from_model(
            model, resolved, bars_by_symbol=bars_by, costs=targets['cost'],
            funding_by_symbol=funding_by, config=config, profiles=profiles,
        )


def test_in_sample_is_available_when_explicitly_requested(config):
    features, targets, bars_by, funding_by, profiles = _features_targets(config, 0)
    model = train_forecast_model(features, targets, config=config)
    resolved = features.loc[targets.dropna(subset=['price']).index]

    result = backtest_from_model(
        model, resolved, bars_by_symbol=bars_by, costs=targets['cost'],
        funding_by_symbol=funding_by, config=config, profiles=profiles,
        allow_in_sample=True,
    )

    assert result.n_trades >= 0        # runs; the number is not to be trusted


def test_walk_forward_forecasts_are_all_out_of_sample(config):
    """Every forecast row must postdate the model that produced it."""
    features, targets, bars_by, funding_by, profiles = _features_targets(config, 1)
    _, generated = walk_forward_backtest(
        features, targets, bars_by_symbol=bars_by, funding_by_symbol=funding_by,
        config=config, profiles=profiles, n_periods=3,
    )

    for model, (period_start, _) in zip(generated.models, generated.periods):
        assert model.train_end < period_start


# ---------------------------------------------------------------------------
# Slippage and fills
# ---------------------------------------------------------------------------


def test_slippage_grows_with_participation():
    small = slippage_bps(1, 100.0, 1e6, 'SLP', spread_bps=4.0)
    large = slippage_bps(50_000, 100.0, 1e6, 'SLP', spread_bps=4.0)

    assert small == pytest.approx(2.0)      # half the spread, no impact
    assert large > small * 5


def test_participation_is_bounded():
    assert participation_rate(10, 100.0, 0.0, 'SLP') == 1.0        # no volume
    assert 0.0 <= participation_rate(1, 100.0, 1e9, 'SLP') <= 1.0


def test_slippage_always_moves_against_the_order():
    assert fill_price(100.0, 1, 10.0) > 100.0        # buying pays more
    assert fill_price(100.0, -1, 10.0) < 100.0       # selling receives less


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def test_long_pays_positive_funding():
    position = Position('SLP', 1, 10, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=1.0, margin=1_000.0)

    payment = accrue_funding(position, 0.0001, 100.0)

    assert payment > 0                       # a cost to the account
    assert position.funding_paid == pytest.approx(payment)


def test_short_receives_positive_funding():
    position = Position('SLP', -1, 10, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=1.0, margin=1_000.0)

    assert accrue_funding(position, 0.0001, 100.0) < 0


def test_funding_is_charged_on_notional_not_margin():
    """This is why it bites at leverage."""
    position = Position('SLP', 1, 10, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=0.0, margin=1_250.0)

    payment = accrue_funding(position, 0.0001, 100.0)
    notional = position.notional(100.0)

    assert payment == pytest.approx(notional * 0.0001)
    assert notional > position.margin


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


def _bar(high: float, low: float, close: float) -> pd.Series:
    return pd.Series({'open': close, 'high': high, 'low': low, 'close': close,
                      'volume': 1e6})


def test_stop_resolves_before_take_profit_in_the_same_bar():
    """Bar data cannot order two touches; assuming the good one inflates results."""
    position = Position('SLP', 1, 10, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=0.0, margin=100_000.0,
                        take_profit=105.0, stop_loss=98.0)

    outcome = resolve_bar(position, _bar(106.0, 97.0, 100.0),
                          pd.Timestamp('2026-01-01T01', tz='UTC'))

    assert outcome.exited
    assert outcome.reason is ExitReason.STOP_LOSS


def test_liquidation_takes_precedence_over_every_other_exit():
    """If the bar's extreme wiped the account, nothing else in it happened.

    Sized at 4x, where liquidation sits well below the stop, so the ordering is
    actually being tested rather than an already-dead position.
    """
    position = Position('SLP', 1, 100, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=0.0, margin=12_500.0,
                        take_profit=105.0, stop_loss=99.0)

    liquidation = position.liquidation_price()
    assert not position.under_margined
    assert 0 < liquidation < 99.0, f'expected liquidation below the stop, got {liquidation}'

    outcome = resolve_bar(position, _bar(106.0, liquidation - 1.0, 100.0),
                          pd.Timestamp('2026-01-01T01', tz='UTC'))

    assert outcome.reason is ExitReason.LIQUIDATION
    assert outcome.liquidated


def test_short_liquidates_above_entry_at_the_right_level():
    """The short denominator was wrong, liquidating about ten percent too late.

        long   P = (units * entry - available) / (units * (1 - maintenance))
        short  P = (units * entry + available) / (units * (1 + maintenance))
    """
    units = 500.0        # 100 SLP contracts at 5 units
    short = Position('SLP', -1, 100, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                     entry_fee=0.0, margin=12_500.0)
    expected = (units * 100.0 + 12_500.0) / (units * (1 + MAINTENANCE_MARGIN_FRACTION))

    assert short.liquidation_price() == pytest.approx(expected)
    assert short.liquidation_price() > 100.0        # a short is hurt by a rise
    assert short.liquidation_price() == pytest.approx(119.05, abs=0.01)


def test_under_margined_position_reports_no_liquidation_level():
    """At 50x with 5% maintenance the position is dead on arrival.

    The solved price then lands on the wrong side of entry, which is meaningless,
    so the level is reported as zero and the position exits immediately.
    """
    position = Position('SLP', 1, 100, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=0.0, margin=1_000.0)

    assert position.under_margined
    assert position.liquidation_price() == 0.0

    outcome = resolve_bar(position, _bar(101.0, 99.0, 100.0),
                          pd.Timestamp('2026-01-01T01', tz='UTC'))
    assert outcome.reason is ExitReason.LIQUIDATION


def test_horizon_exit_when_no_barrier_is_touched():
    entry = pd.Timestamp('2026-01-01', tz='UTC')
    position = Position('SLP', 1, 10, 100.0, entry, entry_fee=0.0, margin=100_000.0,
                        take_profit=200.0, stop_loss=1.0,
                        hold_until=entry + pd.Timedelta(hours=2))

    assert not resolve_bar(position, _bar(101.0, 99.0, 100.0),
                           entry + pd.Timedelta(hours=1)).exited
    outcome = resolve_bar(position, _bar(101.0, 99.0, 100.0),
                          entry + pd.Timedelta(hours=2))
    assert outcome.reason is ExitReason.HORIZON


def test_barriers_are_mirrored_for_shorts():
    long_tp, long_sl = barrier_prices(100.0, 0.01, 1, tp_mult=5.0, sl_mult=3.0)
    short_tp, short_sl = barrier_prices(100.0, 0.01, -1, tp_mult=5.0, sl_mult=3.0)

    assert long_tp > 100.0 > long_sl
    assert short_tp < 100.0 < short_sl


def test_liquidation_price_reflects_maintenance_margin():
    position = Position('SLP', 1, 100, 100.0, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=0.0, margin=12_500.0)

    liquidation = position.liquidation_price()

    # Equity at that price should equal maintenance on the notional.
    equity = position.margin + position.unrealised(liquidation)
    maintenance = MAINTENANCE_MARGIN_FRACTION * position.notional(liquidation)
    assert equity == pytest.approx(maintenance, rel=1e-6)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_kelly_is_fractional_and_capped():
    """Full Kelly overbets whenever the mean is estimated, which it always is."""
    full = 0.02 / (0.05 ** 2)
    sized = fractional_kelly(0.02, 0.05, fraction=0.25, cap=0.25)

    assert sized < full
    assert sized <= 0.25


def test_no_size_without_a_positive_edge():
    assert fractional_kelly(-0.01, 0.05) == 0.0
    assert fractional_kelly(0.01, 0.0) == 0.0


def test_size_returns_zero_below_one_contract(config):
    """A real constraint for a small account on an expensive contract."""
    contracts = size_from_forecast(
        equity=50.0, price=60_000.0, symbol='BIP',
        expected_return=0.01, sigma=0.02, config=config,
    )

    assert contracts == 0


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _forecast(**overrides) -> pd.Series:
    base = {'price': 0.01, 'carry': 0.0, 'cost': 0.0005, 'sigma': 0.02,
            'side': 1.0, 'expected_net': 0.0095, 'edge_to_risk': 0.475}
    base.update(overrides)
    return pd.Series(base)


def _context(**overrides) -> DecisionContext:
    base = {'equity': 100_000.0, 'volatility': 0.02, 'bar_volume': 1e7, 'price': 150.0}
    base.update(overrides)
    return DecisionContext(**base)


def test_zero_valued_components_are_not_read_as_missing(config):
    """Zero is falsy, and `forecast.get(k) or default` treated it as absent.

    Zero expected carry is entirely normal, and the bug rejected every such
    forecast as NO_FORECAST — silently, since a rejection looks like a decision.
    """
    decision = decide(
        symbol='SLP', timestamp=pd.Timestamp('2026-01-01', tz='UTC'),
        forecast=_forecast(carry=0.0), context=_context(), config=config,
    )

    assert decision.gate is not Gate.NO_FORECAST
    assert decision.tradeable


def test_a_good_forecast_becomes_a_position(config):
    decision = decide(
        symbol='SLP', timestamp=pd.Timestamp('2026-01-01', tz='UTC'),
        forecast=_forecast(), context=_context(), config=config,
    )

    assert decision.tradeable
    assert decision.side == 1
    assert decision.contracts > 0


def test_every_rejection_is_named(config):
    counter = GateCounter()
    cases = [
        (Gate.NO_FORECAST, _forecast(sigma=np.nan), _context()),
        (Gate.VOLATILITY_REGIME, _forecast(), _context(volatility=5.0)),
        (Gate.RISK_UNAVAILABLE, _forecast(sigma=0.0), _context()),
        (Gate.EDGE_BELOW_COST, _forecast(side=0.0, expected_net=-0.001), _context()),
        (Gate.POSITION_LIMIT, _forecast(), _context(open_positions=9, max_positions=5)),
    ]
    for expected, forecast, context in cases:
        decision = decide(
            symbol='SLP', timestamp=pd.Timestamp('2026-01-01', tz='UTC'),
            forecast=forecast, context=context, config=config, counter=counter,
        )
        assert decision.gate is expected, f'expected {expected}, got {decision.gate}'
        assert not decision.tradeable

    assert counter.evaluated == len(cases)
    assert counter.accepted == 0


def test_participation_limit_blocks_oversized_orders(config):
    """A backtest that trades half a bar's volume is describing a fiction."""
    decision = decide(
        symbol='SLP', timestamp=pd.Timestamp('2026-01-01', tz='UTC'),
        forecast=_forecast(), context=_context(equity=1e9, bar_volume=100.0),
        config=config,
    )

    assert decision.gate is Gate.PARTICIPATION_LIMIT


def test_cooldown_is_respected(config):
    decision = decide(
        symbol='SLP', timestamp=pd.Timestamp('2026-01-01', tz='UTC'),
        forecast=_forecast(), context=_context(bars_since_exit=1), config=config,
    )

    assert decision.gate is Gate.COOLDOWN


def test_decision_reports_its_carry_share():
    decision = Decision(
        symbol='SLP', timestamp=pd.Timestamp('2026-01-01', tz='UTC'),
        side=-1, contracts=5, expected_price=0.001, expected_carry=0.009,
    )

    assert decision.carry_share == pytest.approx(0.9)


def test_the_equity_curve_and_the_trade_ledger_reconcile(config):
    """The top-level accounting invariant, and nothing was asserting it.

        final_equity - initial_equity == sum(trade.net_pnl)

    Funding is the reason this can silently break. It is applied to equity
    *during* the hold (`equity -= accrue_funding(...)`, one settlement at a time)
    and recorded on the trade *at exit* (`funding_pnl = -position.funding_paid`),
    so the two paths meet only if the sign convention holds at both ends. Every
    piece of that is individually tested — a long pays a positive rate, funding is
    charged on notional rather than margin — and the sum of the pieces was not.

    A dropped or double-counted term here is invisible in every other number: the
    Sharpe comes off the equity curve, the P&L attribution comes off the ledger,
    and both look plausible while disagreeing. Run with funding deliberately far
    from zero, so a term that goes missing has something to be missing.
    """
    from core.backtest import run_backtest

    features, targets, bars_by, funding_by, profiles = _features_targets(
        config, seed_offset=7, drift=0.0004, funding_mean=4e-5,
    )
    model = train_forecast_model(features, targets, config=config, horizon_bars=48)
    assert model is not None

    cost = targets['cost'].reindex(features.index).to_numpy()
    forecasts = model.predict(features, cost=cost)

    initial = 100_000.0
    result = run_backtest(
        forecasts=forecasts, bars_by_symbol=bars_by, funding_by_symbol=funding_by,
        config=config, profiles=profiles, initial_equity=initial, horizon_bars=48,
    )

    assert result.n_trades > 0, 'no trades: the invariant is untested at zero'
    assert abs(result.funding_pnl) > 0.0, (
        'funding never accrued, so the path this test exists for is not exercised'
    )

    walked = result.final_equity - initial
    assert walked == pytest.approx(result.net_pnl, rel=1e-9, abs=1e-6), (
        f'the equity curve moved {walked:,.2f} while the ledger recorded '
        f'{result.net_pnl:,.2f} — a term is dropped or double-counted between '
        f'them (price {result.price_pnl:,.2f}, funding {result.funding_pnl:,.2f}, '
        f'fees {result.fees:,.2f})'
    )

    # And the decomposition the report prints must itself add up.
    assert result.net_pnl == pytest.approx(
        result.price_pnl + result.funding_pnl - result.fees, rel=1e-9, abs=1e-6,
    )


def test_a_stop_fills_at_the_stop_and_that_is_optimistic(config):
    """Recorded as a known understatement rather than left implicit.

    `resolve_bar` exits a stopped position *at* `position.stop_loss`. A real stop
    on a thin nano perp gaps through: the median close-to-next-open move on this
    book is 1.7-28bp, and a stop triggered inside a bar fills wherever the book
    is, not at the trigger. So modelled stop exits are better than reachable ones
    by roughly that gap, on every stopped trade.

    Not corrected, because the correction only makes a losing book lose more and
    the size of it is a modelling choice rather than a measurement. Pinned,
    because the day someone reads a stop-heavy result as tradeable this is the
    caveat they need, and because a change that made stops fill *better* than the
    trigger would be a real bug this catches.
    """
    from core.execution import Position, resolve_bar

    entry, stop = 100.0, 98.0
    position = Position('BIP', 1, 10, entry, pd.Timestamp('2026-01-01', tz='UTC'),
                        entry_fee=0.0, margin=100_000.0,
                        take_profit=110.0, stop_loss=stop)

    # A bar that gaps well through the stop.
    bar = _bar(97.5, 90.0, 95.0)
    outcome = resolve_bar(position, bar, pd.Timestamp('2026-01-02', tz='UTC'))

    assert outcome.exited
    assert outcome.reason is ExitReason.STOP_LOSS
    assert outcome.exit_price == pytest.approx(stop), (
        'a stop that fills anywhere other than its trigger changed behaviour; if '
        'it now fills worse that is a correction, if better it is a bug'
    )
    # The optimism, stated in the units that matter.
    assert outcome.exit_price > float(bar['low']), 'the bar traded through the modelled fill'
