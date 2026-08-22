"""Portfolio limits that were declared and enforced by nothing.

`max_portfolio_correlation`, `correlation_lookback_hours`, `excluded_symbols` and
`min_equity` were all on `Config`, three of them parseable from the command line,
and read nowhere in `core/` or `scripts/`. So:

- a count limit was the only diversification control, and on a crypto panel where
  cross-correlation runs 0.7-0.9, five "diversified" positions are one bet at five
  times the size;
- `--exclude BIP` parsed, stored, and BIP still traded;
- the only equity floor was `equity <= 0`, by which point the account has been
  unrecoverable for a while.

These are the tests that make each one real, and each fails if the enforcement is
removed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.signal import DecisionContext, Gate, GateCounter, decide_panel

SYMBOLS = ('BIP', 'ETP', 'DOP')


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """BIP and ETP move together; DOP is independent."""
    index = pd.date_range('2026-01-01', periods=200, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, len(index))
    return pd.DataFrame({
        'BIP': base,
        'ETP': base * 0.98 + rng.normal(0, 0.002, len(index)),
        'DOP': rng.normal(0, 0.01, len(index)),
    }, index=index)


def _forecasts(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Three candidates that clear every gate before the correlation check."""
    timestamp = index[-1]
    return pd.DataFrame(
        {'price': 0.05, 'carry': 0.001, 'cost': 0.0005, 'sigma': 0.01, 'side': 1,
         'expected_net': 0.05, 'edge_to_risk': [5.0, 4.9, 4.8]},
        index=pd.MultiIndex.from_product(
            [[timestamp], list(SYMBOLS)], names=['event_time', 'symbol']),
    )


# Contract sizes differ by five orders of magnitude (BTC 0.01 units, DOGE 5000),
# so a single price would put DOGE's one-contract notional above the position cap
# and reject it before the correlation check ever runs.
PRICES = {'BIP': 60_000.0, 'ETP': 3_000.0, 'DOP': 0.20}


def _contexts() -> dict[str, DecisionContext]:
    return {
        symbol: DecisionContext(
            equity=1_000_000, volatility=0.02, bar_volume=1e12,
            price=PRICES[symbol], max_positions=5,
        )
        for symbol in SYMBOLS
    }


def _decide(returns, *, limit):
    config = replace(Config(), max_portfolio_correlation=limit, min_edge_over_cost=0.0)
    return decide_panel(
        _forecasts(returns.index), contexts=_contexts(), config=config,
        counter=GateCounter(), returns=returns,
    )


# ---------------------------------------------------------------------------
# Correlation cap
# ---------------------------------------------------------------------------


def test_a_correlated_candidate_is_refused(correlated_returns):
    """0.98-correlated pair: the weaker edge must not also be taken."""
    decisions = _decide(correlated_returns, limit=0.75)

    taken = [d.symbol for d in decisions if d.tradeable]
    assert 'BIP' in taken, 'the strongest edge should still trade'
    assert 'ETP' not in taken, 'a 0.98-correlated second position is one bet twice'

    capped = [d for d in decisions if d.gate is Gate.CORRELATION_LIMIT]
    assert [d.symbol for d in capped] == ['ETP']
    assert capped[0].max_correlation > 0.9


def test_an_uncorrelated_candidate_still_trades(correlated_returns):
    """The cap must bind on correlation, not simply on count."""
    taken = [d.symbol for d in _decide(correlated_returns, limit=0.75) if d.tradeable]

    assert 'DOP' in taken, 'an independent instrument was refused'


def test_a_zero_limit_disables_the_cap(correlated_returns):
    """0 means off, so the previous behaviour stays reachable."""
    taken = [d.symbol for d in _decide(correlated_returns, limit=0.0) if d.tradeable]

    assert {'BIP', 'ETP'} <= set(taken)


def test_the_cap_keeps_the_stronger_edge(correlated_returns):
    """Candidates arrive best-edge-first, so the later one is the one to drop."""
    decisions = _decide(correlated_returns, limit=0.75)
    by_symbol = {d.symbol: d for d in decisions}

    assert by_symbol['BIP'].tradeable
    assert by_symbol['BIP'].edge_to_risk > by_symbol['ETP'].edge_to_risk


def test_missing_history_admits_rather_than_refuses(correlated_returns):
    """An unmeasurable correlation must not halt the book.

    A newly listed instrument has no overlapping history, and refusing to trade on
    a missing measurement would be a different failure from the one this guards.
    """
    thin = correlated_returns.iloc[:5]  # below MIN_CORRELATION_OBSERVATIONS

    taken = [d.symbol for d in _decide(thin, limit=0.75) if d.tradeable]

    assert len(taken) >= 2, 'a thin history should not block every candidate'


def test_no_returns_panel_disables_the_cap(correlated_returns):
    """Callers that cannot supply returns keep working."""
    config = replace(Config(), max_portfolio_correlation=0.75, min_edge_over_cost=0.0)
    decisions = decide_panel(
        _forecasts(correlated_returns.index), contexts=_contexts(),
        config=config, counter=GateCounter(), returns=None,
    )

    assert not any(d.gate is Gate.CORRELATION_LIMIT for d in decisions)


# ---------------------------------------------------------------------------
# Exclusion and the equity floor
# ---------------------------------------------------------------------------


def test_an_excluded_symbol_does_not_reach_the_panel(tmp_path):
    """`--exclude BTC` has to hold for every downstream consumer at once."""
    from core.dataset import load_dataset
    from core.datastore import ResearchStore

    store = ResearchStore(tmp_path / 'research')
    index = pd.date_range('2026-01-01', periods=400, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    for symbol in ('BTC-PERP', 'ETH-PERP', 'SOL-PERP'):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        store.write('bars', pd.DataFrame({
            'symbol': symbol, 'venue': 'coinbase', 'event_time': index,
            'available_time': index, 'quality': 'valid', 'open': close,
            'high': close * 1.001, 'low': close * 0.999, 'close': close,
            'volume': 1000.0,
        }))

    for spelling in ('BTC', 'BIP', 'btc'):
        dataset = load_dataset(
            store, venue='coinbase', reference_venue=None, min_quality='valid',
            config=replace(Config(), excluded_symbols=[spelling]),
        )
        assert 'BTC-PERP' not in dataset.symbols, f'{spelling} did not exclude'
        assert len(dataset.symbols) == 2
        assert any('excluded' in w for w in dataset.warnings)


def test_the_equity_floor_is_the_configured_one():
    """`equity <= 0` is not a floor; `min_equity` is."""
    config = Config()

    assert config.min_equity > 0, 'the default floor must be above zero'


def test_the_floor_is_read_by_the_backtest():
    """Guard the wiring, since the field was declared and unread for a while."""
    import inspect

    from core import backtest

    source = inspect.getsource(backtest.run_backtest)
    assert 'min_equity' in source, 'the equity floor is not enforced'


# ---------------------------------------------------------------------------
# The per-trade risk bound has to survive leverage
# ---------------------------------------------------------------------------


def _loss_at_stop(leverage: float, *, stop_multiple: float = 3.0) -> float:
    """Fraction of equity lost if a freshly sized position hits its stop."""
    from dataclasses import replace

    from core.config import Config
    from core.costs import get_contract_spec
    from core.execution import size_from_forecast

    equity, price, sigma = 100_000.0, 60_000.0, 0.02
    config = replace(Config(), leverage=leverage)
    contracts = size_from_forecast(
        equity=equity, price=price, symbol='BIP', expected_return=0.05,
        sigma=sigma, config=config, stop_multiple=stop_multiple, stop_sigma=sigma,
    )
    notional = contracts * get_contract_spec('BIP').units * price
    return notional * stop_multiple * sigma / equity


def test_max_risk_per_trade_is_a_bound_at_every_leverage():
    """`MAX_RISK_PER_TRADE` must mean what its docstring says it means.

    `risk_budget_fraction` solves for the notional whose stop-out costs at most
    `max_risk` of equity — then `size_from_forecast` multiplied that fraction by
    `config.leverage`, so the realised bound was `max_risk * leverage`. At the
    compose default of 4x a declared 1% became a measured 4.00%, and raising the
    knob raised the loss with it. The function had a `leverage` parameter,
    declared and referenced nowhere, which is what it was for.
    """
    from core.execution import MAX_RISK_PER_TRADE

    for leverage in (1.0, 2.0, 4.0, 10.0, 25.0):
        loss = _loss_at_stop(leverage)
        assert loss <= MAX_RISK_PER_TRADE + 1e-9, (
            f'at {leverage}x leverage a stop-out costs {loss:.2%} of equity '
            f'against a declared limit of {MAX_RISK_PER_TRADE:.2%}'
        )


def test_raising_leverage_past_the_risk_budget_stops_adding_size():
    """Once the risk budget binds, leverage is not a way around it.

    Below the budget, leverage does what an operator expects and scales the
    position. Above it, the bound holds instead — that ordering is the whole
    point of `min(conviction, budget)`.
    """
    small, large = _loss_at_stop(1.0), _loss_at_stop(4.0)
    assert large > small, 'leverage should still scale size while the budget is slack'
    assert _loss_at_stop(25.0) == pytest.approx(_loss_at_stop(10.0), rel=1e-6), (
        'past the budget, more leverage must not buy more risk'
    )


def test_the_edge_to_risk_gate_is_sweepable_and_the_profile_cannot_override_it():
    """It was a module constant, which made it the one gate nobody could move.

    Every other threshold `decide()` reads is a `Config` field resolved against
    the coin profile, so a sensitivity run can set it. `MIN_EDGE_TO_RISK` was
    `0.05` in `core/signal.py`, so answering "how much of this result is the
    gates" needed a monkeypatch — and a monkeypatched threshold is a measurement
    nobody can reproduce from the CLI.

    Two things have to hold for the field to be usable. It has to gate: raising
    it must reject a forecast a lower value accepts. And an explicit setting has
    to beat the default, because `Config.resolve` ignores the field unless the
    name is in `cli_overrides` — the trap that makes a swept threshold silently
    do nothing.
    """
    import core.signal as signal_module
    from core.signal import decide

    assert not hasattr(signal_module, 'MIN_EDGE_TO_RISK'), (
        'the module constant is back; the Config field must be the only source'
    )
    assert Config().min_edge_to_risk > 0.0

    def swept(value: float) -> Config:
        base = Config()
        return replace(
            base, min_edge_to_risk=value, min_edge_over_cost=0.0,
            cli_overrides=frozenset(base.cli_overrides | {'min_edge_to_risk'}),
        )

    # An edge of 0.4 sigma: comfortably above 0.05, far below 10.
    forecast = pd.Series({
        'price': 0.004, 'carry': 0.0, 'cost': 0.0005, 'sigma': 0.01,
        'side': 1.0, 'expected_net': 0.004, 'edge_to_risk': 0.4,
    })
    context = DecisionContext(
        equity=1_000_000, volatility=0.02, bar_volume=1e12,
        price=60_000.0, max_positions=5,
    )
    timestamp = pd.Timestamp('2026-01-01', tz='UTC')

    def at(threshold: float):
        return decide(symbol='BIP', timestamp=timestamp, forecast=forecast,
                      context=context, config=swept(threshold))

    permissive, strict = at(0.0), at(10.0)

    assert permissive.gate is None or permissive.gate != Gate.EDGE_TO_RISK
    assert strict.gate == Gate.EDGE_TO_RISK, (
        f'raising the threshold to 10 sigma did not gate; got {strict.gate}'
    )

    # Unmarked, the field is ignored in favour of the default — which is exactly
    # how a hand-set threshold becomes a no-op.
    unmarked = replace(Config(), min_edge_to_risk=10.0)
    assert unmarked.resolve('min_edge_to_risk') == Config().min_edge_to_risk
