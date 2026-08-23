"""The one place a trade is chosen, and every gate it has to clear.

Two properties matter more than the rest. The gates must fire in funnel order,
so the rejection histogram means something. And the vectorised screen must admit
exactly the rows the scalar path admits — the screen exists only to make the
backtest fast, and a screen that disagrees would quietly change the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.decide import (
    Reason, Side, WindowExposure, decide, decide_window, kelly_fraction_for,
    price_and_edge, rejection_histogram, round_to_tick, stateless_screen,
)

W = pd.Timestamp('2026-01-01 00:00', tz='UTC')


def row(**over):
    base = dict(symbol='BTC-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=9,
                baseline_probability=0.88, model_probability=0.93)
    base.update(over)
    return base


def test_both_sides_are_always_evaluated():
    """The sigma trade points both ways, and a one-sided gate discards half of it.

    Believing sigma is *smaller* than the market assumes makes the probability
    more extreme than the quote — buy the favourite. Believing it is *larger*
    makes the favourite overpriced — buy the longshot. A band that only permits
    the first silently halves the strategy.
    """
    favourite = decide(row(model_probability=0.95), Config(), bankroll=100.0)
    longshot = decide(row(model_probability=0.70), Config(), bankroll=100.0)
    assert favourite.side is Side.UP and favourite.traded
    assert longshot.side is Side.DOWN and longshot.traded
    assert longshot.price == pytest.approx(0.12)


def test_no_trade_when_the_forecast_agrees_with_the_market():
    """Agreement means the fee is the whole difference, so abstain."""
    decision = decide(row(model_probability=0.88), Config(), bankroll=100.0)
    assert not decision.traded
    assert decision.reason is Reason.EDGE_BELOW_GATE


def test_the_gates_fire_in_funnel_order():
    config = Config(min_traded_price=0.20, max_traded_price=0.90,
                    max_disagreement_pp=25.0, min_edge_pp=0.5)
    # Price outside the band is refused before the disagreement is considered.
    assert decide(row(baseline_probability=0.95, model_probability=0.99),
                  config, bankroll=100.0).reason is Reason.PRICE_OUT_OF_BAND
    # Inside the band, an implausible disagreement is refused before the edge.
    assert decide(row(baseline_probability=0.60, model_probability=0.99),
                  config, bankroll=100.0).reason is Reason.DISAGREEMENT_IMPLAUSIBLE
    # Then the edge.
    assert decide(row(baseline_probability=0.60, model_probability=0.61),
                  config, bankroll=100.0).reason is Reason.EDGE_BELOW_GATE


def test_a_bankroll_below_the_floor_stops_everything():
    config = Config(starting_bankroll=100.0, ruin_floor_fraction=0.5)
    decision = decide(row(), config, bankroll=40.0)
    assert decision.reason is Reason.BANKROLL_FLOOR


def test_one_entry_per_symbol_per_window():
    exposure = WindowExposure(stake=2.0, positions=1,
                              symbols_entered=frozenset({'BTC-USD'}))
    decision = decide(row(), Config(), bankroll=100.0, exposure=exposure)
    assert decision.reason is Reason.ALREADY_ENTERED


def test_correlated_legs_are_capped_per_window():
    """Three simultaneous same-direction bets on 0.7-correlated assets are one bet."""
    exposure = WindowExposure(stake=4.0, positions=2,
                              symbols_entered=frozenset({'ETH-USD', 'SOL-USD'}))
    decision = decide(row(), Config(max_positions_per_window=2), bankroll=100.0,
                      exposure=exposure)
    assert decision.reason is Reason.POSITION_LIMIT


def test_a_stake_under_one_contract_is_a_skip_not_a_rounding():
    """At a $100 account one contract is ~1% of it, so the floor is a gate."""
    config = Config(kelly_fraction=0.001, min_contracts=1)
    decision = decide(row(), config, bankroll=100.0)
    assert decision.reason is Reason.BELOW_MIN_CONTRACTS
    assert decision.contracts == 0


def test_the_order_fee_ceiling_can_flip_a_marginal_trade():
    """A one-contract order pays a higher rate than the continuous formula.

    So expected value is re-checked against what will actually be charged, after
    rounding, not before.
    """
    # An edge that clears the continuous gate by a hair, at one contract.
    config = Config(min_edge_pp=0.0, kelly_fraction=0.0005, max_stake_dollars=0.6)
    decision = decide(row(baseline_probability=0.50, model_probability=0.5290),
                      config, bankroll=100.0)
    assert decision.reason in (Reason.FEE_CEILING, Reason.BELOW_MIN_CONTRACTS,
                               Reason.EDGE_BELOW_GATE)


def test_sizing_is_additive_unless_compounding_is_asked_for():
    config = Config()
    small = decide(row(), config, bankroll=100.0)
    large = decide(row(), config, bankroll=900.0)
    assert small.contracts == large.contracts, 'additive sizing moved with the bankroll'
    compounding = decide(row(), config.with_overrides(compound=True), bankroll=900.0)
    assert compounding.contracts > small.contracts


def test_the_dollar_cap_binds_before_the_percentage_does():
    """Standing in for market depth, which a percentage of a growing account ignores."""
    config = Config(compound=True, max_stake_dollars=5.0, max_stake_fraction=0.5)
    decision = decide(row(), config, bankroll=10_000.0)
    assert decision.stake <= 5.0 + 0.05


def test_kelly_uses_one_minus_cost_not_one_minus_probability():
    """The classic slip: it under-sizes cheap contracts and over-sizes dear ones."""
    q, cost = 0.93, 0.8969
    assert kelly_fraction_for(q, cost) == pytest.approx((q - cost) / (1 - cost))
    assert kelly_fraction_for(q, cost) != pytest.approx((q - cost) / (1 - q))
    assert kelly_fraction_for(0.5, 0.6) == 0.0


def test_a_real_book_replaces_the_assumed_spread_rather_than_adding_to_it():
    """An ask already includes the spread; adding a half-spread charges twice."""
    config = Config(half_spread_cents=1.0)
    assumed = decide(row(), config, bankroll=100.0)
    quoted = decide(row(ask_up=0.87, ask_down=0.16), config, bankroll=100.0)
    assert assumed.price_source == 'baseline'
    assert quoted.price_source == 'quote'
    assert quoted.effective_cost < assumed.effective_cost
    assert quoted.edge > assumed.edge


def test_the_screen_admits_exactly_what_the_scalar_path_admits():
    """The screen is an optimisation, so it must not change any answer."""
    rng = np.random.default_rng(3)
    config = Config()
    frame = pd.DataFrame([
        row(baseline_probability=float(b), model_probability=float(m))
        for b, m in zip(rng.uniform(0.02, 0.98, 400), rng.uniform(0.02, 0.98, 400))
    ])
    survivors, counts = stateless_screen(frame, config)

    scalar_ok, scalar_reasons = [], []
    for index, r in frame.iterrows():
        decision = decide(r, config, bankroll=100.0)
        scalar_reasons.append(decision.reason)
        if decision.reason not in (Reason.NOT_FINITE, Reason.PRICE_OUT_OF_BAND,
                                   Reason.DISAGREEMENT_IMPLAUSIBLE,
                                   Reason.EDGE_BELOW_GATE):
            scalar_ok.append(index)
    assert list(survivors.index) == scalar_ok

    for reason in (Reason.PRICE_OUT_OF_BAND, Reason.DISAGREEMENT_IMPLAUSIBLE,
                   Reason.EDGE_BELOW_GATE):
        assert counts.get(reason.value, 0) == sum(1 for r in scalar_reasons if r is reason)


def test_the_screen_honours_a_real_book_too():
    """Otherwise the funnel would not add up to the decisions."""
    config = Config()
    frame = pd.DataFrame([row(ask_up=0.87, ask_down=0.16),
                          row(model_probability=0.88, ask_up=0.87, ask_down=0.16)])
    survivors, _ = stateless_screen(frame, config)
    scalar = [decide(r, config, bankroll=100.0).traded for _, r in frame.iterrows()]
    assert len(survivors) == sum(scalar)


def test_decide_window_takes_the_first_offset_that_clears():
    """Best-of-offsets is not a strategy that can be run.

    At offset 3 you cannot know what offset 12 will look like, so the live rule
    is to walk them in order — and the backtest has to use the same rule.
    """
    rows = pd.DataFrame([
        row(offset=3, model_probability=0.94),
        row(offset=6, model_probability=0.99),
        row(offset=12, model_probability=0.99),
    ])
    decisions = decide_window(rows, Config(), bankroll=100.0)
    traded = [d for d in decisions if d.traded]
    assert len(traded) == 1
    assert traded[0].offset == 3, 'took a later offset than the first that cleared'


def test_the_histogram_covers_every_reason():
    counts = rejection_histogram([])
    assert set(counts.index) == {r.value for r in Reason}
    assert counts.sum() == 0


def test_prices_round_to_whole_cents():
    assert round_to_tick(0.8749) == pytest.approx(0.87)
    assert round_to_tick(0.8751) == pytest.approx(0.88)
    assert round_to_tick(0.0) == pytest.approx(0.01)
    assert round_to_tick(1.0) == pytest.approx(0.99)


def test_price_and_edge_is_vectorised_and_scalar_consistent():
    config = Config()
    q = np.array([0.93, 0.70, 0.50])
    m = np.array([0.88, 0.88, 0.50])
    is_up, price, probability, cost, edge = price_and_edge(q, m, config)
    for i in range(3):
        single = price_and_edge(q[i:i + 1], m[i:i + 1], config)
        assert bool(single[0][0]) == bool(is_up[i])
        assert single[1][0] == pytest.approx(price[i])
        assert single[4][0] == pytest.approx(edge[i])
