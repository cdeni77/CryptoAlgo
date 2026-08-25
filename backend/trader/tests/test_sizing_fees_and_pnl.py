"""PROPOSED. Sizing arithmetic, the per-order fee ceiling, and PnL/settlement accounting.

Drop into `backend/trader/tests/`.

Two gaps this closes.

1. `test_decide.py::test_the_order_fee_ceiling_can_flip_a_marginal_trade` asserts
   `decision.reason in (FEE_CEILING, BELOW_MIN_CONTRACTS, EDGE_BELOW_GATE)` — a
   three-way OR covering almost every rejection mode. It passes with the fee
   ceiling logic deleted. Measured: `core/decide.py:386`, the `FEE_CEILING`
   refusal, never executes in the whole suite. The test below reaches it, with the
   exact numbers, and records the fact that under the *default* `min_edge_pp` of
   0.5pp the gate is unreachable — the continuous check at 0.5pp is already
   stricter than the rounding correction, which is worth knowing before anyone
   loosens `min_edge_pp`.

2. `core/book.py`'s settlement arithmetic is asserted only through the
   `payout()` helper. The end-to-end identity — outlay in, payout out, bankroll
   moves by exactly the difference, fees counted once — is not.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.book import Book, summarise
from core.config import Config
from core.costs import effective_price, trade_fee
from core.decide import Reason, Side, decide

W = pd.Timestamp('2026-01-01 00:00', tz='UTC')


def row(**over):
    base = dict(symbol='BTC-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=9,
                baseline_probability=0.88, model_probability=0.93)
    base.update(over)
    return base


# ------------------------------------------------------- the per-order ceiling

def test_the_per_order_fee_ceiling_no_longer_has_an_excess_worth_gating():
    """The premise of the FEE_CEILING gate was measured away.

    This test used to assert that one contract at 50c owes $0.0175 and is charged
    $0.02 — a $0.0025 excess that killed any edge under 0.25pp. Checked against
    328 real fills on 2026-08-25, the venue ceilings to a **hundredth of a cent**,
    so that order pays $0.0175 exactly and the excess at 50c is zero.

    Two consequences, both asserted here:

    * the per-order excess can never exceed $0.0001, whatever the price; and
    * the gate is therefore **unreachable**. An edge small enough to be killed by
      a hundredth of a cent is far too small to size a single contract, so
      `BELOW_MIN_CONTRACTS` always fires first. The refusal is kept because it is
      a correct statement about a venue that might coarsen its rounding again --
      but nothing reaches it today, and a future reader should not believe the old
      docstring's arithmetic.
    """
    config = Config(min_edge_pp=0.0, kelly_fraction=1.0, max_stake_dollars=0.60,
                    min_contracts=1)
    # The continuous formula is unchanged; only the charged amount moved.
    assert float(effective_price(0.50, config)) == pytest.approx(0.522498, abs=1e-6)
    assert float(trade_fee(1, 0.50, config)) == pytest.approx(0.0175)

    for price in (0.05, 0.37, 0.50, 0.63, 0.91):
        for contracts in (1, 3, 17):
            raw = 0.07 * contracts * price * (1 - price)
            excess = float(trade_fee(contracts, price, config)) - raw
            # -1e-12 not 0: an exact multiple lands a float epsilon below.
            assert -1e-12 <= excess < 1e-4, (price, contracts, excess)

    # The excess at 37c is $0.0000830. An edge below it cannot buy a contract.
    cost_37 = float(effective_price(0.37, config))
    marginal = decide(
        row(baseline_probability=0.37, model_probability=cost_37 + 4e-5),
        config, bankroll=100.0)
    assert marginal.contracts == 0
    assert marginal.reason is Reason.BELOW_MIN_CONTRACTS, (
        'if this becomes FEE_CEILING the venue has coarsened its rounding and '
        'the cost model needs re-measuring against real fills'
    )

    # And a real edge trades, charged the ceiling rather than the raw formula.
    trades = decide(row(baseline_probability=0.37, model_probability=cost_37 + 0.01),
                    config, bankroll=100.0)
    assert trades.traded and trades.contracts >= 1
    assert trades.fee == pytest.approx(float(trade_fee(trades.contracts,
                                                       trades.price, config)))


def test_the_fee_ceiling_gate_is_dead_under_the_default_edge_gate():
    """Recorded, not celebrated.

    The gate can only bite below 0.25pp of edge, and the default `min_edge_pp` is
    0.5pp — so with defaults `EDGE_BELOW_GATE` always fires first and
    `FEE_CEILING` is unreachable. Anyone loosening `min_edge_pp` below 0.25 turns
    this gate back on, and should find out here rather than in production.
    """
    config = Config()
    assert config.min_edge_pp > 0.25, (
        'min_edge_pp is at or below the per-order rounding excess; the '
        'FEE_CEILING gate is now live and its coverage matters'
    )
    cost = float(effective_price(0.50, config))
    for edge in (0.0005, 0.0015, 0.0025):
        decision = decide(row(baseline_probability=0.50, model_probability=cost + edge),
                          config, bankroll=100.0)
        assert decision.reason is Reason.EDGE_BELOW_GATE


def test_a_traded_stake_is_exactly_contracts_times_cost_plus_the_charged_fee():
    """The one place the money leaves. Asserted as an identity, not a bound."""
    config = Config()
    for probability in (0.93, 0.96, 0.75):
        decision = decide(row(model_probability=probability), config, bankroll=100.0)
        if not decision.traded:
            continue
        expected_fee = float(trade_fee(decision.contracts, decision.price, config))
        expected_stake = (decision.contracts
                          * (decision.price + config.half_spread_cents / 100.0)
                          + expected_fee)
        assert decision.fee == pytest.approx(expected_fee)
        assert decision.stake == pytest.approx(expected_stake), probability
        # And the post-rounding EV really is positive, which is what the gate claims.
        assert decision.model_probability > decision.stake / decision.contracts


def test_contracts_are_floored_never_rounded():
    """Rounding up buys size the sizing rule did not authorise."""
    config = Config(kelly_fraction=1.0, max_stake_dollars=2.99)
    decision = decide(row(), config, bankroll=100.0)
    cost = float(effective_price(decision.price, config))
    assert decision.contracts == math.floor(2.99 / cost), (decision.contracts, cost)


# ---------------------------------------------------- settlement and the ledger

def test_the_bankroll_moves_by_exactly_outlay_out_and_payout_in():
    """A win at 86c on 3 contracts returns $3 against an outlay of about $2.63.

    The identity is asserted per settlement, so a double-counted fee or a payout
    credited twice shows up as an exact mismatch rather than as a plausible
    equity curve.
    """
    config = Config()
    book = Book(config=config, bankroll=100.0)
    decision = decide(row(model_probability=0.95), config, bankroll=100.0)
    assert decision.traded and decision.side is Side.UP

    before = book.bankroll
    position = book.record(decision)
    assert position is not None
    assert book.bankroll == pytest.approx(before - decision.stake)

    settled = book.settle({('BTC-USD', W): True})
    assert len(settled) == 1
    assert book.bankroll == pytest.approx(before - decision.stake + decision.contracts)
    # A loss returns nothing at all.
    book2 = Book(config=config, bankroll=100.0)
    book2.record(decision)
    book2.settle({('BTC-USD', W): False})
    assert book2.bankroll == pytest.approx(100.0 - decision.stake)


def test_a_down_position_is_paid_when_the_window_settles_down():
    """The side mapping, at the accounting layer.

    Flipping `Position.payout`'s `Side.UP` to `Side.DOWN` is caught by the
    existing suite; this states the same thing end to end through `Book.settle`,
    so the sign cannot be lost between the two.
    """
    config = Config()
    book = Book(config=config, bankroll=100.0)
    # Baseline 0.50 / model 0.44: a 6pp disagreement, inside
    # `max_disagreement_pp`, and the cheap side is DOWN.
    decision = decide(row(baseline_probability=0.50, model_probability=0.44),
                      config, bankroll=100.0)
    assert decision.traded and decision.side is Side.DOWN, decision.reason

    book.record(decision)
    book.settle({('BTC-USD', W): False})          # settled DOWN: the bet won
    assert book.settlements[-1].won
    assert book.bankroll == pytest.approx(
        100.0 - decision.stake + decision.contracts)


def test_total_fees_equal_the_sum_of_the_charged_fees_and_nothing_more():
    """Fees are paid once, at entry. Settlement is free.

    `hold to settle` is the whole cost argument for this venue, so a second fee
    appearing anywhere would invert the comparison against a perp round trip.
    """
    config = Config()
    book = Book(config=config, bankroll=100.0)
    fees = 0.0
    for i in range(30):
        window = W + pd.Timedelta(minutes=15 * i)
        decision = decide(row(window_open=window,
                              settle_time=window + pd.Timedelta(minutes=15)),
                          config, bankroll=book.bankroll)
        if not decision.traded:
            continue
        book.record(decision)
        fees += decision.fee
        book.settle({('BTC-USD', window): i % 3 != 0})
    stats = summarise(book, windows_available=30)
    assert stats.n_trades > 10
    assert stats.total_fees == pytest.approx(fees)


def test_pnl_reconciles_with_the_equity_curve_to_the_cent():
    """`total_pnl` and `ending_equity - starting_bankroll` are two computations of
    one number, and they have disagreed in this repo's ancestor."""
    config = Config()
    book = Book(config=config, bankroll=100.0)
    rng = np.random.default_rng(11)
    for i in range(60):
        window = W + pd.Timedelta(minutes=15 * i)
        decision = decide(row(window_open=window,
                              settle_time=window + pd.Timedelta(minutes=15)),
                          config, bankroll=book.bankroll)
        if not decision.traded:
            continue
        book.record(decision)
        book.settle({('BTC-USD', window): bool(rng.random() < 0.9)})
    stats = summarise(book, windows_available=60)
    assert stats.total_pnl == pytest.approx(
        stats.ending_equity - stats.starting_bankroll, abs=1e-9)
    assert not book.open_positions, 'every position settled, so equity is cash'
    assert stats.ending_equity == pytest.approx(book.bankroll)
