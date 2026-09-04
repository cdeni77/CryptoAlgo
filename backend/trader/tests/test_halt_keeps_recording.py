"""A latched breaker must stop the entries and not the measurement.

`run_cycle` used to set `offset = None` on a halt, which returns before
`score_live` — so a halted account wrote no predictions and recorded no quotes.
The breaker exists to stop risking money; freezing the data collection was
collateral damage nobody chose, and the expensive kind.

The arithmetic that makes it expensive, measured on the live account: ~285
symbol-windows recorded per day, 2,000 needed before the market comparison means
anything, and a `max_daily_loss_fraction` of 0.15 on a $154 account is ~$23/day
against a ~$24/day expected burn. So the breaker fires most days, and every day it
fires used to contribute zero windows. The recording run could not reach its own
sample-size target.

`decide()` refuses with `Reason.HALTED` instead, which keeps the row honest: it is
scored, priced against the real book, written with that reason, and settled like
any other.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG, Config
from core.decide import Reason, WindowExposure, decide

WINDOW = pd.Timestamp('2026-08-24 15:00', tz='UTC')


def row(**over) -> dict:
    base = dict(
        symbol='BTC-USD', window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset=3,
        model_probability=0.72, baseline_probability=0.60,
        ask_up=0.60, ask_down=0.41)
    base.update(over)
    return base


class TestTheHaltedGate:
    def test_a_tradeable_row_is_refused_when_halted(self):
        assert decide(row(), DEFAULT_CONFIG, bankroll=150.0).reason is Reason.TRADED
        out = decide(row(), DEFAULT_CONFIG, bankroll=150.0, halted=True)
        assert out.reason is Reason.HALTED

    def test_it_stakes_nothing_and_buys_nothing(self):
        out = decide(row(), DEFAULT_CONFIG, bankroll=150.0, halted=True)
        assert out.contracts == 0
        assert out.stake == 0.0

    def test_the_default_is_not_halted_so_nothing_else_changes(self):
        """The backtest and the paper engine call the same `decide()` and must be
        unaffected — there is one `decide()` on purpose."""
        assert decide(row(), DEFAULT_CONFIG, bankroll=150.0).reason is Reason.TRADED

    def test_halted_is_distinct_from_the_bankroll_floor(self):
        """A latched breaker and a breached ruin floor are different states with
        different remedies: one is cleared by hand, the other by having money."""
        floored = decide(row(), DEFAULT_CONFIG, bankroll=1.0)
        assert floored.reason is Reason.BANKROLL_FLOOR
        assert decide(row(), DEFAULT_CONFIG, bankroll=150.0,
                      halted=True).reason is Reason.HALTED

    def test_the_ruin_floor_still_wins_when_both_apply(self):
        """Ordering: no money is a harder stop than a latched breaker, and the
        recorded reason should name the one that cannot be cleared by hand."""
        out = decide(row(), DEFAULT_CONFIG, bankroll=1.0, halted=True)
        assert out.reason is Reason.BANKROLL_FLOOR

    def test_a_halt_does_not_suppress_the_scored_probability(self):
        """The row must still carry what the measurement needs. A refusal that
        blanked the model probability would record the abstention and lose the
        forecast, which is the whole reason to keep scoring."""
        out = decide(row(), DEFAULT_CONFIG, bankroll=150.0, halted=True)
        assert out.model_probability == pytest.approx(0.72)
        assert out.baseline_probability == pytest.approx(0.60)


class TestTheCycleKeepsScoringWhileHalted:
    """Behaviour, not source text.

    The first version of this asserted that `run_cycle`'s halt branch did not
    contain the string `offset = None`. That is a test of the comment — the
    pattern `tests/test_kalshi.py` was written to shame, and one this repo has
    already been burned by: two tests there passed for the entire time the
    reconciliation they described did not exist. So: stub the scorer, run a real
    cycle against a halted account, and assert the scorer was reached and a
    prediction was written.
    """

    def _cycle(self, tmp_path, *, halt: bool):
        import argparse
        import asyncio
        import dataclasses

        import numpy as np

        from core.pg_writer import PgWriter
        from scripts import live as live_mod

        writer = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
        writer.ensure_account(150.0, mode='paper')
        if halt:
            writer.update_account(halted=True, halted_reason='daily loss')

        now = pd.Timestamp.now('UTC').floor('min')
        window = now.floor('15min')
        # Fresh bars, so `stale_symbols` does not fire and clear the offset for a
        # reason unrelated to the halt.
        stamps = pd.date_range(now - pd.Timedelta(minutes=90), now, freq='1min')
        bars = {s: pd.DataFrame({
            'event_time': stamps, 'open': 100.0, 'high': 100.0, 'low': 100.0,
            'close': 100.0, 'volume': 1.0, 'quote_volume': 100.0,
            'trade_count': 10.0}) for s in DEFAULT_CONFIG.symbols}

        calls = {'scored': 0}

        def fake_score(bars_, scoring, config, *, window_open, offset, groups=None, deferred=()):
            calls['scored'] += 1
            calls['offset'] = offset
            return pd.DataFrame({
                'symbol': ['BTC-USD'], 'window_open': [window_open],
                'settle_time': [window_open + pd.Timedelta(minutes=15)],
                'offset': [offset], 'strike': [100.0], 'last_price': [100.4],
                'displacement': [0.004], 'sigma_remaining': [0.005],
                'z_score': [0.8], 'baseline_probability': [0.60],
                'baseline_probability_logit': [np.log(0.6 / 0.4)],
                'strike_source': [None],
            })

        class Model:
            scoring, groups, init_score_source = object(), (), 'baseline'

            def predict(self, table):
                return np.array([0.72] * len(table))

        async def fake_bars(config, minutes=None):
            return bars

        async def fake_quotes(kalshi, symbols, settle_time):
            return {}

        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        mp.setattr(live_mod, 'fetch_bars', fake_bars)
        mp.setattr(live_mod, 'score_live', fake_score)
        mp.setattr(live_mod, 'fetch_quotes', fake_quotes)
        try:
            args = argparse.Namespace(offset=3, reconcile=False, mode='paper',
                                      place_orders=False, dry_run=False)
            # The fixture carries a handful of synthetic bars, not the 1,455
            # a live feed must supply — `min_usable_bars` guards the band where
            # a partial Coinbase answer produces confident wrong features.
            # Lowered here so the test exercises the halt path rather than the
            # short-feed refusal.
            config = dataclasses.replace(DEFAULT_CONFIG, min_usable_bars=1)
            asyncio.run(live_mod.run_cycle(args, config, writer,
                                           Model(), None))
        finally:
            mp.undo()
        return writer, calls

    def test_a_halted_account_still_scores_the_window(self, tmp_path):
        """The regression. With `offset = None` this was 0."""
        _, calls = self._cycle(tmp_path, halt=True)
        assert calls['scored'] == 1, (
            'a halted account skipped scoring, so no quote or outcome is '
            'recorded and the window count freezes'
        )

    def test_a_halted_account_still_writes_the_prediction(self, tmp_path):
        """What `market_benchmark` actually reads."""
        writer, _ = self._cycle(tmp_path, halt=True)
        rows = writer.scored_against_market()
        assert rows is not None
        with writer._session() as session:      # noqa: SLF001 - same package
            from core.pg_writer import Prediction

            saved = session.query(Prediction).all()
            assert len(saved) == 1
            assert saved[0].reason == 'halted'
            assert not saved[0].traded

    def test_a_halted_account_opens_no_position(self, tmp_path):
        writer, _ = self._cycle(tmp_path, halt=True)
        assert writer.open_positions() == []

    def test_an_unhalted_account_is_unaffected(self, tmp_path):
        writer, calls = self._cycle(tmp_path, halt=False)
        assert calls['scored'] == 1
        with writer._session() as session:      # noqa: SLF001
            from core.pg_writer import Prediction

            saved = session.query(Prediction).all()
            assert len(saved) == 1 and saved[0].reason != 'halted'
