"""Tests that a latched breaker stops entries but not scoring and recording.

`decide()` refuses with `Reason.HALTED`, so the row is still scored, written and
settled.
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
        """`decide()` is not halted by default, so the backtest and paper engine are unaffected."""
        assert decide(row(), DEFAULT_CONFIG, bankroll=150.0).reason is Reason.TRADED

    def test_halted_is_distinct_from_the_bankroll_floor(self):
        """A latched breaker (cleared by hand) and the ruin floor are different reasons."""
        floored = decide(row(), DEFAULT_CONFIG, bankroll=1.0)
        assert floored.reason is Reason.BANKROLL_FLOOR
        assert decide(row(), DEFAULT_CONFIG, bankroll=150.0,
                      halted=True).reason is Reason.HALTED

    def test_the_ruin_floor_still_wins_when_both_apply(self):
        """When both apply, the ruin floor is the recorded reason."""
        out = decide(row(), DEFAULT_CONFIG, bankroll=1.0, halted=True)
        assert out.reason is Reason.BANKROLL_FLOOR

    def test_a_halt_does_not_suppress_the_scored_probability(self):
        """A halted row still carries the model and baseline probabilities."""
        out = decide(row(), DEFAULT_CONFIG, bankroll=150.0, halted=True)
        assert out.model_probability == pytest.approx(0.72)
        assert out.baseline_probability == pytest.approx(0.60)


class TestTheCycleKeepsScoringWhileHalted:
    """Runs a real cycle on a halted account with a stubbed scorer."""

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

            def score(self, table):
                return pd.DataFrame({'model_probability': self.predict(table)},
                                    index=table.index)

        async def fake_bars(config, minutes=None, **_):
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
            # The fixture has only a few synthetic bars, so `min_usable_bars` is lowered
            # to
            # reach the halt path. `min_remaining_seconds=0` stops the real clock ending
            # the
            # run in the last 45s of a quarter-hour.
            config = dataclasses.replace(DEFAULT_CONFIG, min_usable_bars=1,
                                         min_remaining_seconds=0)
            asyncio.run(live_mod.run_cycle(args, config, writer,
                                           Model(), None))
        finally:
            mp.undo()
        return writer, calls

    def test_a_halted_account_still_scores_the_window(self, tmp_path):
        _, calls = self._cycle(tmp_path, halt=True)
        assert calls['scored'] == 1, (
            'a halted account skipped scoring, so no quote or outcome is '
            'recorded and the window count freezes'
        )

    def test_a_halted_account_still_writes_the_prediction(self, tmp_path):
        """A halted account still writes the prediction that `market_benchmark` reads."""
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
