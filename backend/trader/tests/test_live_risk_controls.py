"""Tests for the live risk controls: circuit breakers, stale data, and `decide()` refusals.

The halt state lives on the serving account, so it survives a restart and shows
on the dashboard.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.decide import Reason, Side, WindowExposure, decide
from core.pg_writer import PgWriter
from scripts.live import check_circuit_breakers, stale_symbols

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
CONFIG = Config()


@pytest.fixture
def url(tmp_path) -> str:
    return f'sqlite:///{tmp_path}/serving.db'


@pytest.fixture
def writer(url) -> PgWriter:
    w = PgWriter(database_url=url)
    w.ensure_account(CONFIG.starting_bankroll, mode='paper')
    return w


def settle(w: PgWriter, *, index: int, pnl: float, when: datetime) -> None:
    """Book and resolve one position with a chosen PnL."""
    window = when - timedelta(minutes=15 * (index + 1))
    position_id = w.open_position(
        symbol=f'SYM{index}-USD', window_open=window,
        settle_time=window + timedelta(minutes=15), offset_minutes=3,
        side='up', contracts=10, price=0.5, outlay=abs(pnl) if pnl < 0 else 10 - pnl,
        fee=0.02, model_probability=0.6, baseline_probability=0.5, edge=0.01)
    assert position_id is not None
    # `settle_position` derives the payout from side and outcome, so the outcome steers
    # the PnL.
    w.settle_position(position_id, settled_up=pnl > 0)


class TestCircuitBreakers:
    def test_a_clean_account_is_not_halted(self, writer):
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None
        assert not writer.account().halted

    def test_a_losing_day_halts_the_account(self, writer):
        # Each loss is the full outlay. Twelve $2 losses is 24% of a $100 start, over
        # the 15% daily limit.
        for i in range(12):
            settle(writer, index=i, pnl=-2.0, when=NOW)
        reason = check_circuit_breakers(writer, CONFIG, now=NOW)
        assert reason is not None and 'limit' in reason
        assert writer.account().halted
        assert writer.account().halted_reason

    def test_a_losing_streak_halts_the_account(self, writer):
        config = Config(max_daily_loss_fraction=0.99, max_consecutive_losses=4)
        for i in range(4):
            settle(writer, index=i, pnl=-0.10, when=NOW)
        reason = check_circuit_breakers(writer, config, now=NOW)
        assert reason is not None and 'consecutive' in reason
        assert writer.account().halted

    def test_a_win_breaks_the_streak(self, writer):
        config = Config(max_daily_loss_fraction=0.99, max_consecutive_losses=3)
        for i in range(3):
            settle(writer, index=i, pnl=-0.10, when=NOW)
        settle(writer, index=99, pnl=+0.10, when=NOW)  # most recent
        assert check_circuit_breakers(writer, config, now=NOW) is None

    def test_a_halt_survives_a_restart(self, writer, url):
        """A halt is stored on the account, so it survives a restart."""
        writer.update_account(halted=True, halted_reason='set by hand')
        restarted = PgWriter(database_url=url)
        assert check_circuit_breakers(restarted, CONFIG, now=NOW) == 'set by hand'
        assert restarted.account().halted

    def test_a_non_finite_bankroll_halts(self, writer):
        """A non-finite bankroll halts.

        Postgres can store NaN but SQLite cannot, so the guard is checked on the account
        object directly.
        """
        real = writer.account()

        class NanBankroll:
            halted = False
            halted_reason = None
            bankroll = float('nan')
            id = real.id

        writer.account = lambda: NanBankroll()      # type: ignore[method-assign]
        reason = check_circuit_breakers(writer, CONFIG, now=NOW)
        assert reason is not None and 'finite' in reason


class TestStaleness:
    def _bars(self, newest: pd.Timestamp) -> pd.DataFrame:
        index = pd.date_range(newest - pd.Timedelta(minutes=30), newest,
                              freq='1min', tz='UTC')
        return pd.DataFrame({'event_time': index, 'open': 1.0, 'high': 1.0,
                             'low': 1.0, 'close': 1.0, 'volume': 1.0})

    def test_a_fresh_universe_is_accepted(self):
        newest = pd.Timestamp(NOW) - pd.Timedelta(minutes=1)
        bars = {s: self._bars(newest) for s in CONFIG.symbols}
        assert stale_symbols(bars, CONFIG, now=NOW) == {}

    def test_an_old_feed_is_named(self):
        fresh = pd.Timestamp(NOW) - pd.Timedelta(minutes=1)
        bars = {s: self._bars(fresh) for s in CONFIG.symbols}
        first = sorted(CONFIG.symbols)[0]
        bars[first] = self._bars(pd.Timestamp(NOW) - pd.Timedelta(minutes=20))
        stale = stale_symbols(bars, CONFIG, now=NOW)
        assert set(stale) == {first}
        assert 'old' in stale[first]

    def test_a_missing_symbol_is_named(self):
        """A missing symbol is named: the symbol set defines every cross-asset feature.
        """
        newest = pd.Timestamp(NOW) - pd.Timedelta(minutes=1)
        bars = {s: self._bars(newest) for s in CONFIG.symbols}
        dropped = sorted(CONFIG.symbols)[-1]
        del bars[dropped]
        stale = stale_symbols(bars, CONFIG, now=NOW)
        assert set(stale) == {dropped}
        assert 'no bars' in stale[dropped]


class TestDecideRefusals:
    def row(self, **over):
        base = dict(symbol='BTC-USD',
                    window_open=pd.Timestamp('2026-08-23 00:30', tz='UTC'),
                    settle_time=pd.Timestamp('2026-08-23 00:45', tz='UTC'),
                    offset=3, model_probability=0.72, baseline_probability=0.60)
        base.update(over)
        return base

    @pytest.mark.parametrize('q', (1.02, 1.20, -0.05, 2.0))
    def test_a_probability_outside_zero_one_is_refused(self, q):
        """A probability outside [0, 1] is refused before Kelly sizes it."""
        d = decide(self.row(model_probability=q), CONFIG, bankroll=100.0,
                   exposure=WindowExposure())
        assert d.reason is Reason.PROBABILITY_INVALID
        assert d.contracts == 0

    def test_a_non_finite_bankroll_is_refused(self):
        """A NaN bankroll is refused (`nan < floor` is False)."""
        d = decide(self.row(), CONFIG, bankroll=float('nan'),
                   exposure=WindowExposure())
        assert d.reason is Reason.BANKROLL_FLOOR

    def test_live_refuses_to_price_from_our_own_baseline(self):
        """With `require_quote=True` a row with no book is refused instead of priced at the
        baseline.
        """
        without = decide(self.row(), CONFIG, bankroll=100.0,
                         exposure=WindowExposure(), require_quote=True)
        assert without.reason is Reason.NO_QUOTE

        withbook = decide(self.row(ask_up=0.60, ask_down=0.40), CONFIG,
                          bankroll=100.0, exposure=WindowExposure(),
                          require_quote=True)
        assert withbook.traded and withbook.price_source == 'quote'

    def test_the_half_spread_is_not_charged_on_top_of_a_real_ask(self):
        """No half-spread is charged on top of a real ask, which already includes the
        spread.
        """
        from core.costs import trade_fee

        quoted = decide(self.row(ask_up=0.60, ask_down=0.40), CONFIG,
                        bankroll=100.0, exposure=WindowExposure())
        assert quoted.traded
        cash = quoted.contracts * quoted.price + float(
            trade_fee(quoted.contracts, quoted.price, CONFIG))
        assert quoted.stake == pytest.approx(cash, abs=1e-9)

        # The counterfactual price is a mid, so crossing it does cost the spread.
        counterfactual = decide(self.row(), CONFIG, bankroll=100.0,
                                exposure=WindowExposure())
        assert counterfactual.traded
        assert counterfactual.stake > (
            counterfactual.contracts * counterfactual.price
            + float(trade_fee(counterfactual.contracts, counterfactual.price, CONFIG)))

    def test_both_probabilities_are_reported_on_the_traded_side(self):
        """Both probabilities are reported for the traded side."""
        d = decide(self.row(model_probability=0.28, baseline_probability=0.40),
                   CONFIG, bankroll=100.0, exposure=WindowExposure())
        assert d.side is Side.DOWN
        assert d.model_probability == pytest.approx(0.72)
        assert d.baseline_probability == pytest.approx(0.60)


class TestMarketBenchmark:
    """`settle_predictions` writes the outcome on every prediction, traded or not.

    The unselected sample is what the market comparison needs.
    """

    def prediction(self, w: PgWriter, *, symbol: str, window: datetime,
                   offset: int, strike: float, mid: float | None) -> None:
        w.write_prediction(
            symbol=symbol, window_open=window,
            settle_time=window + timedelta(minutes=15), offset_minutes=offset,
            decision_time=window + timedelta(minutes=offset), strike=strike,
            last_price=strike, displacement=0.0, sigma_remaining=0.001,
            z_score=0.0, baseline_probability=0.5, model_probability=0.55,
            market_probability=mid, market_ask_up=0.53, market_ask_down=0.49,
            price_source='quote', reason='edge_below_gate', traded=False,
            side=None, price=None, effective_cost=None, edge=None,
            contracts=None, model_version=None)

    def bars(self, symbol: str, minute: datetime, value: float) -> dict:
        return {symbol: pd.DataFrame([{
            'event_time': pd.Timestamp(minute), 'open': value, 'high': value,
            'low': value, 'close': value, 'volume': 1.0}])}

    def test_a_refused_window_still_gets_its_outcome(self, writer):
        """A refused window still gets its outcome."""
        from scripts.live import settle_predictions

        window = NOW - timedelta(minutes=30)
        for offset in (3, 6, 9, 12):
            self.prediction(writer, symbol='BTC-USD', window=window,
                            offset=offset, strike=100.0, mid=0.52)
        settle_minute = window + timedelta(minutes=14)
        filled = settle_predictions(
            writer, self.bars('BTC-USD', settle_minute, 101.0), now=NOW)

        assert filled == 4, f'expected all four offsets filled, got {filled}'
        with writer._session() as session:  # noqa: SLF001
            from core.pg_writer import Prediction
            rows = session.query(Prediction).all()
            assert {r.outcome for r in rows} == {1}
            assert {r.market_probability for r in rows} == {0.52}

    def test_the_outcome_uses_the_trained_rule(self, writer):
        """The outcome uses the training rule: `>=` on the mean of the minute ending at
        settle_time.

        `resolve_window` is shared with `settle_due` so the two cannot drift.
        """
        from scripts.live import settle_predictions

        window = NOW - timedelta(minutes=30)
        self.prediction(writer, symbol='BTC-USD', window=window, offset=3,
                        strike=100.0, mid=0.5)
        settle_predictions(
            writer, self.bars('BTC-USD', window + timedelta(minutes=14), 100.0),
            now=NOW)
        with writer._session() as session:  # noqa: SLF001
            from core.pg_writer import Prediction
            assert session.query(Prediction).one().outcome == 1, (
                'a dead-flat window must pay the up side'
            )

    def test_a_window_with_no_bar_yet_is_left_alone(self, writer):
        from scripts.live import settle_predictions

        window = NOW - timedelta(minutes=30)
        self.prediction(writer, symbol='BTC-USD', window=window, offset=3,
                        strike=100.0, mid=0.5)
        empty = {'BTC-USD': pd.DataFrame(
            columns=['event_time', 'open', 'high', 'low', 'close', 'volume'])}
        assert settle_predictions(writer, empty, now=NOW) == 0
        with writer._session() as session:  # noqa: SLF001
            from core.pg_writer import Prediction
            assert session.query(Prediction).one().outcome is None

    def test_an_unsettled_window_is_not_touched(self, writer):
        """A window whose `settle_time` is in the future is not resolved."""
        from scripts.live import settle_predictions

        future = NOW + timedelta(minutes=30)
        self.prediction(writer, symbol='BTC-USD', window=future, offset=3,
                        strike=100.0, mid=0.5)
        assert settle_predictions(
            writer, self.bars('BTC-USD', future + timedelta(minutes=14), 101.0),
            now=NOW) == 0

    def test_filling_twice_is_a_no_op(self, writer):
        from scripts.live import settle_predictions

        window = NOW - timedelta(minutes=30)
        self.prediction(writer, symbol='BTC-USD', window=window, offset=3,
                        strike=100.0, mid=0.5)
        bars = self.bars('BTC-USD', window + timedelta(minutes=14), 101.0)
        assert settle_predictions(writer, bars, now=NOW) == 1
        assert settle_predictions(writer, bars, now=NOW) == 0


class TestClearingAHalt:
    """`--clear-halt` clears a sticky breaker and requires a reason."""

    def parser(self):
        from scripts.live import build_parser
        return build_parser()

    def test_clearing_requires_a_reason(self):
        """`--clear-halt` needs `--reason`."""
        args = self.parser().parse_args(['--clear-halt'])
        assert args.clear_halt and args.reason is None

    def test_a_reason_is_carried(self):
        args = self.parser().parse_args(
            ['--clear-halt', '--reason', 'investigated the streak'])
        assert args.clear_halt
        assert args.reason == 'investigated the streak'

    def test_a_halt_can_be_cleared_and_the_previous_reason_survives(self, writer):
        """A cleared account is tradeable again, and the previous reason is returned."""
        writer.update_account(halted=True, halted_reason='12 consecutive losses')
        assert writer.account().halted

        previous = writer.account().halted_reason
        writer.update_account(halted=False, halted_reason=None)

        assert previous == '12 consecutive losses'
        assert not writer.account().halted
        assert writer.account().halted_reason is None
        # The breakers do not re-halt a clean account.
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None

    def test_clearing_does_not_touch_the_bankroll(self, writer):
        """Clearing a halt leaves the bankroll unchanged."""
        writer.update_account(bankroll=61.25, halted=True, halted_reason='daily loss')
        writer.update_account(halted=False, halted_reason=None)
        assert writer.account().bankroll == pytest.approx(61.25)


class TestTheDrawdownBreaker:
    """Tests the peak-to-current drawdown breaker on realised equity.

    The daily rule cannot see a run-up and give-back within one UTC day, so a
    separate breaker watches drawdown from the peak.
    """

    def _run_up_then_give_back(self, w: PgWriter, up: float, down: float,
                               when: datetime, *, legs: int = 6) -> None:
        """Win `up` dollars, then lose `down`, in a few large settlements.

        Few legs keep the loss streak under `max_consecutive_losses`, so only the
        drawdown breaker can fire.
        """
        assert legs < CONFIG.max_consecutive_losses, (
            'the loss streak would trip the consecutive-loss breaker and mask '
            'the drawdown breaker this class is about'
        )
        i = 0
        for _ in range(legs):
            settle(w, index=i, pnl=+up / legs, when=when); i += 1
        for _ in range(legs):
            settle(w, index=i, pnl=-down / legs, when=when); i += 1

    def test_a_deep_drawdown_that_is_still_above_water_does_not_halt(self, writer):
        """A deep drawdown while still above the starting bankroll does not halt.

        The stake is already guarded by the ruin floor and the daily limit.
        """
        self._run_up_then_give_back(writer, up=66.0, down=64.0, when=NOW)
        account = writer.account()
        assert float(account.realized_pnl) == pytest.approx(2.0, abs=0.5)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None, (
            'halted an account that is up on the money the user put in'
        )

    def test_the_same_drawdown_below_the_stake_does_halt(self, writer):
        """+$40 then -$50 is 35.7% off the peak with $10 of capital lost, so it halts.

        The -$10 day stays inside the daily limit, so only the drawdown breaker can
        fire.
        """
        self._run_up_then_give_back(writer, up=40.0, down=50.0, when=NOW)
        assert float(writer.account().realized_pnl) == pytest.approx(-10.0, abs=0.5)
        reason = check_circuit_breakers(writer, CONFIG, now=NOW)
        assert reason is not None and 'drawdown' in reason, reason
        assert writer.account().halted

    def test_a_shallow_giveback_does_not_halt(self, writer):
        """+$20 then -$4 is 3.3% off the peak, well under the limit."""
        self._run_up_then_give_back(writer, up=20.0, down=4.0, when=NOW)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None

    def test_it_measures_from_the_peak_and_not_from_the_start(self, writer):
        """Drawdown is measured from the peak, not from the start."""
        self._run_up_then_give_back(writer, up=60.0, down=60.0, when=NOW)
        account = writer.account()
        assert float(account.realized_pnl) == pytest.approx(0.0, abs=0.5)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None, (
            'exactly flat on the stake: nothing of the users money is gone, so '
            'the drawdown is a signal rather than a stop'
        )

    def test_a_steadily_winning_account_is_never_halted(self, writer):
        """The high-water mark tracks upward, so steady wins never halt."""
        for i in range(20):
            settle(writer, index=i, pnl=+2.0, when=NOW)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None


class TestTheHighWaterMark:
    def test_it_is_the_running_maximum_not_the_final_value(self, writer):
        settle(writer, index=0, pnl=+10.0, when=NOW)
        settle(writer, index=1, pnl=+10.0, when=NOW)
        settle(writer, index=2, pnl=-15.0, when=NOW)
        assert writer.realised_high_water() == pytest.approx(20.0, abs=0.5)
        assert float(writer.account().realized_pnl) == pytest.approx(5.0, abs=0.5)

    def test_an_account_that_only_ever_lost_has_a_peak_of_zero(self, writer):
        """An account that only lost has a peak of zero, the starting bankroll."""
        settle(writer, index=0, pnl=-5.0, when=NOW)
        assert writer.realised_high_water() == pytest.approx(0.0, abs=1e-6)

    def test_an_empty_account_has_no_peak(self, writer):
        assert writer.realised_high_water() == 0.0
