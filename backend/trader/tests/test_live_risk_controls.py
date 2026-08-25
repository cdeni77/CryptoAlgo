"""The live path's refusals, which mostly did not exist.

`Account.halted` and `halted_reason` were columns on the serving table, rendered
on the dashboard as a safety chip, and written by nothing — the only code that
set them was `core/book.py`, the *backtest's* in-memory account. So the
indicator could never turn on, and the only live limit was the ruin floor, which
fires after half the account is gone.
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
    # `settle_position` derives the payout from the side and the outcome, so
    # steer it with the outcome rather than by writing pnl directly.
    w.settle_position(position_id, settled_up=pnl > 0)


class TestCircuitBreakers:
    def test_a_clean_account_is_not_halted(self, writer):
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None
        assert not writer.account().halted

    def test_a_losing_day_halts_the_account(self, writer):
        # Each loss is the full outlay. Twelve $2 losses is 24% of a $100 start,
        # over the 15% daily limit.
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
        settle(writer, index=99, pnl=+0.10, when=NOW)     # most recent
        assert check_circuit_breakers(writer, config, now=NOW) is None

    def test_a_halt_survives_a_restart(self, writer, url):
        """The whole point of putting it on the account rather than in memory."""
        writer.update_account(halted=True, halted_reason='set by hand')
        restarted = PgWriter(database_url=url)
        assert check_circuit_breakers(restarted, CONFIG, now=NOW) == 'set by hand'
        assert restarted.account().halted

    def test_a_non_finite_bankroll_halts(self, writer):
        """Reachable on Postgres, not through SQLite.

        `double precision NOT NULL` accepts NaN; SQLite coerces it to NULL and
        rejects it, so the column cannot hold the value this guard is for. The
        route in is `reconcile_with_venue` writing the venue's balance —
        `KalshiClient.balance()` returns 0.0 on a missing field and parses the
        string "nan" happily. So the guard is checked against the account object
        directly rather than through a store that cannot represent the problem.
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
        """Not merely one fewer row — the universe defines every cross-asset
        feature, so a short universe is a different model. Measured: dropping a
        symbol moved `beta_1440` 7.7x with no error and no NaN."""
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
        """Kelly does not object to q > 1: q=1.20 sized at 7.1x bankroll, and
        only `max_stake_fraction` stood between that and an order."""
        d = decide(self.row(model_probability=q), CONFIG, bankroll=100.0,
                   exposure=WindowExposure())
        assert d.reason is Reason.PROBABILITY_INVALID
        assert d.contracts == 0

    def test_a_non_finite_bankroll_is_refused(self):
        """`nan < floor` is False, so a NaN bankroll used to be sized."""
        d = decide(self.row(), CONFIG, bankroll=float('nan'),
                   exposure=WindowExposure())
        assert d.reason is Reason.BANKROLL_FLOOR

    def test_live_refuses_to_price_from_our_own_baseline(self):
        """With no book, `decide` falls back to the counterfactual price. That is
        right in a backtest and wrong live, where it booked a position for an
        order that could not be sent."""
        without = decide(self.row(), CONFIG, bankroll=100.0,
                         exposure=WindowExposure(), require_quote=True)
        assert without.reason is Reason.NO_QUOTE

        withbook = decide(self.row(ask_up=0.60, ask_down=0.40), CONFIG,
                          bankroll=100.0, exposure=WindowExposure(),
                          require_quote=True)
        assert withbook.traded and withbook.price_source == 'quote'

    def test_the_half_spread_is_not_charged_on_top_of_a_real_ask(self):
        """The ask already includes the spread. Charging it again debited
        $0.005/contract that never left the account, which is exactly the balance
        drift the operator is told to read as an unrecorded fill."""
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
        """A DOWN trade stored P(down) beside P(up), so their difference — which
        reads like the disagreement being traded — was meaningless."""
        d = decide(self.row(model_probability=0.28, baseline_probability=0.40),
                   CONFIG, bankroll=100.0, exposure=WindowExposure())
        assert d.side is Side.DOWN
        assert d.model_probability == pytest.approx(0.72)
        assert d.baseline_probability == pytest.approx(0.60)


class TestMarketBenchmark:
    """The recorded quote is worthless without the answer beside it.

    `market_probability` was written on every decision and read by nothing, there
    was no `outcome` column on `predictions` at all, and `positions` — the only
    place an outcome existed — covers just the windows that traded. So the one
    economically meaningful question, "is the market's probability better than
    ours", could not be asked of an unselected sample.
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
        """The whole point: the unselected sample is the useful one."""
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
        """`>=` on the mean of the minute ENDING at settle_time.

        A window that lands exactly on its strike pays the up side, and the bar
        read must be the one before `settle_time` — not the one starting there.
        `resolve_window` is shared with `settle_due` so the two cannot drift, which
        is how they drifted before.
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
        """`settle_time` in the future must not be resolved from a stray bar."""
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
    """A sticky breaker with no way to clear it is a trap.

    The breakers are deliberately sticky — one that resets itself at midnight is a
    speed bump rather than a breaker — but for a while there was no CLI to clear
    one, so a halt overnight left no documented recovery short of hand-editing the
    database. `--clear-halt` closes that, and requires a reason.
    """

    def parser(self):
        from scripts.live import build_parser
        return build_parser()

    def test_clearing_requires_a_reason(self):
        """A breaker cleared without a recorded cause is one nobody learns from."""
        args = self.parser().parse_args(['--clear-halt'])
        assert args.clear_halt and args.reason is None

    def test_a_reason_is_carried(self):
        args = self.parser().parse_args(
            ['--clear-halt', '--reason', 'investigated the streak'])
        assert args.clear_halt
        assert args.reason == 'investigated the streak'

    def test_a_halt_can_be_cleared_and_the_previous_reason_survives(self, writer):
        """The account must come back tradeable, and say what stopped it."""
        writer.update_account(halted=True, halted_reason='12 consecutive losses')
        assert writer.account().halted

        previous = writer.account().halted_reason
        writer.update_account(halted=False, halted_reason=None)

        assert previous == '12 consecutive losses'
        assert not writer.account().halted
        assert writer.account().halted_reason is None
        # And the breakers must not immediately re-halt a clean account.
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None

    def test_clearing_does_not_touch_the_bankroll(self, writer):
        """Clearing a breaker is not a reset. The money stays where it is."""
        writer.update_account(bankroll=61.25, halted=True, halted_reason='daily loss')
        writer.update_account(halted=False, halted_reason=None)
        assert writer.account().bankroll == pytest.approx(61.25)


class TestTheDrawdownBreaker:
    """Peak-to-current drawdown on realised equity.

    **The daily rule cannot see this shape, and it is the shape that happened.**
    Over the first two live days equity ran $100 -> $166.86 by 13:00 UTC and gave
    back $63.92 over the next ten hours — all inside one UTC day, so that day's
    realised was **+$3.81** against a -$15.00 limit. The daily rule saw a good day
    while the account sat 38.3% below its high, and nothing else was watching:
    `max_drawdown <= 0.35` was a promotion gate on the *backtest* only.
    """

    def _run_up_then_give_back(self, w: PgWriter, up: float, down: float,
                               when: datetime, *, legs: int = 6) -> None:
        """Win `up` dollars, then lose `down`, in a FEW large settlements.

        Few on purpose. The first version used $2 steps, which meant giving back
        $64 took 32 consecutive losses — and that trips
        `max_consecutive_losses = 12` on its own. Two of these tests then passed
        while the drawdown breaker did nothing, which mutation testing caught:
        blanking the peak calculation left them green. Keeping every streak under
        the limit is what isolates the breaker under test.
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
        """+$66 then -$64 on a $100 start. 38% below the peak, and up 8% on the
        stake — which is the real case that fired first time and should not have.

        Halting here protects banked gains, which is a fund's job. This account
        exists to find out whether an edge is real, and the stake is already
        guarded by the ruin floor and the daily limit.
        """
        self._run_up_then_give_back(writer, up=66.0, down=64.0, when=NOW)
        account = writer.account()
        assert float(account.realized_pnl) == pytest.approx(2.0, abs=0.5)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None, (
            'halted an account that is up on the money the user put in'
        )

    def test_the_same_drawdown_below_the_stake_does_halt(self, writer):
        """+$40 then -$50: 35.7% off the peak AND $10 of real capital gone.

        The numbers are chosen so the *daily* rule cannot fire — -$10 is inside
        its -$15 limit — because the first version used -$60 and the daily rule
        caught it first, which would have let this test pass without the
        drawdown breaker doing anything. Same masking as the loss-streak one.
        """
        self._run_up_then_give_back(writer, up=40.0, down=50.0, when=NOW)
        assert float(writer.account().realized_pnl) == pytest.approx(-10.0, abs=0.5)
        reason = check_circuit_breakers(writer, CONFIG, now=NOW)
        assert reason is not None and 'drawdown' in reason, reason
        assert writer.account().halted

    def test_a_shallow_giveback_does_not_halt(self, writer):
        """+$20 then -$4 is 3.3% off the peak. Nowhere near the limit."""
        self._run_up_then_give_back(writer, up=20.0, down=4.0, when=NOW)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None

    def test_it_measures_from_the_peak_and_not_from_the_start(self, writer):
        """The whole point. An account back at its starting bankroll has lost
        nothing by the start-based view and everything it gained by the peak."""
        self._run_up_then_give_back(writer, up=60.0, down=60.0, when=NOW)
        account = writer.account()
        assert float(account.realized_pnl) == pytest.approx(0.0, abs=0.5)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None, (
            'exactly flat on the stake: nothing of the users money is gone, so '
            'the drawdown is a signal rather than a stop'
        )

    def test_a_steadily_winning_account_is_never_halted(self, writer):
        """The high-water mark must track upward, or every win looks like a
        drawdown from some earlier lower peak."""
        for i in range(20):
            settle(writer, index=i, pnl=+2.0, when=NOW)
        assert check_circuit_breakers(writer, CONFIG, now=NOW) is None

    def test_the_threshold_matches_the_promotion_gate(self):
        """A drawdown that blocks promotion should stop the money too."""
        from core.metrics import DEFAULT_GATES

        assert Config().max_drawdown_fraction == DEFAULT_GATES['max_drawdown'][0]


class TestTheHighWaterMark:
    def test_it_is_the_running_maximum_not_the_final_value(self, writer):
        settle(writer, index=0, pnl=+10.0, when=NOW)
        settle(writer, index=1, pnl=+10.0, when=NOW)
        settle(writer, index=2, pnl=-15.0, when=NOW)
        assert writer.realised_high_water() == pytest.approx(20.0, abs=0.5)
        assert float(writer.account().realized_pnl) == pytest.approx(5.0, abs=0.5)

    def test_an_account_that_only_ever_lost_has_a_peak_of_zero(self, writer):
        """Not a negative peak — the drawdown is measured from the starting
        bankroll, which is the highest the account has ever been."""
        settle(writer, index=0, pnl=-5.0, when=NOW)
        assert writer.realised_high_water() == pytest.approx(0.0, abs=1e-6)

    def test_an_empty_account_has_no_peak(self, writer):
        assert writer.realised_high_water() == 0.0
