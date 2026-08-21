"""Paper trading: act on `signals` rows and account for the result honestly.

    python -m scripts.paper_engine
    python -m scripts.paper_engine --active-coins ETH,BTC --min-edge-to-risk 0.1

This is the only component that reports what the strategy is actually earning, so
its arithmetic has to match `core.backtest` exactly or the two will disagree and
neither will be believable. It now shares the primitives — `core.costs` for
commission, `core.execution` for barriers, fill prices, funding accrual and
liquidation — rather than carrying a second implementation of each.

Four things it used to get wrong, all in the same direction of flattering or
distorting the result:

* **Fees were charged about twice.** The engine deducted the entry commission
  from cash at open, then closed with `calculate_pnl_exact`, which nets a *round
  trip* of fees internally, and then subtracted a separately computed exit fee on
  top. Accounting now moves cash once per side, as the backtest does.
* **Funding was never accrued** — every PnL call passed `accum_funding=0.0`. On
  hourly-funding perps that is the largest cost in the system after commission:
  2bp/hour over a day is 48bp against a round trip of 5-54bp. Funding could
  exceed fees and the paper book would not show it.
* **The fee schedule was never loaded**, so every contract was priced at the
  hardcoded 10bp/side. That is wrong for every Coinbase CDE contract, by 0.06x
  to 2.5x depending on which one.
* **Liquidation was not modelled at all.** A levered position could mark far
  through its maintenance margin and keep running.

Sizing comes from the signal. `contracts_suggested` is what `core.signal.decide`
produced, including the risk budget and the participation cap; recomputing a size
here would let paper trading take positions the backtest never tested. A signal
without a size did not pass `decide`, so it is not actionable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.config import DEFAULT_COST_CONFIG_NAME, Config, find_cost_config
from core.costs import get_contract_spec
from core.execution import (
    MAINTENANCE_MARGIN_FRACTION,
    Position,
    barrier_prices,
    entry_cost,
    fill_price,
)
from core.pg_writer import PgWriter
from core.profiles import COIN_PROFILES, CoinProfile

logger = logging.getLogger('paper_engine')

INITIAL_EQUITY = 100_000.0

# Funding is hourly, so re-reading the store more often than this buys nothing.
FUNDING_CACHE_SECONDS = 300.0

# Volatility window for the barriers, matching `core.backtest.VOL_WINDOW_BARS`.
VOL_WINDOW_BARS = 24


@dataclass
class EngineState:
    cash_balance: float = INITIAL_EQUITY
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_signal_id: int = 0


class FundingSource:
    """Latest funding rate per instrument, read from the research store.

    Funding does not live in Postgres — the scraper writes it to the research
    store — so the engine reads it there and caches it. Absent data returns zero
    and says so once per instrument: silently assuming zero funding on a perp is
    how the previous version lost track of its largest cost.
    """

    def __init__(self, store_root: Optional[str] = None) -> None:
        self._store_root = store_root
        self._rates: dict[str, float] = {}
        self._fetched_at = 0.0
        self._warned: set[str] = set()

    def _refresh(self) -> None:
        try:
            from core.datastore import ResearchStore

            store = ResearchStore(self._store_root) if self._store_root else ResearchStore()
            frame = store.read('funding')
        except Exception as exc:  # noqa: BLE001 - a missing store must not stop trading
            logger.warning('cannot read funding from the research store: %s', exc)
            self._rates = {}
            self._fetched_at = time.monotonic()
            return

        rates: dict[str, float] = {}
        if not frame.empty and {'symbol', 'rate'} <= set(frame.columns):
            latest = frame.sort_values('event_time').groupby('symbol').last()
            rates = {
                str(symbol): float(row['rate'])
                for symbol, row in latest.iterrows()
                if pd.notna(row['rate'])
            }
        self._rates = rates
        self._fetched_at = time.monotonic()

    def hourly_rate(self, symbol: str) -> float:
        if time.monotonic() - self._fetched_at > FUNDING_CACHE_SECONDS:
            self._refresh()
        key = (symbol or '').upper()
        if key in self._rates:
            return self._rates[key]
        if key not in self._warned:
            self._warned.add(key)
            logger.warning('no funding rate for %s: accruing zero, which understates cost', key)
        return 0.0


class PaperTradingEngine:
    def __init__(
        self,
        *,
        poll_seconds: float = 2.0,
        max_signal_age_minutes: float = 30.0,
        tier_map: Optional[dict[str, str]] = None,
        tier_size_multipliers: Optional[dict[str, float]] = None,
        min_edge_to_risk: float = 0.0,
        active_coins: Optional[list[str]] = None,
        cost_config: Optional[str] = None,
        store: Optional[str] = None,
    ) -> None:
        self.poll_seconds = poll_seconds
        self.max_signal_age_minutes = max_signal_age_minutes
        self.writer = PgWriter()
        self.config = _build_config(cost_config)
        self.min_edge_to_risk = min_edge_to_risk
        self.state = EngineState()
        self.funding = FundingSource(store)

        self.tier_map = {k.upper(): v.upper() for k, v in (tier_map or {}).items()}
        self.tier_size_multipliers = {'FULL': 1.0, 'PILOT': 0.5, 'SHADOW': 0.0}
        if tier_size_multipliers:
            self.tier_size_multipliers.update(
                {k.upper(): float(v) for k, v in tier_size_multipliers.items()}
            )
        self.active_coins = {c.upper() for c in active_coins} if active_coins else None
        if self.active_coins:
            logger.info('active_coins filter: %s', sorted(self.active_coins))
        self._funding_marked_at: dict[int, datetime] = {}

    # -- lookups -----------------------------------------------------------

    def _profile_for(self, coin: str) -> Optional[CoinProfile]:
        return COIN_PROFILES.get((coin or '').upper())

    def _coin_tier(self, coin: str) -> str:
        return self.tier_map.get((coin or '').upper(), 'FULL')

    def _volatility(self, coin: str) -> Optional[float]:
        """Realised hourly volatility from recent signal prices.

        The same quantity the backtest computes from bars, from the only price
        series this process has. Returns None rather than a guess when there is
        not enough history — the caller falls back to the profile's target.
        """
        series = self.writer.get_recent_signal_prices_for_coin(coin, limit=VOL_WINDOW_BARS * 2)
        prices = [p for _, p in series if p and p > 0]
        if len(prices) < 8:
            return None
        frame = pd.Series(prices, dtype=float)
        deviation = frame.pct_change().dropna().std()
        return float(deviation) if pd.notna(deviation) and deviation > 0 else None

    def _barrier_volatility(self, coin: str) -> float:
        measured = self._volatility(coin)
        if measured:
            return measured
        profile = self._profile_for(coin)
        target = float(getattr(profile, 'label_vol_target', 0.0)) if profile else 0.0
        # The profile target is expressed in percent of price, so scale it, and
        # never fall below 0.5% — a zero would collapse the barriers onto entry.
        return max(target * 0.01, 0.005)

    # -- position reconstruction ------------------------------------------

    def _as_position(self, row: Any) -> Position:
        """Rebuild a `core.execution.Position` from a database row.

        Going through the shared type is what keeps the risk arithmetic — margin,
        maintenance, liquidation level — identical to the backtest's, instead of
        re-deriving it here and drifting.
        """
        direction = 1 if row.side == 'long' else -1
        contracts = int(row.contracts)
        entry_price = float(row.entry_price)
        spec = get_contract_spec(row.coin)
        notional = spec.notional(contracts, entry_price)
        entry_time = row.opened_at or datetime.now(timezone.utc)
        if getattr(entry_time, 'tzinfo', None) is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        return Position(
            symbol=row.coin,
            direction=direction,
            contracts=contracts,
            entry_price=entry_price,
            entry_time=pd.Timestamp(entry_time),
            entry_fee=float(row.fees_paid or 0.0),
            margin=notional / max(float(self.config.leverage), 1e-9),
            take_profit=float(row.tp_price) if row.tp_price is not None else None,
            stop_loss=float(row.sl_price) if row.sl_price is not None else None,
            hold_until=pd.Timestamp(row.max_hold_until) if row.max_hold_until else None,
            funding_paid=float(getattr(row, 'funding_paid', 0.0) or 0.0),
        )

    # -- accounting -------------------------------------------------------

    def _accrue_funding(self, row: Any) -> float:
        """Charge funding for the hours elapsed since this position was last marked.

        Elapsed hours rather than poll ticks: the loop polls every couple of
        seconds and funding settles hourly, so charging per tick would inflate the
        cost by three orders of magnitude while charging per poll-with-a-flag
        would miss hours the process was down.
        """
        position_id = int(row.id)
        now = datetime.now(timezone.utc)
        last = self._funding_marked_at.get(position_id)
        if last is None:
            opened = row.opened_at or now
            if getattr(opened, 'tzinfo', None) is None:
                opened = opened.replace(tzinfo=timezone.utc)
            last = opened

        hours = (now - last).total_seconds() / 3600.0
        if hours < 1.0:
            return 0.0

        rate = self.funding.hourly_rate(row.coin)
        mark = self.writer.get_latest_signal_price(row.coin) or float(row.entry_price)
        spec = get_contract_spec(row.coin)
        direction = 1 if row.side == 'long' else -1
        # A long pays a positive funding rate; a short receives it.
        charge = rate * int(hours) * spec.notional(int(row.contracts), float(mark)) * direction

        self._funding_marked_at[position_id] = last + timedelta(hours=int(hours))
        if charge:
            self.state.cash_balance -= charge
            self.state.realized_pnl -= charge
            self.writer.accrue_paper_position_funding(position_id, charge)
        return float(charge)

    def _close(self, row: Any, exit_price: float, reason: str) -> float:
        """Close a position and move cash once, exactly as the backtest does.

        Only the price move and the exit commission land here. Funding was
        charged against cash as it accrued, and the entry commission was charged
        at open, so including either again would double-count it — which is the
        defect this replaces.
        """
        position = self._as_position(row)
        filled = fill_price(exit_price, -position.direction, self.config.slippage_bps)
        exit_fee = entry_cost(position.contracts, filled, position.symbol, self.config)
        price_pnl = position.unrealised(filled)
        realized = price_pnl - exit_fee

        self.state.realized_pnl += realized
        self.state.cash_balance += realized
        self._funding_marked_at.pop(int(row.id), None)

        self.writer.close_paper_position(
            position_id=row.id,
            mark_price=filled,
            realized_pnl=float(row.realized_pnl or 0.0) + realized,
            fees_paid=float(row.fees_paid or 0.0) + exit_fee,
            exit_reason=reason,
        )
        logger.info(
            'closed %s %s @ %.6f reason=%s price_pnl=%.2f exit_fee=%.2f realized=%.2f',
            row.coin, row.side, filled, reason, price_pnl, exit_fee, realized,
        )
        return realized

    # -- signal handling --------------------------------------------------

    def _signal_is_fresh(self, signal: Any) -> bool:
        if signal.timestamp is None:
            return False
        stamp = signal.timestamp
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - stamp).total_seconds() / 60.0 <= (
            self.max_signal_age_minutes
        )

    def _act_on(self, signal: Any) -> None:
        coin = (signal.coin or '').upper()

        def dismiss(reason: str, level: int = logging.INFO) -> None:
            logger.log(level, 'signal %s (%s): %s', signal.id, coin, reason)
            self.writer.mark_signal_acted(signal.id)

        if self.active_coins and coin not in self.active_coins:
            return dismiss('not in active_coins')
        if signal.direction not in {'long', 'short'}:
            return dismiss('no direction')
        if not signal.passed_gates:
            return dismiss(f'blocked upstream ({signal.gate_failure_reason or "unknown"})')
        if not self._signal_is_fresh(signal):
            return dismiss('stale')

        price = float(signal.price_at_signal or 0.0)
        if price <= 0:
            return dismiss('no price')

        edge_to_risk = float(signal.confidence or 0.0)
        if edge_to_risk < self.min_edge_to_risk:
            return dismiss(f'edge/risk {edge_to_risk:.3f} < {self.min_edge_to_risk:.3f}')

        tier = self._coin_tier(coin)
        if tier == 'SHADOW':
            return dismiss('shadow tier: logged only')

        # The size is `decide`'s, not ours. A signal without one did not pass it.
        if not signal.contracts_suggested:
            return dismiss('no size on the signal; decide() did not clear it',
                           level=logging.WARNING)
        contracts = int(int(signal.contracts_suggested) * self.tier_size_multipliers.get(tier, 1.0))
        if contracts < 1:
            return dismiss(f'tier {tier} multiplier reduced the size below one contract')

        if self.writer.count_open_positions() >= self.config.max_positions:
            return dismiss('at the position limit')

        side = signal.direction
        direction = 1 if side == 'long' else -1
        existing = self.writer.get_all_open_paper_positions_for_coin(coin)
        if any(p.side == side for p in existing):
            return dismiss('same side already open; not pyramiding')

        filled = fill_price(price, direction, self.config.slippage_bps)
        spec = get_contract_spec(coin)
        notional = spec.notional(contracts, filled)

        # Leverage caps the notional a paper account can carry, and the cap binds
        # on the filled price, not the signal price.
        max_notional = max(self.state.cash_balance, 100.0) * float(self.config.leverage)
        if notional > max_notional:
            contracts = int(max_notional / max(spec.units * filled, 1e-9))
            if contracts < 1:
                return dismiss('leverage cap leaves less than one contract')
            notional = spec.notional(contracts, filled)

        fee = entry_cost(contracts, filled, coin, self.config)

        order_id = self.writer.create_paper_order(
            signal_id=signal.id, coin=coin, side=side,
            contracts=contracts, target_price=price,
        )
        self.writer.mark_paper_order_filled(order_id)
        self.writer.create_paper_fill(
            order_id=order_id, signal_id=signal.id, coin=coin, side=side,
            contracts=contracts, fill_price=filled, fee=fee, notional=notional,
            slippage_bps=self.config.slippage_bps,
        )

        # An opposite-side signal is a reversal: flatten first, at the same fill.
        for row in existing:
            self._close(row, filled, 'opposite_signal')

        self.state.cash_balance -= fee

        profile = self._profile_for(coin)
        volatility = self._barrier_volatility(coin)
        take_profit, stop_loss = barrier_prices(
            filled, volatility, direction,
            tp_mult=float(self.config.resolve('vol_mult_tp', profile)),
            sl_mult=float(self.config.resolve('vol_mult_sl', profile)),
        )
        hold_hours = self.config.label_horizon_hours(profile)

        self.writer.upsert_paper_position(
            coin=coin, side=side, contracts=contracts,
            entry_price=filled, mark_price=filled, notional=notional,
            realized_pnl=0.0, unrealized_pnl=0.0, fees_paid=fee, is_open=True,
            tp_price=take_profit, sl_price=stop_loss,
            max_hold_until=datetime.now(timezone.utc) + timedelta(hours=int(hold_hours)),
        )
        logger.info(
            'opened %s %s @ %.6f contracts=%d notional=%.0f fee=%.2f tp=%.6f sl=%.6f hold=%dh',
            coin, side, filled, contracts, notional, fee, take_profit, stop_loss, hold_hours,
        )
        self._write_equity_point()
        self.writer.mark_signal_acted(signal.id)

    # -- the loop ---------------------------------------------------------

    def _manage_positions(self) -> int:
        """Accrue funding, then check liquidation, stop, target and horizon.

        The order is the backtest's, and it matters: funding is what pushes a
        position into liquidation, and a position that liquidates never reaches
        its stop. Checking the stop first would report a bounded loss where the
        account actually lost its whole margin.
        """
        rows = self.writer.get_all_open_paper_positions()
        if not rows:
            return 0

        now = datetime.now(timezone.utc)
        closed = 0
        for row in rows:
            mark = self.writer.get_latest_signal_price(row.coin)
            if mark is None or mark <= 0:
                continue
            mark = float(mark)

            self._accrue_funding(row)
            position = self._as_position(row)

            liquidation = position.liquidation_price()
            if position.under_margined or (
                liquidation > 0
                and ((position.direction == 1 and mark <= liquidation)
                     or (position.direction == -1 and mark >= liquidation))
            ):
                logger.error(
                    'liquidating %s %s: mark %.6f through %.6f (maintenance %.0f%%)',
                    row.coin, row.side, mark, liquidation, MAINTENANCE_MARGIN_FRACTION * 100,
                )
                self._close(row, liquidation or mark, 'liquidation')
                closed += 1
                continue

            take_profit, stop_loss = position.take_profit, position.stop_loss
            if stop_loss is not None and (
                (position.direction == 1 and mark <= stop_loss)
                or (position.direction == -1 and mark >= stop_loss)
            ):
                self._close(row, stop_loss, 'stop_loss')
                closed += 1
                continue
            if take_profit is not None and (
                (position.direction == 1 and mark >= take_profit)
                or (position.direction == -1 and mark <= take_profit)
            ):
                self._close(row, take_profit, 'take_profit')
                closed += 1
                continue

            deadline = row.max_hold_until
            if deadline is not None:
                if getattr(deadline, 'tzinfo', None) is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                if now >= deadline:
                    self._close(row, mark, 'max_hold')
                    closed += 1

        if closed:
            self._write_equity_point()
        return closed

    def _mark_to_market(self) -> None:
        """Update unrealised PnL on open positions.

        Unrealised means the price move alone. The entry commission is already out
        of cash and funding is charged as it accrues, so folding either into the
        mark would double-count it and make the equity curve disagree with the
        realised total on every close.
        """
        rows = self.writer.get_all_open_paper_positions()
        if not rows:
            if self.state.unrealized_pnl != 0.0:
                self.state.unrealized_pnl = 0.0
                self._write_equity_point()
            return

        total = 0.0
        changed = False
        for row in rows:
            mark = self.writer.get_latest_signal_price(row.coin)
            if mark is None or mark <= 0:
                total += float(row.unrealized_pnl or 0.0)
                continue
            mark = float(mark)
            unrealised = self._as_position(row).unrealised(mark)
            if mark != row.mark_price or unrealised != row.unrealized_pnl:
                self.writer.update_paper_position_mark(row.id, mark, unrealised)
                changed = True
            total += unrealised

        self.state.unrealized_pnl = total
        if changed:
            self._write_equity_point()

    def _write_equity_point(self) -> None:
        self.writer.write_paper_equity_point(
            equity=self.state.cash_balance + self.state.unrealized_pnl,
            cash_balance=self.state.cash_balance,
            unrealized_pnl=self.state.unrealized_pnl,
            realized_pnl=self.state.realized_pnl,
            open_positions=self.writer.count_open_positions(),
            timestamp=datetime.now(timezone.utc),
        )

    def _restore_state(self) -> None:
        state = self.writer.compute_paper_state_from_history(initial_equity=INITIAL_EQUITY)
        self.state.cash_balance = state['cash_balance']
        self.state.realized_pnl = state['realized_pnl']
        self.state.unrealized_pnl = state['unrealized_pnl']
        logger.info(
            'restored: cash=%.2f realized=%.2f unrealized=%.2f',
            self.state.cash_balance, self.state.realized_pnl, self.state.unrealized_pnl,
        )

    def _publish_config(self) -> None:
        active = sorted(self.active_coins) if self.active_coins else []
        self.writer.upsert_paper_engine_config(active_coins=active, tier_map=self.tier_map)
        logger.info('published engine config: active_coins=%s', active)

    def run_forever(self) -> None:
        logger.info(
            'paper engine starting (poll=%ss, costs=%s, leverage=%sx)',
            self.poll_seconds, self.config.cost_config_version, self.config.leverage,
        )
        self._restore_state()
        self._publish_config()
        while True:
            # Exits first, so a fresh mark closes stale positions before new
            # fills add risk on top of them.
            self._manage_positions()
            for signal in self.writer.get_unprocessed_signals(self.state.last_signal_id):
                self.state.last_signal_id = max(self.state.last_signal_id, signal.id)
                self._act_on(signal)
            self._mark_to_market()
            time.sleep(self.poll_seconds)


def _build_config(cost_config: Optional[str]) -> Config:
    """A Config with the venue's real fee schedule loaded unless refused.

    Loud on failure. Paper trading exists to tell you what the strategy earns,
    and a paper book priced at the wrong commission answers a different question
    than the one being asked.
    """
    config = Config()
    if cost_config and cost_config.lower() == 'none':
        logger.warning('running on the hardcoded %.1fbp/side default, which is '
                       'wrong for every Coinbase CDE contract', config.taker_bps)
        return config

    path = find_cost_config(cost_config or DEFAULT_COST_CONFIG_NAME)
    if path is None:
        logger.error('cost config not found: %s. Paper PnL will be mispriced.',
                     cost_config or DEFAULT_COST_CONFIG_NAME)
        return config
    return config.with_cost_assumptions(path)


def _parse_tier_map(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    candidate = Path(raw)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and isinstance(payload.get('deployment_tier_map'), dict):
            payload = payload['deployment_tier_map']
    else:
        payload = json.loads(raw)
    return {str(k).upper(): str(v).upper() for k, v in dict(payload).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=float, default=2.0)
    parser.add_argument('--max-signal-age-minutes', type=float, default=30.0)
    parser.add_argument('--min-edge-to-risk', type=float, default=0.0,
                        help='Extra edge/risk floor on top of the one decide() '
                             'already applied. Zero trusts the signal.')
    parser.add_argument('--tier-map', default='',
                        help='JSON, or a path to JSON, of {coin: tier}')
    parser.add_argument('--tier-size-multipliers',
                        default='{"FULL":1.0,"PILOT":0.5,"SHADOW":0.0}')
    parser.add_argument('--active-coins', default='',
                        help='Comma-separated coins to trade. Empty means all.')
    parser.add_argument('--cost-config', default=os.getenv('COST_CONFIG') or None,
                        help="Venue fee schedule. 'none' to use the hardcoded default.")
    parser.add_argument('--store', default=os.getenv('RESEARCH_STORE') or None,
                        help='Research store root, for funding rates')
    parser.add_argument('--log-level', default=os.getenv('LOG_LEVEL', 'INFO'))
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')

    PaperTradingEngine(
        poll_seconds=args.poll_seconds,
        max_signal_age_minutes=args.max_signal_age_minutes,
        tier_map=_parse_tier_map(args.tier_map),
        tier_size_multipliers=json.loads(args.tier_size_multipliers),
        min_edge_to_risk=args.min_edge_to_risk,
        active_coins=[c.strip().upper() for c in args.active_coins.split(',') if c.strip()] or None,
        cost_config=args.cost_config,
        store=args.store,
    ).run_forever()


if __name__ == '__main__':
    main()
