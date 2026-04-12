"""Paper trading engine that consumes live `signals` rows and persists state transitions."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.coin_profiles import COIN_PROFILES, CoinProfile
from core.paper_profile_overrides import load_paper_profile_overrides
from core.pg_writer import PgWriter
from scripts.train_model import Config, calculate_coinbase_fee, calculate_n_contracts, calculate_pnl_exact, get_contract_spec

logger = logging.getLogger("paper_engine")


@dataclass
class EngineState:
    cash_balance: float = 100_000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_signal_id: int = 0


class PaperTradingEngine:
    def __init__(
        self,
        poll_seconds: float = 2.0,
        max_signal_age_minutes: float = 30.0,
        tier_map: dict[str, str] | None = None,
        tier_size_multipliers: dict[str, float] | None = None,
        min_confidence: float = 0.0,
        active_coins: list[str] | None = None,
    ):
        self.poll_seconds = poll_seconds
        self.max_signal_age_minutes = max_signal_age_minutes
        self.writer = PgWriter()
        self.config = Config()
        self.min_confidence = min_confidence
        self.state = EngineState()
        self.tier_map = {k.upper(): v.upper() for k, v in (tier_map or {}).items()}
        self.tier_size_multipliers = {"FULL": 1.0, "PILOT": 0.5, "SHADOW": 0.0}
        if tier_size_multipliers:
            self.tier_size_multipliers.update({k.upper(): float(v) for k, v in tier_size_multipliers.items()})
        self.active_coins: set[str] | None = {c.upper() for c in active_coins} if active_coins else None
        if self.active_coins:
            logger.info("active_coins filter: %s", sorted(self.active_coins))
        self._profile_overrides = load_paper_profile_overrides(os.environ.get("PAPER_PROFILE_OVERRIDES_PATH", "data/paper_candidates"))

    def _profile_for(self, coin: str) -> CoinProfile | None:
        key = (coin or "").upper()
        return self._profile_overrides.get(key) or COIN_PROFILES.get(key)

    def _estimate_vol_24h(self, coin: str) -> float | None:
        """Estimate 24h realized volatility from recent signal prices (hourly cadence).

        Matches the backtest's vol proxy: stdev of pct-change on 1h close prices
        over a 24h window. Falls back to None if we don't have enough history yet.
        """
        series = self.writer.get_recent_signal_prices_for_coin(coin, limit=48)
        if len(series) < 8:
            return None
        prices = [p for _, p in series if p and p > 0]
        if len(prices) < 8:
            return None
        rets = []
        for a, b in zip(prices[:-1], prices[1:]):
            if a > 0:
                rets.append((b - a) / a)
        if len(rets) < 6:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        return math.sqrt(var) if var > 0 else None

    def _coin_tier(self, coin: str) -> str:
        return self.tier_map.get((coin or "").upper(), "FULL")

    def _fill_price(self, base_price: float, side: str) -> float:
        slip = self.config.slippage_bps / 10000.0
        if side == "long":
            return base_price * (1.0 + slip)
        return base_price * (1.0 - slip)

    def _signal_is_fresh(self, signal) -> bool:
        if signal.timestamp is None:
            return False
        signal_ts = signal.timestamp
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - signal_ts).total_seconds() / 60.0
        return age_minutes <= self.max_signal_age_minutes

    def _simulate_fill(self, signal) -> None:
        if self.active_coins and (signal.coin or "").upper() not in self.active_coins:
            logger.debug("coin %s not in active_coins, skipping signal id=%s", signal.coin, signal.id)
            self.writer.mark_signal_acted(signal.id)
            return

        if signal.direction not in {"long", "short"}:
            return
        if not self._signal_is_fresh(signal):
            logger.info("skipping stale signal id=%s (%s)", signal.id, signal.coin)
            self.writer.mark_signal_acted(signal.id)
            return

        side = signal.direction
        direction = 1 if side == "long" else -1
        price = float(signal.price_at_signal or 0)
        if price <= 0:
            return

        confidence = float(signal.confidence or 0.0)
        if confidence < self.min_confidence:
            logger.info(
                "skipping low-confidence signal id=%s coin=%s conf=%.3f<th=%.3f",
                signal.id,
                signal.coin,
                confidence,
                self.min_confidence,
            )
            self.writer.mark_signal_acted(signal.id)
            return

        deployment_tier = self._coin_tier(signal.coin)
        if deployment_tier == "SHADOW":
            logger.info("shadow tier signal logged only id=%s coin=%s", signal.id, signal.coin)
            self.writer.mark_signal_acted(signal.id)
            return

        open_positions = self.writer.count_open_positions()
        if open_positions >= self.config.max_positions:
            logger.info("max open positions reached (%s), skipping signal id=%s", open_positions, signal.id)
            self.writer.mark_signal_acted(signal.id)
            return

        # Get ALL open positions for this coin to prevent duplicate open positions
        open_positions_for_coin = self.writer.get_all_open_paper_positions_for_coin(signal.coin)

        # If any same-side position is open, skip pyramiding
        if any(p.side == side for p in open_positions_for_coin):
            logger.info("same-side position already open for %s, skipping pyramiding signal id=%s", signal.coin, signal.id)
            self.writer.mark_signal_acted(signal.id)
            return

        contracts = signal.contracts_suggested or calculate_n_contracts(
            equity=max(self.state.cash_balance, 100.0),
            price=price,
            symbol=signal.coin,
            config=self.config,
        )
        size_multiplier = float(self.tier_size_multipliers.get(deployment_tier, 1.0))
        contracts = int(contracts * max(0.0, size_multiplier))
        if contracts <= 0:
            logger.info("tier multiplier reduced contracts to zero id=%s coin=%s tier=%s", signal.id, signal.coin, deployment_tier)
            self.writer.mark_signal_acted(signal.id)
            return

        spec = get_contract_spec(signal.coin)
        notional = contracts * spec["units"] * price
        max_notional = max(self.state.cash_balance, 100.0) * self.config.leverage
        if notional > max_notional:
            contracts = int(max_notional / max(spec["units"] * price, 1e-9))
            if contracts <= 0:
                return
            notional = contracts * spec["units"] * price

        fee = calculate_coinbase_fee(contracts, price, signal.coin, self.config)
        fill_price = self._fill_price(price, side)

        order_id = self.writer.create_paper_order(
            signal_id=signal.id,
            coin=signal.coin,
            side=side,
            contracts=contracts,
            target_price=price,
        )
        self.writer.mark_paper_order_filled(order_id)
        self.writer.create_paper_fill(
            order_id=order_id,
            signal_id=signal.id,
            coin=signal.coin,
            side=side,
            contracts=contracts,
            fill_price=fill_price,
            fee=fee,
            notional=notional,
            slippage_bps=self.config.slippage_bps,
        )

        # Close ALL existing open positions for this coin (handles stale/duplicate rows too)
        for open_position in open_positions_for_coin:
            close_dir = 1 if open_position.side == "long" else -1
            _, _, _, _, _, _, pnl_dollars, _ = calculate_pnl_exact(
                entry_price=open_position.entry_price,
                exit_price=fill_price,
                direction=close_dir,
                accum_funding=0.0,
                n_contracts=open_position.contracts,
                symbol=signal.coin,
                config=self.config,
            )
            close_fee = calculate_coinbase_fee(open_position.contracts, fill_price, signal.coin, self.config)
            realized = pnl_dollars - close_fee
            self.state.realized_pnl += realized
            self.state.cash_balance += realized
            self.writer.close_paper_position(
                position_id=open_position.id,
                mark_price=fill_price,
                realized_pnl=open_position.realized_pnl + realized,
                fees_paid=open_position.fees_paid + close_fee,
                exit_reason="opposite_signal",
            )

        self.state.cash_balance -= fee

        tp_price, sl_price, max_hold_until = self._compute_exit_levels(signal.coin, side, fill_price)
        self.writer.upsert_paper_position(
            coin=signal.coin,
            side=side,
            contracts=contracts,
            entry_price=fill_price,
            mark_price=fill_price,
            notional=notional,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_paid=fee,
            is_open=True,
            tp_price=tp_price,
            sl_price=sl_price,
            max_hold_until=max_hold_until,
        )
        logger.info(
            "opened %s %s @ %.6f contracts=%s tp=%s sl=%s max_hold=%s",
            signal.coin, side, fill_price, contracts,
            f"{tp_price:.6f}" if tp_price else "-",
            f"{sl_price:.6f}" if sl_price else "-",
            max_hold_until.isoformat() if max_hold_until else "-",
        )

        n_open = self.writer.count_open_positions()
        self.writer.write_paper_equity_point(
            equity=self.state.cash_balance + self.state.unrealized_pnl,
            cash_balance=self.state.cash_balance,
            unrealized_pnl=self.state.unrealized_pnl,
            realized_pnl=self.state.realized_pnl,
            open_positions=n_open,
            timestamp=datetime.now(timezone.utc),
        )
        self.writer.mark_signal_acted(signal.id)

    def _compute_exit_levels(self, coin: str, side: str, entry_price: float) -> tuple[float | None, float | None, datetime | None]:
        """TP = entry * (1 + vol_mult_tp * vol * dir), SL = entry * (1 - vol_mult_sl * vol * dir).

        Matches backtest semantics in train_model.run_backtest. Falls back to
        label_vol_target * 0.01 if we don't yet have enough signal history for 24h vol.
        """
        profile = self._profile_for(coin)
        if profile is None:
            return None, None, None
        direction = 1 if side == "long" else -1
        vol = self._estimate_vol_24h(coin)
        if vol is None or vol <= 0:
            # Cold-start fallback: use the per-coin label_vol_target as a %-of-price approximation
            vol = max(float(profile.label_vol_target) * 0.01, 0.005)
        tp_price = entry_price * (1.0 + profile.vol_mult_tp * vol * direction)
        sl_price = entry_price * (1.0 - profile.vol_mult_sl * vol * direction)
        max_hold_until = datetime.now(timezone.utc) + timedelta(hours=int(profile.max_hold_hours))
        return tp_price, sl_price, max_hold_until

    def _close_position(self, position, exit_price: float, reason: str) -> None:
        direction = 1 if position.side == "long" else -1
        _, _, _, _, _, _, pnl_dollars, _ = calculate_pnl_exact(
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=direction,
            accum_funding=0.0,
            n_contracts=position.contracts,
            symbol=position.coin,
            config=self.config,
        )
        close_fee = calculate_coinbase_fee(position.contracts, exit_price, position.coin, self.config)
        realized = pnl_dollars - close_fee
        self.state.realized_pnl += realized
        self.state.cash_balance += realized
        self.writer.close_paper_position(
            position_id=position.id,
            mark_price=exit_price,
            realized_pnl=float(position.realized_pnl or 0.0) + realized,
            fees_paid=float(position.fees_paid or 0.0) + close_fee,
            exit_reason=reason,
        )
        logger.info(
            "closed %s %s @ %.6f reason=%s realized=%.2f",
            position.coin, position.side, exit_price, reason, realized,
        )

    def _manage_exits(self) -> int:
        """Close positions whose TP, SL, or max_hold_until has been hit. Returns count closed."""
        open_positions = self.writer.get_all_open_paper_positions()
        if not open_positions:
            return 0
        now = datetime.now(timezone.utc)
        closed = 0
        for position in open_positions:
            mark_price = self.writer.get_latest_signal_price(position.coin)
            if mark_price is None or mark_price <= 0:
                continue
            side = position.side
            direction = 1 if side == "long" else -1
            tp = float(position.tp_price) if position.tp_price is not None else None
            sl = float(position.sl_price) if position.sl_price is not None else None
            max_hold_until = position.max_hold_until

            # Back-fill exit levels for legacy rows that pre-date this feature
            if tp is None or sl is None or max_hold_until is None:
                tp_new, sl_new, mh_new = self._compute_exit_levels(position.coin, side, float(position.entry_price))
                tp = tp if tp is not None else tp_new
                sl = sl if sl is not None else sl_new
                max_hold_until = max_hold_until or mh_new

            tp_hit = tp is not None and ((direction == 1 and mark_price >= tp) or (direction == -1 and mark_price <= tp))
            sl_hit = sl is not None and ((direction == 1 and mark_price <= sl) or (direction == -1 and mark_price >= sl))

            if tp_hit:
                self._close_position(position, tp, "take_profit")
                closed += 1
                continue
            if sl_hit:
                self._close_position(position, sl, "stop_loss")
                closed += 1
                continue
            if max_hold_until is not None:
                deadline = max_hold_until
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                if now >= deadline:
                    self._close_position(position, mark_price, "max_hold")
                    closed += 1
        if closed:
            n_open = self.writer.count_open_positions()
            self.writer.write_paper_equity_point(
                equity=self.state.cash_balance + self.state.unrealized_pnl,
                cash_balance=self.state.cash_balance,
                unrealized_pnl=self.state.unrealized_pnl,
                realized_pnl=self.state.realized_pnl,
                open_positions=n_open,
                timestamp=now,
            )
        return closed

    def _mark_to_market(self) -> None:
        """Update unrealized PnL for all open positions using latest signal prices."""
        open_positions = self.writer.get_all_open_paper_positions()
        if not open_positions:
            if self.state.unrealized_pnl != 0.0:
                self.state.unrealized_pnl = 0.0
            return

        total_unrealized = 0.0
        any_updated = False
        for position in open_positions:
            mark_price = self.writer.get_latest_signal_price(position.coin)
            if mark_price is None or mark_price <= 0:
                total_unrealized += position.unrealized_pnl
                continue
            direction = 1 if position.side == "long" else -1
            _, _, _, _, _, _, pnl_dollars, _ = calculate_pnl_exact(
                entry_price=position.entry_price,
                exit_price=mark_price,
                direction=direction,
                accum_funding=0.0,
                n_contracts=position.contracts,
                symbol=position.coin,
                config=self.config,
            )
            unrealized = pnl_dollars
            if mark_price != position.mark_price or unrealized != position.unrealized_pnl:
                self.writer.update_paper_position_mark(position.id, mark_price, unrealized)
                any_updated = True
            total_unrealized += unrealized

        self.state.unrealized_pnl = total_unrealized
        if any_updated:
            n_open = self.writer.count_open_positions()
            self.writer.write_paper_equity_point(
                equity=self.state.cash_balance + self.state.unrealized_pnl,
                cash_balance=self.state.cash_balance,
                unrealized_pnl=self.state.unrealized_pnl,
                realized_pnl=self.state.realized_pnl,
                open_positions=n_open,
                timestamp=datetime.now(timezone.utc),
            )

    def _restore_state(self) -> None:
        """Restore cash_balance/realized/unrealized from DB history to survive container restarts."""
        state = self.writer.compute_paper_state_from_history(initial_equity=100_000.0)
        self.state.cash_balance = state["cash_balance"]
        self.state.realized_pnl = state["realized_pnl"]
        self.state.unrealized_pnl = state["unrealized_pnl"]
        logger.info(
            "restored state from history: cash=%.2f realized=%.2f unrealized=%.2f",
            self.state.cash_balance, self.state.realized_pnl, self.state.unrealized_pnl,
        )

    def _publish_config(self) -> None:
        """Write active_coins + tier_map to DB so the API can expose them to the frontend."""
        active = sorted(self.active_coins) if self.active_coins else []
        self.writer.upsert_paper_engine_config(active_coins=active, tier_map=self.tier_map)
        logger.info("published engine config to DB: active_coins=%s", active)

    def run_forever(self) -> None:
        logger.info("starting paper engine loop (poll=%ss)", self.poll_seconds)
        self._restore_state()
        self._publish_config()
        while True:
            # Exits first so a fresh mark can close stale positions before new fills accrue risk
            self._manage_exits()
            signals = self.writer.get_unprocessed_signals(self.state.last_signal_id)
            for signal in signals:
                self.state.last_signal_id = max(self.state.last_signal_id, signal.id)
                self._simulate_fill(signal)
            self._mark_to_market()
            time.sleep(self.poll_seconds)


def main() -> None:
    def _parse_tier_map(raw: str) -> dict[str, str]:
        if not raw:
            return {}
        maybe_path = Path(raw)
        if maybe_path.exists():
            payload = json.loads(maybe_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("deployment_tier_map"), dict):
                payload = payload["deployment_tier_map"]
            return {str(k).upper(): str(v).upper() for k, v in dict(payload).items()}
        parsed = json.loads(raw)
        return {str(k).upper(): str(v).upper() for k, v in dict(parsed).items()}

    def _parse_multipliers(raw: str) -> dict[str, float]:
        if not raw:
            return {}
        return {str(k).upper(): float(v) for k, v in dict(json.loads(raw)).items()}

    parser = argparse.ArgumentParser(description="paper trading engine")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-signal-age-minutes", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum signal confidence to act on (default 0.0 trusts train_model pre-filtering)")
    parser.add_argument(
        "--tier-map",
        type=str,
        default="",
        help="JSON string or path to JSON containing {coin: tier} or a launch_summary with deployment_tier_map",
    )
    parser.add_argument(
        "--tier-size-multipliers",
        type=str,
        default='{"FULL":1.0,"PILOT":0.5,"SHADOW":0.0}',
        help="JSON map of deployment tier to position size multiplier",
    )
    parser.add_argument(
        "--active-coins",
        type=str,
        default="",
        help="Comma-separated list of coins to trade (e.g. ETH,BTC). Empty means all coins are active.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    active_coins = [c.strip().upper() for c in args.active_coins.split(",") if c.strip()] if args.active_coins else None
    PaperTradingEngine(
        poll_seconds=args.poll_seconds,
        max_signal_age_minutes=args.max_signal_age_minutes,
        tier_map=_parse_tier_map(args.tier_map),
        tier_size_multipliers=_parse_multipliers(args.tier_size_multipliers),
        min_confidence=args.min_confidence,
        active_coins=active_coins,
    ).run_forever()


if __name__ == "__main__":
    main()
