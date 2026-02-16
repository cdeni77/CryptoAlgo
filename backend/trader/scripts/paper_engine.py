"""Paper trading engine that consumes live `signals` rows and persists state transitions."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from core.pg_writer import PgWriter
from scripts.train_model import Config, calculate_coinbase_fee, calculate_n_contracts, calculate_pnl_exact, get_contract_spec

logger = logging.getLogger("paper_engine")


@dataclass
class EngineState:
    cash_balance: float = 10_000.0
    realized_pnl: float = 0.0
    last_signal_id: int = 0


class PaperTradingEngine:
    def __init__(self, poll_seconds: float = 2.0):
        self.poll_seconds = poll_seconds
        self.writer = PgWriter()
        self.config = Config()
        self.state = EngineState()

    def _fill_price(self, base_price: float, side: str) -> float:
        slip = self.config.slippage_bps / 10000.0
        if side == "long":
            return base_price * (1.0 + slip)
        return base_price * (1.0 - slip)

    def _simulate_fill(self, signal) -> None:
        if signal.direction not in {"long", "short"}:
            return

        side = signal.direction
        direction = 1 if side == "long" else -1
        price = float(signal.price_at_signal or 0)
        if price <= 0:
            return

        contracts = signal.contracts_suggested or calculate_n_contracts(
            equity=max(self.state.cash_balance, 100.0),
            price=price,
            symbol=signal.coin,
            config=self.config,
        )
        if contracts <= 0:
            return

        spec = get_contract_spec(signal.coin)
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

        open_position = self.writer.get_open_paper_position(signal.coin)
        if open_position and open_position.side != side:
            close_dir = 1 if open_position.side == "long" else -1
            _, _, _, pnl_dollars, close_notional = calculate_pnl_exact(
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
            )

        self.state.cash_balance -= fee

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
        )

        open_positions = self.writer.count_open_positions()
        self.writer.write_paper_equity_point(
            equity=self.state.cash_balance + self.state.realized_pnl,
            cash_balance=self.state.cash_balance,
            unrealized_pnl=0.0,
            realized_pnl=self.state.realized_pnl,
            open_positions=open_positions,
            timestamp=datetime.now(timezone.utc),
        )
        self.writer.mark_signal_acted(signal.id)

    def run_forever(self) -> None:
        logger.info("starting paper engine loop (poll=%ss)", self.poll_seconds)
        while True:
            signals = self.writer.get_unprocessed_signals(self.state.last_signal_id)
            for signal in signals:
                self.state.last_signal_id = max(self.state.last_signal_id, signal.id)
                self._simulate_fill(signal)
            time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="paper trading engine")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    PaperTradingEngine(poll_seconds=args.poll_seconds).run_forever()


if __name__ == "__main__":
    main()
