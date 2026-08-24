"""The serving store: what the API and the frontend read.

Two stores, two jobs. `core/datastore.py` is the research store — immutable
Parquet, point-in-time reads, the thing feature builds and backtests run on.
This is PostgreSQL, mutable, and holds only what a dashboard needs to show:
predictions we actually made, positions we actually took, and the account.

**A binary needs almost none of what a perpetual future needed.** The previous
schema carried mark price, unrealised PnL, funding accrual, take-profit and
stop-loss levels, notional and leverage. None of them exists here. A contract is
bought once at a known price, cannot lose more than its stake, accrues nothing,
and settles once at $1 or $0. Carrying those columns would mean writing zeros
into them forever and inviting a dashboard to render a zero as a measurement.

**Refusals are stored, not just trades.** A dashboard showing only trades cannot
show that the system declined 99% of windows because the forecast did not cover
the fee — which is the most informative thing it could say, and was on the perp
system.

**These models are duplicated in `backend/api/models/` for container isolation,
and `tests/test_orm_parity.py` fails when they diverge** in columns, types,
nullability, defaults or the migration list. That test exists because
`wallet.balance` once differed by a factor of ten between the two copies, so
whichever container created the row decided the account's starting balance.
"""

from __future__ import annotations

import enum
import logging
from contextlib import contextmanager
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text,
    UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Outcome(str, enum.Enum):
    """A position's terminal state. There is no third possibility."""

    PENDING = 'pending'
    WON = 'won'
    LOST = 'lost'


class Prediction(Base):
    """One scored decision point: (symbol, window, offset), traded or refused."""

    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False, index=True)
    offset_minutes = Column(Integer, nullable=False)
    decision_time = Column(DateTime(timezone=True), nullable=False)

    # the barrier, as observed
    strike = Column(Float, nullable=False)
    last_price = Column(Float, nullable=False)
    displacement = Column(Float, nullable=False)
    sigma_remaining = Column(Float, nullable=True)
    z_score = Column(Float, nullable=True)

    # the two probabilities, and never one without the other
    baseline_probability = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)

    # What the forecast was actually judged against, and where that came from.
    # A backtest has no quotes and stands the calibrated baseline in for the
    # market; a live decision reads the real book. Those are different claims and
    # a row that does not distinguish them makes a backtest look like a fill.
    # The venue's own view, and the realised answer. These three exist for one
    # purpose: to eventually score the MARKET's probability against the outcome,
    # on every window, traded or refused.
    #
    # That is the only economically meaningful benchmark and nothing in this
    # system could compute it. `market_probability` was written and read by
    # nothing, there was no `outcome` column at all, and `positions` only covers
    # the ~6% of windows that traded — a selected sample. Beating F(x/sigma) says
    # nothing about beating the price you would actually pay.
    #
    # `market_probability` is the MID (the venue's belief). The two asks are what
    # a trade would cost, which is a different question and needed for expected
    # value rather than for calibration.
    market_probability = Column(Float, nullable=True)
    market_ask_up = Column(Float, nullable=True)
    market_ask_down = Column(Float, nullable=True)
    outcome = Column(Integer, nullable=True)
    price_source = Column(String, nullable=False, default='baseline')

    # the decision
    reason = Column(String, nullable=False, index=True)
    traded = Column(Boolean, nullable=False, default=False)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    effective_cost = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    contracts = Column(Integer, nullable=True)

    model_version = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'window_open', 'offset_minutes',
                         name='uq_prediction_point'),
        Index('ix_predictions_traded_window', 'traded', 'window_open'),
    )


class MinutePrice(Base):
    """One minute of price per symbol, for the window chart.

    The prediction rows carry `last_price` at four offsets, which is enough to
    decide with and far too sparse to draw. A barrier problem's natural picture
    is the path against the line it has to finish above, and that needs every
    minute.

    Cheap: three symbols at one row a minute is 4,320 rows a day. The research
    store keeps the authoritative bars; this is a rolling window for the screen,
    and `prune` drops what has scrolled off.
    """

    __tablename__ = 'minute_prices'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    minute = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'minute', name='uq_minute_price'),
        Index('ix_minute_prices_symbol_minute', 'symbol', 'minute'),
    )


class Position(Base):
    """One contract purchase, held to settlement. No marking, no exit."""

    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False, index=True)
    offset_minutes = Column(Integer, nullable=False)

    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    outlay = Column(Float, nullable=False)     # everything paid, fee included
    fee = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    baseline_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)

    outcome = Column(String, nullable=False, default=Outcome.PENDING.value, index=True)
    settled_up = Column(Boolean, nullable=True)
    payout = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'window_open', name='uq_position_window'),
        Index('ix_positions_open', 'outcome', 'settle_time'),
    )


class Account(Base):
    """The single account row: the bankroll, and what it is now.

    `mode` is 'paper' or 'live' and it is not cosmetic. A live account holds real
    money, so every surface that shows a number from this row has to be able to
    say which kind it is — a live account that renders identically to a paper one
    is the single worst failure this schema could permit.
    """

    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, nullable=False, default='paper', index=True)
    starting_bankroll = Column(Float, nullable=False, default=100.0)
    bankroll = Column(Float, nullable=False, default=100.0)
    staked = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    fees_paid = Column(Float, nullable=False, default=0.0)
    halted = Column(Boolean, nullable=False, default=False)
    halted_reason = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())


class EquityPoint(Base):
    """Equity at a settlement. Never marked to a model probability.

    Marking an open binary at our own forecast books the edge we believe in as
    profit we have not received, which is how a losing system draws a rising
    equity curve. `staked` is carried at cost.
    """

    __tablename__ = 'equity_curve'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    bankroll = Column(Float, nullable=False)
    staked = Column(Float, nullable=False, default=0.0)
    open_positions = Column(Integer, nullable=False, default=0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRun(Base):
    """A promotion attempt. Blocked ones are kept: they are the trial count."""

    __tablename__ = 'model_runs'

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    installed = Column(Boolean, nullable=False, default=False)
    forced = Column(Boolean, nullable=False, default=False)
    force_reason = Column(Text, nullable=True)

    folds = Column(Integer, nullable=True)
    windows_evaluated = Column(Integer, nullable=True)
    log_loss_skill = Column(Float, nullable=True)
    log_loss_skill_se = Column(Float, nullable=True)
    folds_positive = Column(Integer, nullable=True)
    calibration_error = Column(Float, nullable=True)
    residual_scale = Column(Float, nullable=True)
    control_gain_share = Column(Float, nullable=True)
    sharpe = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    gates = Column(JSON, nullable=True)
    failed_gates = Column(Text, nullable=True)
    provenance = Column(JSON, nullable=True)


class CalibrationBin(Base):
    """A row of the reliability table, for the one chart that cannot be faked.

    A model can hit the base rate exactly while being wrong at every level of
    confidence. Since this system only trades its confident predictions, a
    miscalibration in the 0.85-0.95 band matters far more than the headline.
    """

    __tablename__ = 'calibration'

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)        # 'model' or 'baseline'
    bin_low = Column(Float, nullable=False)
    bin_high = Column(Float, nullable=False)
    predicted = Column(Float, nullable=True)
    observed = Column(Float, nullable=True)
    count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('model_version', 'source', 'bin_low',
                         name='uq_calibration_bin'),
    )


class OrderTicket(Base):
    """A live decision, as something a person can act on.

    This repository has no Kalshi API client and no Kalshi credentials, so in
    live mode the engine does not place orders — it writes a ticket carrying
    everything needed to place one, and records what came back. Being explicit
    about that boundary is the point: a system that silently did nothing while
    appearing to trade would be worse than one that says a human is in the loop.

    `status` moves new -> placed -> filled, or new -> skipped. Nothing here
    advances it automatically; the dashboard does, or a future order client will.
    """

    __tablename__ = 'order_tickets'

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False)
    offset_minutes = Column(Integer, nullable=False)

    # The venue's own identifier for the market, resolved by asking the venue
    # which market opens and closes on this window rather than by building a
    # ticker from a pattern. A pattern is a guess that keeps working until the
    # venue renames a series.
    market_ticker = Column(String, nullable=True, index=True)
    venue_order_id = Column(String, nullable=True)

    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    # The price the decision was sized at, and the worst price still worth
    # paying. A ticket without a limit is an instruction to pay anything.
    limit_price = Column(Float, nullable=False)
    max_price = Column(Float, nullable=False)
    expected_cost = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)

    status = Column(String, nullable=False, default='new', index=True)
    filled_contracts = Column(Integer, nullable=True)
    filled_price = Column(Float, nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    note = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'window_open', name='uq_ticket_window'),
        Index('ix_order_tickets_status_created', 'status', 'created_at'),
    )


# Indexes and constraints added after a table first shipped. Applied by
# `_run_migrations` and asserted identical to the API's list by test_orm_parity —
# an index that exists in one container and not the other is a query that is
# fast in development and a table scan in production.
MIGRATIONS: tuple[str, ...] = (
    'CREATE INDEX IF NOT EXISTS ix_predictions_reason_window '
    'ON predictions (reason, window_open DESC)',
    'CREATE INDEX IF NOT EXISTS ix_positions_settled_at '
    'ON positions (settled_at DESC)',
    'CREATE INDEX IF NOT EXISTS ix_equity_curve_timestamp_desc '
    'ON equity_curve (timestamp DESC)',
    'CREATE INDEX IF NOT EXISTS ix_minute_prices_recent '
    'ON minute_prices (symbol, minute DESC)',
    'CREATE INDEX IF NOT EXISTS ix_order_tickets_window '
    'ON order_tickets (window_open DESC)',
    # NOTE: the market-benchmark columns on `predictions` (market_ask_up,
    # market_ask_down, outcome) deliberately have NO migration here.
    #
    # `ADD COLUMN IF NOT EXISTS` is not parseable on SQLite, so it would be
    # skipped by the dialect guard in `_run_migrations` and production would be
    # running a statement the test suite had never executed — which is exactly
    # the trap `tests/test_orm_parity.py` was written to close, and exactly what
    # the previous all-ADD-COLUMN list did.
    #
    # The ORM gives them to a fresh database via `create_all`, and the serving
    # store is regenerated telemetry rather than a record of account, so an
    # existing one is wiped: `docker compose down -v`. The research store and the
    # scraped SQLite are the irreplaceable parts and neither is touched.
)


class TraderAlreadyRunning(RuntimeError):
    """Another process holds the trading lock on this database."""


class AccountModeMismatch(RuntimeError):
    """The stored account was opened in a different mode than this run asks for."""


class PgWriter:
    """Everything the trader writes to the serving store."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.getenv('DATABASE_URL')
        if not url:
            raise ValueError('DATABASE_URL is not set')
        self._engine = create_engine(url, pool_pre_ping=True, future=True)
        self._sessions = sessionmaker(bind=self._engine, future=True)
        Base.metadata.create_all(self._engine)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply the additive migrations, tolerating only SQLite's dialect gaps.

        This used to catch bare `Exception` and log at `debug`, so *any* migration
        failure against real Postgres — a type mismatch, a lock timeout, a
        permissions problem — vanished at a level most deployments do not
        capture, and the failure resurfaced later as a confusing runtime error on
        the accounting database instead of a loud one at startup.

        The tolerance it was written for is real but narrow: `MIGRATIONS` uses
        `ADD COLUMN IF NOT EXISTS` and similar, which SQLite does not parse, and
        the tests run on SQLite. So tolerate it on SQLite and nowhere else.
        """
        sqlite = self._engine.dialect.name == 'sqlite'
        with self._engine.begin() as connection:
            for statement in MIGRATIONS:
                try:
                    connection.execute(text(statement))
                except Exception as exc:  # noqa: BLE001 - re-raised unless SQLite
                    if not sqlite:
                        logger.error('migration failed on %s: %s\n  %s',
                                     self._engine.dialect.name, exc, statement)
                        raise
                    logger.debug('migration skipped on sqlite (%s): %s', exc, statement)

    def _session(self) -> Session:
        return self._sessions()

    # ---- predictions ----------------------------------------------------
    def write_prediction(self, **fields) -> int:
        """Upsert one decision point. Idempotent on (symbol, window, offset).

        Idempotent because the orchestrator can re-run a cycle — a retry after a
        network failure must not double-count a refusal or, worse, a trade.
        """
        with self._session() as session:
            existing = session.query(Prediction).filter_by(
                symbol=fields['symbol'], window_open=fields['window_open'],
                offset_minutes=fields['offset_minutes'],
            ).one_or_none()
            if existing is not None:
                for key, value in fields.items():
                    setattr(existing, key, value)
                session.commit()
                return existing.id
            row = Prediction(**fields)
            session.add(row)
            session.commit()
            return row.id

    def recent_predictions(self, limit: int = 200, *, traded_only: bool = False) -> list[Prediction]:
        with self._session() as session:
            query = session.query(Prediction)
            if traded_only:
                query = query.filter(Prediction.traded.is_(True))
            return list(query.order_by(Prediction.window_open.desc()).limit(limit))

    def refusal_counts(self, since: Optional[datetime] = None) -> dict[str, int]:
        """The funnel, as the dashboard shows it."""
        with self._session() as session:
            query = session.query(Prediction.reason, func.count(Prediction.id))
            if since is not None:
                query = query.filter(Prediction.window_open >= since)
            return {reason: int(count) for reason, count in query.group_by(Prediction.reason)}

    # ---- minute prices --------------------------------------------------
    def write_minute_prices(self, rows: Iterable[dict]) -> int:
        """Upsert minute bars for the chart. Idempotent on (symbol, minute)."""
        written = 0
        with self._session() as session:
            for row in rows:
                existing = session.query(MinutePrice).filter_by(
                    symbol=row['symbol'], minute=row['minute']).one_or_none()
                if existing is not None:
                    for key, value in row.items():
                        setattr(existing, key, value)
                else:
                    session.add(MinutePrice(**row))
                written += 1
            session.commit()
        return written

    def minute_prices(self, symbol: str, since: datetime) -> list[MinutePrice]:
        with self._session() as session:
            return list(
                session.query(MinutePrice)
                .filter(MinutePrice.symbol == symbol, MinutePrice.minute >= since)
                .order_by(MinutePrice.minute)
            )

    def prune_minute_prices(self, before: datetime) -> int:
        """Drop rows that have scrolled off the chart. Called by the engine."""
        with self._session() as session:
            n = session.query(MinutePrice).filter(MinutePrice.minute < before).delete()
            session.commit()
            return int(n)

    # ---- single-writer guard --------------------------------------------
    @contextmanager
    def exclusive_trader_lock(self):
        """Hold a database-wide lock for the duration of a trading process.

        There was no singleton of any kind. `docker-compose` runs a `trader`
        service and nothing stopped an operator also running
        `python -m scripts.live` by hand — two processes, one account, one
        window. Measured, both sized a full position off the same bankroll.

        A Postgres session-level advisory lock is the right primitive: it is
        released automatically when the connection drops, so a killed trader does
        not strand the lock. On SQLite (tests) there is no advisory lock and no
        concurrency to guard, so this yields with a warning rather than pretending.
        """
        if self._engine.dialect.name != 'postgresql':
            logger.warning('no advisory lock on %s; single-writer is unenforced',
                           self._engine.dialect.name)
            yield True
            return
        # An arbitrary but stable key. Distinct from the API's schema-bootstrap
        # lock in backend/api/app.py so the two cannot block each other.
        key = 0x51545241  # 'QTRA'
        connection = self._engine.connect()
        try:
            got = bool(connection.execute(
                text('SELECT pg_try_advisory_lock(:k)'), {'k': key}).scalar())
            if not got:
                raise TraderAlreadyRunning(
                    'another process already holds the trading lock on this '
                    'database. Two traders against one account double every '
                    'position and race on the bankroll. Stop the other one — '
                    'most likely the compose `trader` service — or point this '
                    'run at a different DATABASE_URL.'
                )
            yield True
        finally:
            connection.close()

    # ---- order tickets ---------------------------------------------------
    def write_ticket(self, **fields) -> int:
        """Record a live decision as something a person can place."""
        with self._session() as session:
            existing = session.query(OrderTicket).filter_by(
                symbol=fields['symbol'], window_open=fields['window_open']).one_or_none()
            if existing is not None:
                return existing.id
            row = OrderTicket(**fields)
            session.add(row)
            session.commit()
            return row.id

    def open_tickets(self, limit: int = 50) -> list[OrderTicket]:
        with self._session() as session:
            return list(
                session.query(OrderTicket)
                .filter(OrderTicket.status == 'new')
                .order_by(OrderTicket.created_at.desc()).limit(limit)
            )

    def resolve_ticket(self, ticket_id: int, *, status: str,
                       filled_contracts: Optional[int] = None,
                       filled_price: Optional[float] = None,
                       note: Optional[str] = None) -> None:
        with self._session() as session:
            row = session.get(OrderTicket, ticket_id)
            if row is None:
                return
            row.status = status
            # Only overwrite a recorded fill with another recorded fill. These
            # default to None, so a later call — e.g. the venue rejecting a
            # duplicate submission as `skipped` — used to erase the
            # filled_contracts / filled_price / filled_at of a real, filled,
            # real-money order, destroying the only local record of it.
            if filled_contracts is not None:
                row.filled_contracts = filled_contracts
            if filled_price is not None:
                row.filled_price = filled_price
            if note is not None:
                row.note = note
            if status == 'filled':
                row.filled_at = utcnow()
            session.commit()

    # ---- positions ------------------------------------------------------
    def open_position(self, **fields) -> Optional[int]:
        """Book a position, once, for a (symbol, window). Returns None if it exists.

        A bare insert against `uq_position_window` used to raise `IntegrityError`
        on the second cycle of any window, and `scripts/live.py`'s loop catches
        only `KeyboardInterrupt` — so the process died, *after* the duplicate
        order was already on the wire, and `restart: unless-stopped` brought it
        back to do it again. The constraint was doing its job; the caller was
        not. Get-or-create makes the second attempt a no-op the caller can see,
        matching `write_ticket`.
        """
        with self._session() as session:
            existing = session.query(Position).filter_by(
                symbol=fields['symbol'], window_open=fields['window_open']).one_or_none()
            if existing is not None:
                logger.warning(
                    '%s window %s already holds a position; not booking a second',
                    fields['symbol'], fields['window_open'])
                return None
            row = Position(outcome=Outcome.PENDING.value, **fields)
            session.add(row)
            session.commit()
            return row.id

    def positions_due(self, now: datetime) -> list[Position]:
        with self._session() as session:
            return list(
                session.query(Position)
                .filter(Position.outcome == Outcome.PENDING.value)
                .filter(Position.settle_time <= now)
                .order_by(Position.settle_time)
            )

    def settle_position(self, position_id: int, *, settled_up: bool) -> Optional[float]:
        """Resolve one position, credit the account, and return its PnL.

        Returns None if the position was already settled, which is what makes a
        re-run of `settle_due` idempotent.

        **The payout is credited here, in the same transaction as the outcome.**
        It used to be nowhere: `act_on` debited the stake and nothing ever
        credited a win, so the only two writers of `Account.bankroll` were the
        entry debit and the live venue overwrite. In paper mode there is no venue,
        so the bankroll fell by the stake on every trade and rose never — the
        equity curve decayed to the ruin floor regardless of the win rate, and
        `realized_pnl` was a hard-coded zero rendered as a measurement. The
        backtest (`core/book.py`) always did this correctly, which is exactly why
        no gate caught it: the two accounting paths had diverged and only the
        simulated one was right.

        Crediting inside the position's own transaction is deliberate. A payout
        applied in a second transaction can be lost to a crash between the two,
        and then the position says "won" while the bankroll never saw the money.
        """
        with self._session() as session:
            row = session.get(Position, position_id)
            if row is None or row.outcome != Outcome.PENDING.value:
                return None
            won = settled_up if row.side == 'up' else not settled_up
            payout = float(row.contracts) if won else 0.0
            row.settled_up = settled_up
            row.payout = payout
            row.pnl = payout - row.outlay
            row.outcome = Outcome.WON.value if won else Outcome.LOST.value
            row.settled_at = utcnow()
            pnl = float(row.pnl)
            # Relative UPDATE rather than read-modify-write: the arithmetic
            # happens in the database, so a concurrent debit cannot be lost.
            session.query(Account).update(
                {Account.bankroll: Account.bankroll + payout,
                 Account.realized_pnl: Account.realized_pnl + pnl},
                synchronize_session=False,
            )
            session.commit()
            return pnl

    def entries_for_window(self, window_open: datetime) -> tuple[frozenset[str], float, int]:
        """What is already committed for one window: symbols, stake, count.

        The live loop used to build `WindowExposure()` empty on every cycle, so
        `ALREADY_ENTERED`, `POSITION_LIMIT` and `WINDOW_EXPOSURE` were evaluated
        against nothing and a single window could be entered once per cycle per
        offset — up to twelve times where the backtest permits one. Exposure has
        to come from the durable store, because the process restarts and the
        window does not.

        Tickets count as well as positions: a ticket exists from the moment an
        order is sent, so a crash between sending and booking still shows up.
        """
        with self._session() as session:
            symbols: set[str] = set()
            stake = 0.0
            for row in (session.query(Position)
                        .filter(Position.window_open == window_open)):
                symbols.add(str(row.symbol))
                stake += float(row.outlay or 0.0)
            for row in (session.query(OrderTicket)
                        .filter(OrderTicket.window_open == window_open)
                        .filter(OrderTicket.status != 'skipped')):
                if str(row.symbol) not in symbols:
                    symbols.add(str(row.symbol))
                    stake += float(row.expected_cost or 0.0)
            return frozenset(symbols), stake, len(symbols)

    def open_positions(self) -> list[Position]:
        with self._session() as session:
            return list(
                session.query(Position)
                .filter(Position.outcome == Outcome.PENDING.value)
                .order_by(Position.settle_time)
            )

    def settled_positions_since(self, since: datetime) -> list[Position]:
        with self._session() as session:
            return list(
                session.query(Position)
                .filter(Position.outcome != Outcome.PENDING.value)
                .filter(Position.settled_at >= since)
                .order_by(Position.settled_at.desc())
            )

    def realised_high_water(self) -> float:
        """The highest cumulative realised PnL this account has ever reached.

        Derived from the settlements rather than stored on the account, so it
        needs no column and cannot drift out of sync with the ledger it
        summarises. `account.realized_pnl` is the current value of the same
        series, so the pair gives peak and current without a second source of
        truth.
        """
        with self._session() as session:
            rows = (session.query(Position.pnl)
                    .filter(Position.outcome != Outcome.PENDING.value)
                    .filter(Position.settled_at.isnot(None))
                    .order_by(Position.settled_at.asc()))
            peak = running = 0.0
            for (value,) in rows:
                running += float(value or 0.0)
                peak = max(peak, running)
            return peak

    # ---- the market benchmark -------------------------------------------
    def scored_against_market(self) -> list[tuple]:
        """Every settled window where the venue's own probability was recorded.

        `(symbol, window_open, offset_minutes, market_probability,
        baseline_probability, model_probability, outcome)`.

        Traded *and* refused, which is the point — the traded subset is selected by
        our own model's opinion, and it is our own opinion under test.
        """
        with self._session() as session:
            return list(
                session.query(
                    Prediction.symbol, Prediction.window_open,
                    Prediction.offset_minutes, Prediction.market_probability,
                    Prediction.baseline_probability, Prediction.model_probability,
                    Prediction.outcome)
                .filter(Prediction.market_probability.isnot(None))
                .filter(Prediction.outcome.isnot(None))
                .order_by(Prediction.window_open)
            )


    def windows_awaiting_outcome(self, now: datetime, *, limit: int = 500
                                ) -> list[tuple[str, datetime, datetime, float]]:
        """Settled windows whose predictions still have no realised outcome.

        One row per (symbol, window), with the strike, newest first. This is what
        makes the market benchmark possible: `market_probability` is recorded on
        every window including the refused ones, and without the outcome beside it
        the venue's probability can never be scored. `positions` cannot substitute
        — it only covers the ~6% of windows that traded, which is a selected
        sample and the wrong one.
        """
        with self._session() as session:
            rows = (session.query(Prediction.symbol, Prediction.window_open,
                                  Prediction.settle_time, Prediction.strike)
                    .filter(Prediction.settle_time <= now)
                    .filter(Prediction.outcome.is_(None))
                    .group_by(Prediction.symbol, Prediction.window_open,
                              Prediction.settle_time, Prediction.strike)
                    .order_by(Prediction.window_open.desc())
                    .limit(limit))
            return [(str(a), b, c, float(d)) for a, b, c, d in rows]

    def set_window_outcome(self, symbol: str, window_open: datetime,
                           *, settled_up: bool) -> int:
        """Record the realised outcome on every offset of one window."""
        with self._session() as session:
            n = (session.query(Prediction)
                 .filter(Prediction.symbol == symbol,
                         Prediction.window_open == window_open,
                         Prediction.outcome.is_(None))
                 .update({Prediction.outcome: 1 if settled_up else 0},
                         synchronize_session=False))
            session.commit()
            return int(n)

    # ---- account --------------------------------------------------------
    def ensure_account(self, starting_bankroll: float,
                       *, mode: str = 'paper') -> Account:
        if mode not in ('paper', 'live'):
            raise ValueError(f"account mode must be 'paper' or 'live', got {mode!r}")
        with self._session() as session:
            row = session.query(Account).order_by(Account.id).first()
            if row is None:
                row = Account(mode=mode, starting_bankroll=starting_bankroll,
                              bankroll=starting_bankroll)
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
            # `mode` used to be set only on creation. Paper is the compose
            # default, so the first run created a paper account and every later
            # `--mode live` run kept rendering as paper on every dashboard
            # surface — the failure this schema's own comment calls the worst it
            # could permit. Switching mode is not a display change either: the
            # bankroll history belongs to the other mode, so refuse rather than
            # silently inherit it.
            if row.mode != mode:
                raise AccountModeMismatch(
                    f"this account was opened in {row.mode!r} mode and holds its "
                    f"bankroll and settled history. Running in {mode!r} mode "
                    f"against it would report one mode's money under the other's "
                    f"name. Use a separate DATABASE_URL for {mode!r}, or reset "
                    f"this one deliberately."
                )
            session.refresh(row)
            return row

    def account(self) -> Optional[Account]:
        with self._session() as session:
            return session.query(Account).order_by(Account.id).first()

    def update_account(self, **fields) -> None:
        """Set absolute fields on the account.

        For anything that is an *increment* — a stake debited, a payout credited,
        a fee accrued — use `adjust_account` instead. This method reads the row
        into Python and writes it back, so two overlapping callers lose one
        another's change.
        """
        with self._session() as session:
            row = session.query(Account).order_by(Account.id).first()
            if row is None:
                raise ValueError('no account row; call ensure_account first')
            for key, value in fields.items():
                setattr(row, key, value)
            session.commit()

    def adjust_account(self, *, bankroll_delta: float = 0.0,
                       fees_delta: float = 0.0,
                       realized_delta: float = 0.0) -> None:
        """Apply increments to the account in one atomic statement.

        `bankroll = bankroll - stake` used to be computed in Python between two
        separate transactions. Nothing enforces one trader process, so a
        supervisor restart overlap or an operator running paper and live against
        one `DATABASE_URL` silently dropped one of the two debits. Doing the
        arithmetic in SQL removes the read entirely, so there is no window to
        lose.
        """
        with self._session() as session:
            updated = session.query(Account).update(
                {Account.bankroll: Account.bankroll + bankroll_delta,
                 Account.fees_paid: Account.fees_paid + fees_delta,
                 Account.realized_pnl: Account.realized_pnl + realized_delta},
                synchronize_session=False,
            )
            if not updated:
                raise ValueError('no account row; call ensure_account first')
            session.commit()

    def write_equity_point(self, **fields) -> int:
        with self._session() as session:
            row = EquityPoint(**fields)
            session.add(row)
            session.commit()
            return row.id

    def equity_curve_since(self, since: datetime) -> list[EquityPoint]:
        with self._session() as session:
            return list(
                session.query(EquityPoint)
                .filter(EquityPoint.timestamp >= since)
                .order_by(EquityPoint.timestamp)
            )

    # ---- model runs and calibration -------------------------------------
    def record_model_run(self, **fields) -> int:
        with self._session() as session:
            existing = session.query(ModelRun).filter_by(
                version=fields['version']).one_or_none()
            if existing is not None:
                for key, value in fields.items():
                    setattr(existing, key, value)
                session.commit()
                return existing.id
            row = ModelRun(**fields)
            session.add(row)
            session.commit()
            return row.id

    def latest_model_run(self) -> Optional[ModelRun]:
        with self._session() as session:
            return (session.query(ModelRun)
                    .order_by(ModelRun.created_at.desc()).first())

    def model_runs(self, limit: int = 50) -> list[ModelRun]:
        with self._session() as session:
            return list(session.query(ModelRun)
                        .order_by(ModelRun.created_at.desc()).limit(limit))

    def write_calibration(self, model_version: str, source: str,
                          bins: Iterable[dict]) -> int:
        """Replace the reliability table for one (version, source)."""
        written = 0
        with self._session() as session:
            session.query(CalibrationBin).filter_by(
                model_version=model_version, source=source).delete()
            for row in bins:
                session.add(CalibrationBin(model_version=model_version,
                                           source=source, **row))
                written += 1
            session.commit()
        return written

    def calibration(self, model_version: Optional[str] = None) -> list[CalibrationBin]:
        with self._session() as session:
            query = session.query(CalibrationBin)
            if model_version:
                query = query.filter_by(model_version=model_version)
            return list(query.order_by(CalibrationBin.source, CalibrationBin.bin_low))
