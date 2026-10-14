"""The serving store: what the API and the frontend read.

PostgreSQL, mutable, holding what a dashboard shows: predictions made, positions
taken, the account and the venue's ledger. `core/datastore.py` is the separate
research store.

A binary needs no mark price, unrealised P&L, funding or leverage columns:
it is bought once, cannot lose more than its stake, and settles once at $1 or $0.
Refusals are stored as well as trades.

These models are duplicated in `backend/api/models/` for container isolation.
`tests/test_orm_parity.py` fails when the two copies diverge.
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

from core.book import won as _won

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

    # What the forecast was judged against. `price_source` separates a backtest
    # (the calibrated baseline stands in for the market) from a real book.
    # `market_probability` (the venue's mid) and `outcome` are recorded on every
    # window, traded or refused, so the market's own probability can be scored.
    # The traded subset alone is a selected sample.
    # The asks are what a trade would cost, for expected value, not calibration.
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

    A rolling window for the screen; the research store keeps the authoritative
    bars, and `prune` drops what has scrolled off.
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

    `mode` is 'paper' or 'live'. Every surface that shows a number from this row
    must say which: a live account that looks like a paper one is the worst
    failure this schema could permit.
    """

    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, nullable=False, default='paper', index=True)
    starting_bankroll = Column(Float, nullable=False, default=100.0)
    bankroll = Column(Float, nullable=False, default=100.0)
    staked = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    # The epoch the dashboard reads from. Not a deletion: the API filters on or
    # after this instant and the rows stay, because `predictions` and the venue
    # ledger are evidence that cannot be rebuilt. Clearing it restores the full
    # history. `scripts/evaluate` and `scripts/promote` never read it.
    reset_at = Column(DateTime(timezone=True), nullable=True)
    fees_paid = Column(Float, nullable=False, default=0.0)
    halted = Column(Boolean, nullable=False, default=False)
    halted_reason = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())


class EquityPoint(Base):
    """Equity at a settlement. Never marked to a model probability.

    Marking an open binary at our own forecast books believed edge as profit.
    `staked` is carried at cost.
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
    """A row of the reliability table.

    The system trades its confident predictions, so per-band calibration matters
    more than the average.
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
    """A live order: what was decided, what was sent, and what came back.

    `status` moves new -> placed -> filled, or to skipped, killed or unknown.
    The ticket exists from the moment an order is sent, so a crash before the
    position is booked still shows as exposure.
    """

    __tablename__ = 'order_tickets'

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False)
    offset_minutes = Column(Integer, nullable=False)

    # The venue's identifier for the market, resolved by asking the venue which
    # market covers this window, never built from a ticker pattern.
    market_ticker = Column(String, nullable=True, index=True)
    venue_order_id = Column(String, nullable=True)

    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    # The price the decision was sized at, and the worst price still worth
    # paying. A ticket without a limit is an instruction to pay any price.
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


class VenueFill(Base):
    """One execution, as the venue recorded it.

    `Position` holds what we believed we bought; this holds the account of record.
    Where they differ (partial fills, better fills, mispriced fees), this row is right.

    Keyed on the venue's `trade_id`, so a re-sync is idempotent: the live and
    historical tiers overlap, and a fill counted twice doubles a cost basis.
    `raw` keeps the whole payload, including fields the parser does not read.
    """

    __tablename__ = 'venue_fills'

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String, nullable=False, unique=True, index=True)
    order_id = Column(String, nullable=True, index=True)
    ticker = Column(String, nullable=False, index=True)

    # 'up' or 'down', our vocabulary for the venue's yes/no. Translated once, in
    # the parser: a mislabelled side inverts a trade's P&L.
    side = Column(String, nullable=False)
    action = Column(String, nullable=True)
    # Float, because the wire encoding is fixed-point (`count_fp`: "5.00").
    # An Integer column would truncate a fractional count.
    contracts = Column(Float, nullable=False)
    # Dollars per contract, for `side`. Null when the price did not parse:
    # an unknown price is not a zero price.
    price = Column(Float, nullable=True)
    is_taker = Column(Boolean, nullable=True)
    created_time = Column(DateTime(timezone=True), nullable=True, index=True)

    raw = Column(JSON, nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_fills_ticker_time', 'ticker', 'created_time'),
    )


class VenueSettlement(Base):
    """One market resolved, as the venue paid it. This is the P&L.

    A binary held to expiry pays one fee and settles once, so this row is a
    position's complete economic history: cost basis, payout, fee.

    `pnl` is stored, not derived on read: the API runs in another container and
    cannot import the trader, so storing keeps one definition of the arithmetic.
    Null `pnl` means the venue left a field absent. It is not break-even, and
    nothing downstream may read it as zero.
    """

    __tablename__ = 'venue_settlements'

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, unique=True, index=True)
    event_ticker = Column(String, nullable=True)
    # 'yes' or 'no', as the venue resolved it.
    market_result = Column(String, nullable=True)

    yes_contracts = Column(Float, nullable=False, default=0.0)
    no_contracts = Column(Float, nullable=False, default=0.0)
    yes_cost = Column(Float, nullable=True)
    no_cost = Column(Float, nullable=True)
    revenue = Column(Float, nullable=True)
    fee_cost = Column(Float, nullable=True)
    # revenue - (yes_cost + no_cost) - fee_cost. See the class docstring for why
    # it is a column, and `Settlement.pnl` for why the fee is subtracted.
    pnl = Column(Float, nullable=True)

    settled_time = Column(DateTime(timezone=True), nullable=True, index=True)

    # Our own books beside the venue's: a growing gap is a mispriced fee or a
    # settlement our Coinbase proxy called differently. Null when no position of
    # ours maps to this ticker, meaning the venue settled something we never booked.
    position_id = Column(Integer, nullable=True, index=True)
    our_pnl = Column(Float, nullable=True)

    raw = Column(JSON, nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_settlements_settled', 'settled_time'),
    )


class VenueBalance(Base):
    """The venue's balance, sampled every cycle, for the account chart.

    Cash, not equity: an entry debits it and settlement credits it back.
    `exchange_index` is recorded because balances are local to a shard: a total
    across shards is not money a single order can draw on.
    """

    __tablename__ = 'venue_balance'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    balance = Column(Float, nullable=False)
    exchange_index = Column(Integer, nullable=True)
    # Our own bankroll at the same instant, and the difference. Stored because
    # a drift that grows (an unrecorded fill, a partial, a mispriced fee) is a
    # trend a log line cannot show.
    our_bankroll = Column(Float, nullable=True)
    drift = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_balance_timestamp_desc', 'timestamp'),
    )

# Indexes added after a table first shipped. Applied by `_run_migrations` and
# asserted identical to the API's list by test_orm_parity.
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
    # The venue ledger. Reads are "newest first" or "since a timestamp".
    'CREATE INDEX IF NOT EXISTS ix_venue_fills_created_desc '
    'ON venue_fills (created_time DESC)',
    'CREATE INDEX IF NOT EXISTS ix_venue_settlements_settled_desc '
    'ON venue_settlements (settled_time DESC)',
    'CREATE INDEX IF NOT EXISTS ix_venue_balance_recent '
    'ON venue_balance (timestamp DESC)',
    # No ADD COLUMN migrations here: SQLite cannot parse `ADD COLUMN IF NOT
    # EXISTS`, so the test suite would never run the statement production runs.
    # New columns go through `PgWriter.ADDED_COLUMNS`.
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
        self._ensure_columns()
        self._run_migrations()

    #: Columns added to an existing table after it was first created.
    #: `(table, column, type)`, applied only when genuinely absent.
    ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ('account', 'reset_at', 'TIMESTAMP WITH TIME ZONE'),
    )

    def _ensure_columns(self) -> None:
        """Add columns that `create_all` cannot: it creates missing tables, never columns.

        Not in `MIGRATIONS`, which holds only portable index statements.
        The existence check uses the dialect-neutral inspector, and the
        `ALTER` is bare `ADD COLUMN`, which SQLite and Postgres both accept.
        """
        from sqlalchemy import inspect as _inspect

        inspector = _inspect(self._engine)
        existing = set(inspector.get_table_names())
        for table, column, type_ in self.ADDED_COLUMNS:
            if table not in existing:
                continue
            have = {c['name'] for c in inspector.get_columns(table)}
            if column in have:
                continue
            with self._engine.begin() as con:
                con.exec_driver_sql(
                    f'ALTER TABLE {table} ADD COLUMN {column} {type_}')
            logger.info('added %s.%s', table, column)

    def _run_migrations(self) -> None:
        """Apply the index migrations. Failures raise, except on SQLite.

        SQLite (the test database) lacks some dialect features, so a failure
        there is tolerated and logged. On Postgres every failure is loud.
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

        A retried cycle must not double-count a refusal or a trade.
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

        Two traders on one account would both size off the same bankroll.
        A Postgres session-level advisory lock is released when the connection
        drops, so a killed trader cannot strand it. SQLite (tests) has no
        advisory lock, so this yields with a warning.
        """
        if self._engine.dialect.name != 'postgresql':
            logger.warning('no advisory lock on %s; single-writer is unenforced',
                           self._engine.dialect.name)
            yield True
            return
        # Arbitrary but stable, and distinct from the API's schema-bootstrap
        # lock in backend/api/app.py, so the two cannot block each other.
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
                       note: Optional[str] = None,
                       venue_order_id: Optional[str] = None) -> None:
        with self._session() as session:
            row = session.get(OrderTicket, ticket_id)
            if row is None:
                return
            row.status = status
            if venue_order_id:
                row.venue_order_id = str(venue_order_id)
            # Only overwrite a recorded fill with another recorded fill. A later
            # call with None (e.g. a rejected duplicate marked `skipped`) must
            # not erase a real fill.
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

        Get-or-create, matching `write_ticket`, so a second attempt is a visible
        no-op instead of an IntegrityError.
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
        """Resolve one position, credit the account, and return its P&L.

        Returns None if the position was already settled, so a re-run of
        `settle_due` is idempotent. The payout is credited in the same
        transaction as the outcome, so a crash cannot leave a win uncredited.
        """
        with self._session() as session:
            row = session.get(Position, position_id)
            if row is None or row.outcome != Outcome.PENDING.value:
                return None
            won = _won(side=row.side, settled_up=settled_up)
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

        Read from the durable store, because the process restarts and the window
        does not. Tickets count as well as positions, including 'unknown' (a
        failed POST may have been accepted). 'skipped' and 'killed' tickets do not
        count: the venue took nothing.
        """
        with self._session() as session:
            symbols: set[str] = set()
            stake = 0.0
            for row in (session.query(Position)
                        .filter(Position.window_open == window_open)):
                symbols.add(str(row.symbol))
                stake += float(row.outlay or 0.0)
            dead = ('skipped', 'killed')
            for row in (session.query(OrderTicket)
                        .filter(OrderTicket.window_open == window_open)
                        .filter(OrderTicket.status.notin_(dead))):
                if str(row.symbol) not in symbols:
                    symbols.add(str(row.symbol))
                    stake += float(row.expected_cost or 0.0)
            return frozenset(symbols), stake, len(symbols)

    def first_position_time(self) -> Optional[datetime]:
        """When our books start, or None if they hold nothing.

        `sync_venue` uses this to separate a coverage gap from a P&L error: the
        venue remembers every settlement, while this store is reset between
        experiments. Returned tz-aware in UTC, because SQLite returns naive datetimes.
        """
        with self._session() as session:
            earliest = session.query(func.min(Position.window_open)).scalar()
        if earliest is None:
            return None
        if earliest.tzinfo is None:
            return earliest.replace(tzinfo=timezone.utc)
        return earliest.astimezone(timezone.utc)

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
        """The highest cumulative realised P&L this account has reached.

        Derived from settled positions, so it cannot drift from the ledger.
        Bounded by `account.reset_at` when set, so it spans the same period as
        `account.realized_pnl`. A live-account drawdown breaker only.
        """
        with self._session() as session:
            epoch = (session.query(Account.reset_at)
                     .order_by(Account.id).limit(1).scalar())
            rows = (session.query(Position.pnl)
                    .filter(Position.outcome != Outcome.PENDING.value)
                    .filter(Position.settled_at.isnot(None)))
            if epoch is not None:
                rows = rows.filter(Position.settled_at >= epoch)
            rows = rows.order_by(Position.settled_at.asc())
            peak = running = 0.0
            for (value,) in rows:
                running += float(value or 0.0)
                peak = max(peak, running)
            return peak

    # ---- the market benchmark -------------------------------------------
    def scored_against_market(self) -> list[tuple]:
        """Every settled window where the venue's own probability was recorded.

        `(symbol, window_open, offset_minutes, market_probability,
        baseline_probability, model_probability, outcome, decision_time)`.

        Traded and refused: the traded subset is selected by the model under test.
        `decision_time` is carried because the quote is read when the cycle gets
        there, which can be later than `window_open + offset`.
        """
        with self._session() as session:
            return list(
                session.query(
                    Prediction.symbol, Prediction.window_open,
                    Prediction.offset_minutes, Prediction.market_probability,
                    Prediction.baseline_probability, Prediction.model_probability,
                    Prediction.outcome, Prediction.decision_time)
                .filter(Prediction.market_probability.isnot(None))
                .filter(Prediction.outcome.isnot(None))
                .order_by(Prediction.window_open)
            )


    def windows_awaiting_outcome(self, now: datetime, *, limit: int = 500
                                ) -> list[tuple[str, datetime, datetime, float]]:
        """Settled windows whose predictions still have no realised outcome.

        One row per (symbol, window), with the strike, newest first. The outcome
        beside `market_probability` lets the venue's probability be scored on
        every window, not only the traded ones.
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

    def correct_positions_from_venue(self, labels) -> tuple[int, int, float]:
        """Re-settle positions the venue graded differently, and fix the P&L.

        `settle_due` falls back to our Coinbase label when the venue's settlement
        has not arrived; this re-grades those positions once it does.
        `account.realized_pnl` feeds the daily-loss and drawdown breakers, so it
        must match the venue's grading.

        `bankroll` is not adjusted: `adopt_venue_balance` already writes the
        venue's balance, the account of record.

        `labels` is an iterable of `(symbol, window_open, settled_up)`.
        Returns `(examined, corrected, pnl_delta)`.
        """
        examined = corrected = 0
        delta_total = 0.0
        with self._session() as session:
            for symbol, window_open, settled_up in labels:
                if settled_up is None:
                    continue
                examined += 1
                rows = (session.query(Position)
                        .filter(Position.symbol == symbol,
                                Position.window_open == window_open,
                                Position.settled_at.isnot(None))
                        .all())
                for row in rows:
                    if bool(row.settled_up) == bool(settled_up):
                        continue
                    won = _won(side=row.side, settled_up=bool(settled_up))
                    payout = float(row.contracts) if won else 0.0
                    before = float(row.pnl or 0.0)
                    row.settled_up = bool(settled_up)
                    row.payout = payout
                    row.pnl = payout - float(row.outlay or 0.0)
                    row.outcome = Outcome.WON.value if won else Outcome.LOST.value
                    delta_total += float(row.pnl) - before
                    corrected += 1
            # Restate the invariant on every pass, recomputed from positions,
            # not nudged by the delta: the labels reach back past the epoch, and
            # restating also repairs any existing drift.
            #
            #     realized_pnl == sum(positions.pnl where settled_at >= epoch)
            epoch = (session.query(Account.reset_at)
                     .order_by(Account.id).limit(1).scalar())
            total = session.query(func.coalesce(func.sum(Position.pnl), 0.0)) \
                .filter(Position.settled_at.isnot(None))
            if epoch is not None:
                total = total.filter(Position.settled_at >= epoch)
            session.query(Account).update(
                {Account.realized_pnl: float(total.scalar() or 0.0)},
                synchronize_session=False,
            )
            session.commit()
        return examined, corrected, delta_total

    def correct_outcomes_from_venue(self, labels) -> tuple[int, int]:
        """Overwrite `Prediction.outcome` where the venue settled differently.

        The market gates read this column, so it must carry the venue's label,
        as the backtest does. `set_window_outcome` fills only NULLs; this
        overwrites, and only where the labels differ, so a re-run is free and
        the count is the disagreement.

        `labels` is an iterable of `(symbol, window_open, settled_up)`.
        Returns `(examined, corrected)`.
        """
        examined = corrected = 0
        with self._session() as session:
            for symbol, window_open, settled_up in labels:
                if settled_up is None:
                    continue
                examined += 1
                value = 1 if settled_up else 0
                n = (session.query(Prediction)
                     .filter(Prediction.symbol == symbol,
                             Prediction.window_open == window_open,
                             Prediction.outcome.isnot(None),
                             Prediction.outcome != value)
                     .update({Prediction.outcome: value},
                             synchronize_session=False))
                corrected += int(n)
            session.commit()
        return examined, corrected

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
            # The stored mode must match. The bankroll history belongs to one
            # mode, so switching would report one mode's money under the other's name.
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

        For increments (a stake, a payout, a fee), use `adjust_account`: this
        method reads and writes back, so overlapping callers lose a change.
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
        """Apply increments to the account in one atomic SQL statement.

        The arithmetic runs in the database, so overlapping writers cannot lose
        an update.
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

    # ---- the venue ledger -----------------------------------------------
    #
    # Written by the live loop every cycle and by `scripts.sync_venue` for the
    # paginated backfill. Both upsert on the venue's own key, so running them
    # together, or one twice, converges.

    def upsert_venue_fills(self, rows: Iterable[dict]) -> int:
        """Upsert fills on `trade_id`. Returns the number of rows written.

        Idempotent: the live and historical tiers overlap, so one fill can arrive
        twice. A conflict overwrites the row, so a badly parsed row heals on the
        next sync.
        """
        written = 0
        with self._session() as session:
            for row in rows:
                trade_id = row.get('trade_id')
                if not trade_id:
                    logger.warning('a venue fill arrived with no trade_id; '
                                   'skipping rather than inventing a key')
                    continue
                existing = session.query(VenueFill).filter_by(
                    trade_id=trade_id).one_or_none()
                if existing is not None:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    existing.synced_at = utcnow()
                else:
                    session.add(VenueFill(**row))
                written += 1
            session.commit()
        return written

    def upsert_venue_settlements(self, rows: Iterable[dict]) -> int:
        """Upsert settlements on `ticker`. Returns the number of rows written."""
        written = 0
        with self._session() as session:
            for row in rows:
                ticker = row.get('ticker')
                if not ticker:
                    continue
                existing = session.query(VenueSettlement).filter_by(
                    ticker=ticker).one_or_none()
                if existing is not None:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    existing.synced_at = utcnow()
                else:
                    session.add(VenueSettlement(**row))
                written += 1
            session.commit()
        return written

    def venue_ledger_keys(self) -> tuple[set[str], set[str]]:
        """What is already stored, as two sets, in two queries.

        The caller filters the venue's response against these to skip unchanged
        rows. The settlement set holds only tickers with a non-null `pnl`, so an
        incomplete row is retried and can heal.
        """
        with self._session() as session:
            trade_ids = {tid for (tid,) in session.query(VenueFill.trade_id)}
            tickers = {t for (t,) in session.query(VenueSettlement.ticker)
                       .filter(VenueSettlement.pnl.is_not(None))}
        return trade_ids, tickers

    def venue_fills(self, *, since: Optional[datetime] = None,
                    limit: int = 200) -> list[VenueFill]:
        with self._session() as session:
            query = session.query(VenueFill)
            if since is not None:
                query = query.filter(VenueFill.created_time >= since)
            return list(query.order_by(VenueFill.created_time.desc()).limit(limit))

    def venue_settlements(self, *, since: Optional[datetime] = None,
                          limit: Optional[int] = None) -> list[VenueSettlement]:
        """Settled markets, newest first. `limit=None` for the whole ledger.

        Unlimited by default: a P&L total is a sum over everything, and a limit
        would report the latest N trades as the account's profit.
        """
        with self._session() as session:
            query = session.query(VenueSettlement)
            if since is not None:
                query = query.filter(VenueSettlement.settled_time >= since)
            query = query.order_by(VenueSettlement.settled_time.desc())
            if limit is not None:
                query = query.limit(limit)
            return list(query)

    def position_for_ticker(self, ticker: str):
        """Our own position for a venue ticker, via the ticket that placed it.

        `positions` has no ticker column, so the join goes through
        `order_tickets`. Returns None when nothing maps: the venue settled a
        market we never booked.
        """
        with self._session() as session:
            ticket = (session.query(OrderTicket)
                      .filter(OrderTicket.market_ticker == ticker)
                      .order_by(OrderTicket.created_at.desc())
                      .first())
            if ticket is None:
                return None
            return (session.query(Position)
                    .filter(Position.symbol == ticket.symbol,
                            Position.window_open == ticket.window_open)
                    .one_or_none())

    def write_venue_balance(self, *, timestamp: datetime, balance: float,
                            exchange_index: Optional[int] = None,
                            our_bankroll: Optional[float] = None) -> int:
        """Sample the venue's balance; the drift is computed here, once.

        Appended, not upserted: each cycle's reading is a separate observation.
        """
        drift = (None if our_bankroll is None
                 else float(balance) - float(our_bankroll))
        with self._session() as session:
            row = VenueBalance(
                timestamp=timestamp, balance=float(balance),
                exchange_index=exchange_index, our_bankroll=our_bankroll,
                drift=drift,
            )
            session.add(row)
            session.commit()
            return row.id

    def venue_balances_since(self, since: datetime) -> list[VenueBalance]:
        with self._session() as session:
            return list(
                session.query(VenueBalance)
                .filter(VenueBalance.timestamp >= since)
                .order_by(VenueBalance.timestamp)
            )

    def latest_venue_balance(self) -> Optional[VenueBalance]:
        with self._session() as session:
            return (session.query(VenueBalance)
                    .order_by(VenueBalance.timestamp.desc())
                    .first())

    def prune_venue_balances(self, before: datetime) -> int:
        """Drop balance samples that have scrolled off the chart.

        A rolling window for a screen. Settlements are the record and are never pruned.
        """
        with self._session() as session:
            n = (session.query(VenueBalance)
                 .filter(VenueBalance.timestamp < before).delete())
            session.commit()
            return int(n)

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
