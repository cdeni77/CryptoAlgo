"""One row per unit of collection work, and what happened to it.

**Why this exists.** Four separate claims about what market data exists were
measured, believed and disproved in a single night: Kalshi's book start (wrong
by five months), Polymarket's book start, Kalshi SOL's book start (wrong by
five months in the other direction), and a density curve that appeared to fall
when it does not. Every one conflated *"I got no data"* with *"no data
exists"*.

That confusion was structural rather than careless. Nothing in the old
pipeline could represent "we never asked", so coverage had to be inferred from
absence — and absence has three causes that are indistinguishable from
outside:

    empty     the venue answered, and there is genuinely no book
    error     the request failed, so the question was never answered
    pending   nobody has asked yet

This table makes that a column. Everything else it provides — exact resume,
retry of only real failures, a live coverage map — falls out of having stated
it.

SQLite rather than Parquet because these rows are *updated*: the research
store is append-oriented and is the output of collection, not its bookkeeping.
`scripts/scrape.py` already uses SQLite the same way, for the same reason.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

# After this many failed attempts a window stops being offered. One genuinely
# broken window must not block a 47-hour queue forever, and a window that has
# failed five times with backoff between is not a transient problem.
MAX_ATTEMPTS = 5

PENDING, OK, EMPTY, ERROR, SKIPPED = 'pending', 'ok', 'empty', 'error', 'skipped'
TERMINAL = (OK, EMPTY, SKIPPED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS collection_ledger (
    venue           TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    window_open     TEXT    NOT NULL,   -- ISO-8601, UTC, the join key
    month           TEXT    NOT NULL,   -- YYYY-MM, denormalised for scheduling
    market_id       TEXT    NOT NULL,   -- the venue's own ticker/slug
    status          TEXT    NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    last_error      TEXT,
    snapshots       INTEGER NOT NULL DEFAULT 0,
    bytes           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (venue, symbol, window_open)
);
-- The scheduling query is "oldest month first, anything not finished", and it
-- runs once per batch for the length of the job.
CREATE INDEX IF NOT EXISTS ledger_work ON collection_ledger (status, month);
CREATE INDEX IF NOT EXISTS ledger_month ON collection_ledger (month);
"""


@dataclass(frozen=True)
class WorkItem:
    venue: str
    symbol: str
    window_open: dt.datetime
    market_id: str
    attempts: int = 0


def _iso(when: dt.datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(dt.timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._con = sqlite3.connect(path, timeout=30)
        self._con.row_factory = sqlite3.Row
        # The collector is one process by design (the Predexon bucket is
        # org-wide), so WAL buys crash-safety rather than concurrency: a
        # kill -9 mid-write leaves the last committed state intact.
        self._con.execute('PRAGMA journal_mode=WAL')
        self._con.executescript(SCHEMA)
        self._con.commit()

    # -- seeding ------------------------------------------------------------

    def seed(self, items: Iterable[tuple]) -> int:
        """Create a `pending` row per work item. Safe to re-run.

        `INSERT OR IGNORE` is the important part: Phase 0 may be re-run after
        Phase 2 has already collected half the corpus, and re-seeding must
        never reset a finished row back to pending.
        """
        rows = []
        for venue, symbol, window_open, market_id in items:
            iso = _iso(window_open)
            rows.append((venue, symbol, iso, iso[:7], str(market_id)))
        with self._con:
            self._con.executemany(
                'INSERT OR IGNORE INTO collection_ledger '
                '(venue, symbol, window_open, month, market_id) '
                'VALUES (?, ?, ?, ?, ?)', rows)
        return len(rows)

    # -- scheduling ---------------------------------------------------------

    def claim(self, limit: int, *, month: Optional[str] = None,
              venue: Optional[str] = None,
              since: Optional[dt.datetime] = None) -> list[WorkItem]:
        """The next work items, oldest month first.

        Deliberately does NOT mark anything in progress. A claim that is never
        recorded has to come back on the next call, because the alternative —
        an `in_progress` state — turns every hard kill into a set of orphaned
        rows needing a reaper. The cost of not having one is re-fetching at
        most a batch after a crash.

        Ordering is by month and then window, so the two venues interleave
        naturally within a month: collecting one venue to completion first
        would mean that stopping early yields no cross-venue windows at all,
        which is the pairing the whole exercise is for.
        """
        sql = ("SELECT * FROM collection_ledger WHERE status IN ('pending','error') "
               "AND attempts < ?")
        args: list = [MAX_ATTEMPTS]
        if month:
            sql += ' AND month = ?'
            args.append(month)
        if venue:
            sql += ' AND venue = ?'
            args.append(venue)
        if since is not None:
            # ISO-8601 in UTC sorts lexicographically, which is why the column
            # stores it that way rather than as an epoch.
            sql += ' AND window_open >= ?'
            args.append(_iso(since))
        sql += ' ORDER BY month, window_open, venue, symbol LIMIT ?'
        args.append(limit)
        return [WorkItem(venue=r['venue'], symbol=r['symbol'],
                         window_open=dt.datetime.fromisoformat(r['window_open']),
                         market_id=r['market_id'], attempts=r['attempts'])
                for r in self._con.execute(sql, args)]

    # -- recording ----------------------------------------------------------

    def record(self, item: WorkItem, status: str, *, snapshots: int = 0,
               bytes_: int = 0, error: Optional[str] = None) -> None:
        """What happened to one window. `attempts` always increments."""
        with self._con:
            self._con.execute(
                'UPDATE collection_ledger SET status = ?, attempts = attempts + 1, '
                'last_attempt_at = ?, last_error = ?, snapshots = ?, bytes = ? '
                'WHERE venue = ? AND symbol = ? AND window_open = ?',
                (status, dt.datetime.now(dt.timezone.utc).isoformat(),
                 (error or None) and str(error)[:300], snapshots, bytes_,
                 item.venue, item.symbol, _iso(item.window_open)))

    # -- reporting ----------------------------------------------------------

    def counts(self, **where) -> dict:
        sql = 'SELECT status, COUNT(*) n FROM collection_ledger'
        args: list = []
        if where:
            sql += ' WHERE ' + ' AND '.join(f'{k} = ?' for k in where)
            args = list(where.values())
        sql += ' GROUP BY status'
        return {r['status']: r['n'] for r in self._con.execute(sql, args)}

    def rows(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self._con.execute(
            'SELECT * FROM collection_ledger ORDER BY month, window_open LIMIT ?',
            (limit,))]

    def coverage(self) -> list[dict]:
        """Per month and venue: what we asked, and what came back.

        `yield_pct` is over ANSWERED windows (ok + empty), never over the whole
        month — otherwise a run that is half finished reports the same number
        as a run that half failed, which is the ambiguity this table exists to
        remove.
        """
        out = []
        for r in self._con.execute(
                'SELECT month, venue, '
                " SUM(status='ok') ok, SUM(status='empty') empty, "
                " SUM(status='error') error, SUM(status='pending') pending, "
                ' SUM(snapshots) snapshots, COUNT(*) total '
                'FROM collection_ledger GROUP BY month, venue ORDER BY month, venue'):
            row = dict(r)
            answered = (row['ok'] or 0) + (row['empty'] or 0)
            row['yield_pct'] = (100.0 * row['ok'] / answered) if answered else None
            out.append(row)
        return out

    def close(self) -> None:
        self._con.close()

    def __del__(self):                                    # best-effort
        try:
            self._con.close()
        except Exception:                                 # noqa: BLE001
            pass
