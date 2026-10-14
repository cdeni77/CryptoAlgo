"""One row per unit of collection work, and what happened to it.

The status column separates three outcomes that look the same from outside:

    empty     the venue answered, and there is no book
    error     the request failed, so the question was never answered
    pending   nobody has asked yet

This gives exact resume, retry of real failures only, and a coverage map.
SQLite, not Parquet, because these rows are updated in place.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

# After this many failed attempts a window stops being offered, so one broken
# window cannot block the queue.
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
        # One collector process at a time, so WAL is for crash safety, not
        # concurrency.
        self._con.execute('PRAGMA journal_mode=WAL')
        self._con.executescript(SCHEMA)
        self._con.commit()

    # -- seeding ------------------------------------------------------------

    def seed(self, items: Iterable[tuple]) -> int:
        """Create a `pending` row per work item. Safe to re-run.

        `INSERT OR IGNORE` never resets a finished row to pending.
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

        Does not mark items in progress, so a hard kill leaves no orphaned rows;
        an unrecorded claim returns on the next call. Ordering by month, then
        window, interleaves the two venues, so a partial run still has
        cross-venue pairs.
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
            # ISO-8601 in UTC sorts lexicographically.
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

        `yield_pct` is over answered windows (ok + empty), not the whole month.
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
