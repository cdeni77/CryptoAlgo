# Venue WebSocket Streaming — Implementation Plan (Phases 0–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maintain Kalshi order books from a WebSocket stream in one in-process
cache, archive the raw stream, and migrate `record_ladder` to sample that cache
instead of issuing a REST call per market per minute.

**Architecture:** A venue adapter parses WS frames into a normalized `BookEvent`.
A pure, synchronous `BookCache` folds events into per-market ladders and dates
every read. Consumers read the cache; nobody downstream of `BookEvent` knows
which venue or transport produced it. Raw frames go to an append-only spool that
never touches `ResearchStore.write`'s read-concat-rewrite path.

**Tech Stack:** Python 3.12, `aiohttp==3.13.3` (its `ClientSession.ws_connect`
is the WebSocket client — **no new dependency**), pandas 3.0, DuckDB 1.5.5,
pytest with `-n auto`.

**Spec:** `docs/superpowers/specs/2026-08-26-kalshi-websocket-streaming-design.md`

## Status — executed 2026-08-27

Tasks 1-9 are done and deployed. **Task 10 is deliberately not done:** it is
gated on 24 hours of paired-transport agreement, and that evidence does not
exist yet. Both samplers are running side by side; `record_ladder` still makes
its REST call.

**Where the implementation departed from the plan below, and why.** Each is a
Phase 0 measurement contradicting an assumption this plan was written on, which
is what Phase 0 was for. The task text further down is left as written rather
than retconned, so the difference stays visible.

| planned | actual | why |
|---|---|---|
| gap check per market | per **connection** | `seq` is contiguous per subscription (1..34,956) and not within a market (1, 9, 10). The planned check would flag every delta as a gap. |
| `msg.price` / `msg.delta` | `msg.price_dollars` / `msg.delta_fp` | The documented names are wrong. REST also spells the snapshot differently (`orderbook_fp.yes_dollars` vs `msg.yes_dollars_fp`). |
| `size > 0` keeps a level | `size > MIN_SIZE` | Signed deltas leave float residue (2.4e-12), which read as a phantom best bid 3c above the truth. |
| `venue_book_events` stores JSON text | typed columns, one row per level | A Kalshi delta is flat, so typed columns *are* the message, not a projection — and compress ~10x better, which is what makes any retention affordable at 862 frames/s. |
| `compact` via `store.write` | immutable hour-named Parquet | `write()` rewrites a whole (venue, symbol, month) partition; at this rate that partition reaches tens of GB. |
| retention "a knob with a comfortable default" | measured 20 B/row, 1.09 GB/day, default 14 days = 15 GB | The plan's arithmetic assumed ~5 frames/s. It is ~150x that. |
| `BookCache` special-cases `event.venue == 'kalshi'` | `BookEvent.absolute` flag | Caught in plan self-review before execution: branching on the venue breaks the boundary Task 4 exists to draw. |

**Evidence the fold is correct:** replaying 40,759 captured frames against 12
REST snapshots taken in the same window agrees exactly on the best bid on both
sides, 11 of 11 times; worst whole-ladder difference is one price out of ~100,
and that one is a maker toggling an order fourteen times a minute.

---

## Global Constraints

- **No new dependencies.** `requirements.txt` uses exact pins with a documented
  "move them one at a time, on purpose" policy. `aiohttp` 3.13.3 already
  provides `ws_connect`; verified in this environment.
- **All commands run from `backend/trader`.** Tests: `pytest`. Serial debugging:
  `pytest -n 0`.
- **Trading is halted** on the daily drawdown breaker; `scripts.live` keeps
  scoring and recording with no new entries. Probes against the real venue are
  therefore safe, and the live container `cryptoalgo-live-1` is the place to run
  them.
- **Never edit `scripts/live.py` in this plan.** The trading path is Phase 3, a
  separate plan. Phases 0–2 touch only recording and storage.
- **`_headers` is reused verbatim.** `KalshiClient._headers(method, path)` takes
  the already-prefixed path and signs `timestamp + METHOD + path`; `_request` is
  what adds `/trade-api/v2`. So the WS handshake is
  `client._headers('GET', '/trade-api/ws/v2')` with no change to signing code.
- **Endpoint:** `wss://api.elections.kalshi.com/trade-api/ws/v2`.
- **Series → symbol** comes from `core.config.series_to_symbol()`, never a
  hardcoded copy.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/probe_ws.py` | **Throwaway.** Phase 0 capture tool. Deleted in Task 10. |
| `tests/fixtures/ws/kalshi_orderbook_*.jsonl` | Captured frames + paired REST snapshots. Survives the probe. |
| `data_collection/stream/base.py` | `BookEvent`, `VenueStream` protocol. No venue specifics. |
| `data_collection/stream/kalshi.py` | Handshake, subscribe, frame → `BookEvent`. |
| `core/stream_book.py` | `BookCache`: fold events into ladders, date every read, detect gaps. Pure and synchronous. |
| `core/spool.py` | Append-only frame sink + compaction into `venue_book_events`. |
| `core/datastore.py` | Modified: `union_by_name`, `transport`/`book_age_ms` columns, `venue_book_events` schema, `EVENT_KEY_EXTRA['venue_ladder']`. |
| `scripts/record_ladder.py` | Modified: sample the cache instead of REST. |
| `scripts/run_live.py` | Modified: new supervised `stream` component. |

---

# Phase 0 — Measure before choosing a schema

Kalshi documents snapshot-then-delta and a `subscription buffer overflow` error
but publishes **no sequence-number or gap-detection contract**. Task 6's parser
cannot be written honestly against a guess, so it is written against what Task 2
observes.

### Task 1: The capture probe

**Files:**
- Create: `scripts/probe_ws.py`
- Create: `tests/fixtures/ws/` (directory, populated by the run)

**Interfaces:**
- Consumes: `KalshiClient` from `data_collection.kalshi_client`.
- Produces: a JSONL capture file. Each line is
  `{"t": <unix float>, "kind": "ws"|"rest", "payload": <verbatim object>}`.

**No test.** This file is explicitly throwaway and is deleted in Task 10. Its
output — the fixtures — is what survives and is what everything later is tested
against. Writing tests for code scheduled for deletion is ceremony.

- [ ] **Step 1: Write the probe**

```python
"""THROWAWAY. Phase 0 capture for the WebSocket design. Delete after Task 10.

Records every WS frame verbatim alongside periodic REST orderbook snapshots of
the same tickers, so the two can be compared at the same instant. That
comparison is the only evidence that folding deltas reproduces the real book.
"""
from __future__ import annotations

import argparse, asyncio, json, os, time
from datetime import datetime, timezone
from pathlib import Path

from core.config import series_to_symbol
from data_collection.kalshi_client import KalshiClient

WS_URL = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
WS_PATH = '/trade-api/ws/v2'


async def open_tickers(client: KalshiClient) -> list[str]:
    out = []
    for series in series_to_symbol():
        payload = await client._request(  # noqa: SLF001
            'GET', '/markets',
            params={'series_ticker': series, 'status': 'open', 'limit': 5})
        out += [m['ticker'] for m in payload.get('markets', []) if m.get('ticker')]
    return out


async def rest_sampler(client, tickers, sink, every: float):
    while True:
        for ticker in tickers:
            try:
                book = await client._request(  # noqa: SLF001
                    'GET', f'/markets/{ticker}/orderbook')
            except Exception as exc:  # noqa: BLE001
                book = {'error': str(exc)[:200], 'ticker': ticker}
            book.setdefault('ticker', ticker)
            sink({'t': time.time(), 'kind': 'rest', 'payload': book})
        await asyncio.sleep(every)


async def run(args) -> int:
    pem = os.getenv('KALSHI_PRIVATE_KEY') or open(
        os.environ['KALSHI_PRIVATE_KEY_PATH']).read()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open('w')

    def sink(record):
        handle.write(json.dumps(record) + '\n')
        handle.flush()

    async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                            private_key_pem=pem) as client:
        tickers = await open_tickers(client)
        print(f'subscribing to {len(tickers)}: {tickers}')
        headers = client._headers('GET', WS_PATH)  # noqa: SLF001
        headers.pop('Content-Type', None)
        sampler = asyncio.create_task(
            rest_sampler(client, tickers, sink, args.rest_every))
        try:
            async with client._session.ws_connect(  # noqa: SLF001
                    WS_URL, headers=headers, heartbeat=10) as ws:
                await ws.send_json({'id': 1, 'cmd': 'subscribe', 'params': {
                    'channels': ['orderbook_delta'], 'market_tickers': tickers}})
                deadline = time.time() + args.seconds
                async for msg in ws:
                    sink({'t': time.time(), 'kind': 'ws',
                          'payload': json.loads(msg.data)})
                    if time.time() > deadline:
                        break
        finally:
            sampler.cancel()
            handle.close()
    print(f'wrote {out}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seconds', type=float, default=1200.0)
    p.add_argument('--rest-every', type=float, default=30.0)
    p.add_argument('--out', default='tests/fixtures/ws/kalshi_capture.jsonl')
    return p


if __name__ == '__main__':
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))
```

- [ ] **Step 2: Run it in the live container for one full window**

```bash
docker compose exec live python -m scripts.probe_ws --seconds 1200
```

Expected: prints the subscribed tickers, then exits after 20 minutes having
written the capture. If the handshake 401s, the likely cause is a header the
server rejects — print `headers` and compare against a working REST call.

- [ ] **Step 3: Copy the capture out and commit it**

```bash
docker compose cp live:/app/tests/fixtures/ws/kalshi_capture.jsonl \
  backend/trader/tests/fixtures/ws/kalshi_capture.jsonl
git add backend/trader/scripts/probe_ws.py backend/trader/tests/fixtures/ws/
git commit -m "Capture a window of Kalshi WS frames, so the schema stops being a guess"
```

### Task 2: Answer the four open questions from the capture

**Files:**
- Create: `scripts/analyse_ws_capture.py` (throwaway, deleted in Task 10)
- Modify: `docs/superpowers/specs/2026-08-26-kalshi-websocket-streaming-design.md`

**Interfaces:**
- Consumes: the JSONL from Task 1.
- Produces: **the field names Task 6 parses and Task 5 folds.** Every later task
  depends on this output being written into the spec.

- [ ] **Step 1: Write the analyser**

```python
"""THROWAWAY. Answers the four Phase 0 questions from a probe capture."""
from __future__ import annotations

import json, sys
from collections import Counter


def main(path: str) -> int:
    ws, rest = [], []
    for line in open(path):
        rec = json.loads(line)
        (ws if rec['kind'] == 'ws' else rest).append(rec)

    span = (ws[-1]['t'] - ws[0]['t']) if len(ws) > 1 else 0.0
    print(f'frames={len(ws)} span={span:.0f}s rate={len(ws)/max(span,1):.2f}/s')

    types = Counter(m['payload'].get('type') for m in ws)
    print('message types:', dict(types))

    for kind in types:
        sample = next(m for m in ws if m['payload'].get('type') == kind)
        print(f'\n--- {kind} keys:', sorted(sample["payload"].keys()))
        print('    msg keys:', sorted((sample['payload'].get('msg') or {}).keys()))
        print('    sample:', json.dumps(sample['payload'])[:400])

    seqs = [m['payload'].get('seq') for m in ws if 'seq' in m['payload']]
    print(f'\nseq present on {len(seqs)}/{len(ws)} frames; '
          f'monotonic={seqs == sorted(seqs) if seqs else "n/a"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
```

- [ ] **Step 2: Run it**

```bash
cd backend/trader && python -m scripts.analyse_ws_capture tests/fixtures/ws/kalshi_capture.jsonl
```

- [ ] **Step 3: Write the findings into the spec**

Add a `## Phase 0 findings (measured YYYY-MM-DD)` section recording, in this
order:

1. **Message rate** per market per second, and the peak in the final minute of a
   window. This sets the spool retention default in Task 7.
2. **Whether `seq` exists and is monotonic per market.** If yes, Task 5's gap
   detection is sequence-based. If no, it is REST-reconciliation-based. Both
   branches are written in Task 5; this decides which is armed.
3. **The exact field names** of the snapshot and delta payloads. Task 6's
   `_parse` is written against the documented guess (`market_ticker`, `price`,
   `delta`, `side`, `yes`/`no`); **correct it here if the capture disagrees.**
4. **Anything that arrived that we did not subscribe to** — heartbeats, `ok`
   acknowledgements, errors. Task 6 must not crash on them.

- [ ] **Step 4: Commit**

```bash
git add backend/trader/scripts/analyse_ws_capture.py \
  docs/superpowers/specs/2026-08-26-kalshi-websocket-streaming-design.md
git commit -m "Record what the Kalshi stream actually sends"
```

---

# Phase 1 — The plumbing, with the archive still on REST

### Task 3: Make the store survive an added column, and stop it collapsing two transports

**Files:**
- Modify: `core/datastore.py:112-120` (venue_ladder schema), `:126-132`
  (pm_ladder), `:228-230` (`EVENT_KEY_EXTRA`), `:488` (`read_parquet`)
- Test: `tests/test_store_transport_column.py`

**Interfaces:**
- Produces: `venue_ladder` and `pm_ladder` gain `transport` (str) and
  `book_age_ms` (float). `event_key('venue_ladder')` becomes
  `('venue', 'symbol', 'event_time', 'transport')`.

**Why both changes are required together.** `read()` keeps one row per
`event_key` — the latest `available_time`. During Task 9's parallel run the REST
row and the WS row describe the *same minute of the same book by two
independent observers*, exactly like `venue_depth`'s live-vs-backfill pair. The
comment at `core/datastore.py:212` records what happens without the extra key:
58 overlapping pairs, and the comparison saw **zero rows, silently**. And
verified in this environment, DuckDB's `read_parquet` over a glob whose files
have different columns fails with `schema mismatch in glob` unless
`union_by_name=true` — so the 3.2 MB of existing `venue_ladder` partitions would
become unreadable the moment a new column is written.

- [ ] **Step 1: Write the failing tests**

```python
"""Adding a column must not orphan the archive, and two transports are two observations."""
from __future__ import annotations

import pandas as pd
import pytest

from core.datastore import ResearchStore, event_key


def _row(**over):
    base = dict(venue='kalshi', symbol='BTC-USD',
                event_time=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                available_time=pd.Timestamp('2026-08-26 12:00:05', tz='UTC'),
                quality='valid', market_ticker='KXBTC15M-26AUG2612',
                window_open=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                minute_into_window=0.0, yes_levels='[]', no_levels='[]',
                yes_total=0.0, no_total=0.0)
    base.update(over)
    return base


def test_transport_is_part_of_the_event_key(tmp_path):
    assert 'transport' in event_key('venue_ladder')


def test_two_transports_for_one_minute_both_survive_a_read(tmp_path):
    store = ResearchStore(tmp_path)
    store.write('venue_ladder', pd.DataFrame([
        _row(transport='rest', book_age_ms=0.0),
        _row(transport='ws', book_age_ms=120.0,
             available_time=pd.Timestamp('2026-08-26 12:00:06', tz='UTC')),
    ]))
    got = store.read('venue_ladder')
    assert sorted(got['transport']) == ['rest', 'ws'], (
        'the WS row and the REST row are independent observations, not a '
        'revision and its predecessor')


def test_a_partition_written_before_the_column_existed_still_reads(tmp_path):
    """The 3.2MB already on disk has no `transport` column."""
    store = ResearchStore(tmp_path)
    old = pd.DataFrame([_row()])
    prepared = store._prepare('venue_ladder', old)  # noqa: SLF001
    legacy = prepared.drop(columns=['transport', 'book_age_ms'])
    part = tmp_path / 'venue_ladder' / 'kalshi' / 'BTC-USD' / '2026-08'
    part.mkdir(parents=True)
    legacy.to_parquet(part / 'legacy.parquet', index=False)

    store.write('venue_ladder', pd.DataFrame([
        _row(transport='ws', book_age_ms=5.0, minute_into_window=1.0,
             event_time=pd.Timestamp('2026-08-26 12:01', tz='UTC'))]))
    got = store.read('venue_ladder')
    assert len(got) == 2
    assert got['transport'].isna().sum() == 1
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend/trader && pytest tests/test_store_transport_column.py -n 0 -v`
Expected: FAIL — first on `'transport' in event_key(...)`, then on the schema
mismatch from `read_parquet`.

- [ ] **Step 3: Make the three edits**

In `SCHEMAS['venue_ladder']` and `SCHEMAS['pm_ladder']`, append after
`'no_total'`:

```python
        # How this ladder reached us, and how stale the cache was when sampled.
        # `transport` is part of the event key rather than a plain column:
        # a REST-sampled and a WS-sampled row for the same minute are two
        # independent observations of one book, not a correction and the thing
        # it corrects. Level counts are already known to be incomparable across
        # sources at a measured ratio of 0.579, so an unlabelled row would make
        # a transport change look like a market change.
        'transport', 'book_age_ms',
```

In `EVENT_KEY_EXTRA`:

```python
EVENT_KEY_EXTRA: dict[str, tuple[str, ...]] = {
    'venue_depth': ('source',),
    # Same argument as venue_depth, one layer down: during the WS migration the
    # REST sampler and the WS sampler both write the same minute, and the
    # comparison between them is the only evidence the stream reproduces the
    # book. Without this the later `available_time` silently wins and the
    # comparison reads zero rows.
    'venue_ladder': ('transport',),
    'pm_ladder': ('transport',),
}
```

In `read()`, at the `read_parquet` call:

```python
        # `union_by_name` because partitions written before a column existed do
        # not have it. Verified: without this, a glob spanning an old and a new
        # partition raises `schema mismatch in glob` and the whole archive
        # becomes unreadable the moment a column is added.
        f"    FROM read_parquet(?, hive_partitioning = false, union_by_name = true) {where}"
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `cd backend/trader && pytest tests/test_store_transport_column.py -n 0 -v`
Expected: PASS.

Run: `cd backend/trader && pytest -m "not slow"`
Expected: PASS. `event_key('venue_ladder')` changed, so anything asserting on
ladder de-duplication is the place a regression would show — read the failure
rather than adjusting the test to match.

- [ ] **Step 5: Commit**

```bash
git add backend/trader/core/datastore.py backend/trader/tests/test_store_transport_column.py
git commit -m "A transport is an observer, not a revision"
```

### Task 4: The normalized event

**Files:**
- Create: `data_collection/stream/__init__.py`, `data_collection/stream/base.py`
- Test: `tests/test_stream_base.py`

**Interfaces:**
- Produces:
  - `BookEvent(venue: str, market_ticker: str, kind: str, received: float,
    seq: int | None, yes: list[tuple[float, float]], no: list[tuple[float, float]],
    absolute: bool = True)` where `kind` is `'snapshot'` or `'delta'`.
    `absolute=True` means the size IS the resting size at that price and `0.0`
    removes the level; `absolute=False` means the size is a signed CHANGE to
    apply. A snapshot is always absolute.
  - `VenueStream` protocol: `async connect()`, `async subscribe(tickers)`,
    `events()` async iterator of `BookEvent`, `async close()`.

**`absolute` is the load-bearing decision here.** Kalshi sends a signed change;
Polymarket sends a resulting size. The adapter cannot normalize a signed change
to an absolute size, because that needs the current size and only the cache
holds it. So the convention travels ON THE EVENT: the cache reads *what the
sizes mean*, never *who sent them*. Branching on `event.venue` inside the cache
would work today and quietly break the boundary this file exists to draw —
`BookCache` must stay venue-blind, or every future venue edits it.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest

from data_collection.stream.base import BookEvent


def test_a_zero_size_level_is_a_removal_not_a_price():
    event = BookEvent(venue='kalshi', market_ticker='K', kind='delta',
                      received=1.0, seq=7, yes=[(0.31, 0.0)], no=[])
    assert event.yes == [(0.31, 0.0)]
    assert event.is_delta and not event.is_snapshot


def test_a_snapshot_is_always_absolute_whatever_the_caller_passed():
    event = BookEvent(venue='kalshi', market_ticker='K', kind='snapshot',
                      received=1.0, seq=1, yes=[], no=[], absolute=False)
    assert event.absolute, 'a snapshot IS the book; a signed snapshot is meaningless'


def test_kind_must_be_snapshot_or_delta():
    with pytest.raises(ValueError):
        BookEvent(venue='kalshi', market_ticker='K', kind='update',
                  received=1.0, seq=None, yes=[], no=[])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend/trader && pytest tests/test_stream_base.py -n 0 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.stream'`

- [ ] **Step 3: Implement**

`data_collection/stream/__init__.py` is empty. `data_collection/stream/base.py`:

```python
"""What every venue's stream reduces to, so nothing downstream knows the venue.

**What a size MEANS travels on the event, so the cache never learns the venue.**
Kalshi sends a signed change; Polymarket sends a resulting size. An adapter
cannot convert the first into the second — that needs the current resting size,
and only the cache holds it. So `absolute` says which convention this event
uses, and `BookCache` branches on that rather than on `venue`. Branching on the
venue would work today and quietly make every future venue an edit to the cache.

An absolute size of 0.0 is a REMOVAL, not a price of zero; dropping it would
leave a stale level resting forever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

Level = tuple[float, float]
KINDS = ('snapshot', 'delta')


@dataclass(frozen=True)
class BookEvent:
    venue: str
    market_ticker: str
    kind: str
    received: float          # time.time() at receipt, for staleness
    seq: int | None          # None where the venue publishes no sequence
    yes: list[Level]         # YES bids
    no: list[Level]          # NO bids — NO-denominated on BOTH venues
    # True: the size IS the resting size at that price, and 0.0 removes the
    # level. False: the size is a signed CHANGE to apply. Kalshi sends changes,
    # Polymarket sends resulting sizes, and the adapter cannot convert one to
    # the other because that needs the current size — which only the cache
    # holds. So the convention travels on the event and the cache stays
    # venue-blind.
    absolute: bool = True

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {self.kind!r}')
        if self.kind == 'snapshot' and not self.absolute:
            object.__setattr__(self, 'absolute', True)

    @property
    def is_snapshot(self) -> bool:
        return self.kind == 'snapshot'

    @property
    def is_delta(self) -> bool:
        return self.kind == 'delta'


class VenueStream(Protocol):
    async def connect(self) -> None: ...
    async def subscribe(self, tickers: Sequence[str]) -> None: ...
    def events(self) -> AsyncIterator[BookEvent]: ...
    async def close(self) -> None: ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend/trader && pytest tests/test_stream_base.py -n 0 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/trader/data_collection/stream/ backend/trader/tests/test_stream_base.py
git commit -m "One event shape, so the cache never learns which venue it serves"
```

### Task 5: BookCache — fold, date, and detect a gap

**Files:**
- Create: `core/stream_book.py`
- Test: `tests/test_stream_book.py`

**Interfaces:**
- Consumes: `BookEvent` from Task 4.
- Produces:
  - `BookCache(max_age_seconds: float = 10.0, now: Callable[[], float] = time.time)`
  - `.apply(event: BookEvent) -> None`
  - `.ladder(ticker: str) -> Ladder | None` where
    `Ladder(yes: list[Level], no: list[Level], age_seconds: float, stale: bool)`
  - `.gapped(ticker: str) -> bool` — True once a sequence gap is seen, until the
    next snapshot clears it.

**The default is 10 seconds and it is derived.** The tape measurement in
`project-latency-budget` puts the market's information gain below the noise
floor out to ~30s, so anything under that costs nothing. A book older than a few
seconds means the transport is sick, not that the market is quiet — so 10s sits
inside the free region with room, and a breach reads as a fault.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import pytest

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def snap(**over):
    base = dict(venue='kalshi', market_ticker='K', kind='snapshot',
                received=1000.0, seq=1, yes=[(0.30, 100.0), (0.31, 50.0)],
                no=[(0.68, 20.0)])
    base.update(over)
    return BookEvent(**base)


def test_a_snapshot_replaces_the_book_rather_than_merging():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(seq=2, yes=[(0.40, 7.0)], no=[]))
    assert cache.ladder('K').yes == [(0.40, 7.0)]


def test_an_absolute_delta_sets_the_size_at_a_price():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(kind='delta', seq=2, yes=[(0.30, 5.0)], no=[]))
    assert dict(cache.ladder('K').yes)[0.30] == 5.0
    assert dict(cache.ladder('K').yes)[0.31] == 50.0, 'untouched levels survive'


def test_a_signed_delta_adds_to_the_resting_size():
    cache = BookCache(now=Clock())
    cache.apply(snap())                                    # 0.30 -> 100
    cache.apply(snap(kind='delta', seq=2, absolute=False,
                     yes=[(0.30, -40.0)], no=[]))
    assert dict(cache.ladder('K').yes)[0.30] == 60.0


def test_a_signed_delta_that_empties_a_level_removes_it():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(kind='delta', seq=2, absolute=False,
                     yes=[(0.30, -100.0)], no=[]))
    assert 0.30 not in dict(cache.ladder('K').yes)


def test_the_cache_never_branches_on_the_venue():
    import inspect
    import core.stream_book as mod
    assert 'kalshi' not in inspect.getsource(mod).lower(), (
        'the cache must stay venue-blind; the convention rides on the event')


def test_a_zero_size_delta_removes_the_level():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(kind='delta', seq=2, yes=[(0.30, 0.0)], no=[]))
    assert 0.30 not in dict(cache.ladder('K').yes)


def test_an_unknown_ticker_is_none_not_an_empty_book():
    assert BookCache(now=Clock()).ladder('NOPE') is None, (
        'an empty ladder and no ladder are different claims')


def test_age_is_measured_from_the_last_event_and_marks_stale():
    clock = Clock()
    cache = BookCache(max_age_seconds=10.0, now=clock)
    cache.apply(snap(received=clock.t))
    clock.t += 4.0
    assert cache.ladder('K').age_seconds == pytest.approx(4.0)
    assert not cache.ladder('K').stale
    clock.t += 7.0
    assert cache.ladder('K').stale


def test_a_sequence_gap_is_flagged_and_a_snapshot_clears_it():
    cache = BookCache(now=Clock())
    cache.apply(snap(seq=1))
    cache.apply(snap(kind='delta', seq=3, yes=[(0.30, 1.0)], no=[]))
    assert cache.gapped('K'), 'seq 2 never arrived; the book may be wrong'
    cache.apply(snap(seq=4))
    assert not cache.gapped('K')


def test_no_seq_means_no_gap_detection_rather_than_a_false_gap():
    cache = BookCache(now=Clock())
    cache.apply(snap(seq=None))
    cache.apply(snap(kind='delta', seq=None, yes=[(0.30, 1.0)], no=[]))
    assert not cache.gapped('K')


def test_a_delta_for_a_book_we_never_snapshotted_is_refused():
    cache = BookCache(now=Clock())
    cache.apply(snap(kind='delta', seq=5, yes=[(0.30, 1.0)], no=[]))
    assert cache.ladder('K') is None, (
        'folding a delta into nothing invents a book from one level')
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend/trader && pytest tests/test_stream_book.py -n 0 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.stream_book'`

- [ ] **Step 3: Implement**

```python
"""The live book, folded from a stream, and dated on every read.

**A cached book that quietly stops updating is worse than a REST call**, because
it looks healthy. So there is no way to read a ladder here without also reading
its age: `Ladder` carries `age_seconds` and `stale`, and an unknown ticker
returns None rather than an empty book — an empty ladder and no ladder are
different claims, and conflating them would let a dead subscription read as a
market with nothing resting in it.

Pure and synchronous on purpose. This holds the state a wrong answer would come
from, so it is the part that must be exhaustively testable without a network.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from data_collection.stream.base import BookEvent, Level

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ladder:
    yes: list[Level]
    no: list[Level]
    age_seconds: float
    stale: bool


@dataclass
class _Book:
    yes: dict[float, float] = field(default_factory=dict)
    no: dict[float, float] = field(default_factory=dict)
    last_received: float = 0.0
    last_seq: Optional[int] = None
    gapped: bool = False


class BookCache:
    def __init__(self, max_age_seconds: float = 10.0,
                 now: Callable[[], float] = time.time) -> None:
        self.max_age_seconds = max_age_seconds
        self._now = now
        self._books: dict[str, _Book] = {}

    def apply(self, event: BookEvent) -> None:
        book = self._books.get(event.market_ticker)
        if book is None:
            if event.is_delta:
                # Folding a delta into nothing would invent a whole book out of
                # the one level that happened to change first.
                logger.debug('%s: delta before snapshot, dropped',
                             event.market_ticker)
                return
            book = self._books[event.market_ticker] = _Book()

        if event.is_snapshot:
            book.yes = {p: s for p, s in event.yes if s > 0}
            book.no = {p: s for p, s in event.no if s > 0}
            book.gapped = False
        else:
            if (event.seq is not None and book.last_seq is not None
                    and event.seq != book.last_seq + 1):
                # Not repaired here: the cache cannot request a snapshot. It
                # raises the flag and the stream owner resubscribes.
                logger.warning('%s: seq gap %s -> %s', event.market_ticker,
                               book.last_seq, event.seq)
                book.gapped = True
            for side, levels in (('yes', event.yes), ('no', event.no)):
                target = getattr(book, side)
                for price, size in levels:
                    # `absolute` rides on the event precisely so this line does
                    # not have to know which venue sent it.
                    if not event.absolute:
                        size = target.get(price, 0.0) + size
                    if size > 0:
                        target[price] = size
                    else:
                        target.pop(price, None)

        book.last_received = event.received
        if event.seq is not None:
            book.last_seq = event.seq

    def ladder(self, ticker: str) -> Optional[Ladder]:
        book = self._books.get(ticker)
        if book is None:
            return None
        age = self._now() - book.last_received
        return Ladder(yes=sorted(book.yes.items()), no=sorted(book.no.items()),
                      age_seconds=age, stale=age > self.max_age_seconds)

    def gapped(self, ticker: str) -> bool:
        book = self._books.get(ticker)
        return bool(book and book.gapped)

    def forget(self, ticker: str) -> None:
        """Drop a settled market, so a closed book cannot be sampled as live."""
        self._books.pop(ticker, None)
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd backend/trader && pytest tests/test_stream_book.py -n 0 -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/trader/core/stream_book.py backend/trader/tests/test_stream_book.py
git commit -m "A book you cannot date is a book you cannot trade"
```

### Task 6: The Kalshi adapter, tested against the real capture

**Files:**
- Create: `data_collection/stream/kalshi.py`
- Test: `tests/test_stream_kalshi.py`

**Interfaces:**
- Consumes: `BookEvent` (Task 4), `KalshiClient._headers`, the Task 1 fixture.
- Produces: `KalshiStream(client)` implementing `VenueStream`; and
  `parse_frame(payload: dict, received: float) -> BookEvent | None` — a module
  function so it is testable with no socket. `None` means "not a book message"
  (heartbeat, subscribe ack, error) and must never raise.

**Before writing `parse_frame`, open the Phase 0 findings section of the spec
and use the field names recorded there.** The code below uses the names the
public documentation implies; Task 2 exists because those may be wrong. If they
differ, the fixture test in Step 1 fails and the fixture is right.

- [ ] **Step 1: Write the failing tests**

```python
"""The adapter is tested against real captured frames, not invented ones."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_collection.stream.kalshi import parse_frame

FIXTURE = Path(__file__).parent / 'fixtures' / 'ws' / 'kalshi_capture.jsonl'
pytestmark = pytest.mark.skipif(not FIXTURE.exists(),
                                reason='run scripts.probe_ws first (Task 1)')


def frames(kind='ws'):
    for line in FIXTURE.open():
        rec = json.loads(line)
        if rec['kind'] == kind:
            yield rec


def test_every_captured_frame_parses_or_is_ignored_but_never_raises():
    for rec in frames():
        parse_frame(rec['payload'], rec['t'])  # must not raise


def test_the_capture_contains_at_least_one_snapshot_and_one_delta():
    kinds = {e.kind for rec in frames()
             if (e := parse_frame(rec['payload'], rec['t'])) is not None}
    assert 'snapshot' in kinds and 'delta' in kinds


def test_a_non_book_frame_is_ignored_rather_than_parsed():
    assert parse_frame({'type': 'subscribed', 'id': 1}, 1.0) is None
    assert parse_frame({}, 1.0) is None


def test_the_folded_book_matches_a_rest_snapshot_at_the_same_instant():
    """The only evidence that folding deltas reproduces the real book."""
    from core.stream_book import BookCache
    from data_collection.stream.kalshi import rest_levels

    cache = BookCache(now=lambda: 0.0)
    checked = 0
    rest_by_time = [(r['t'], r['payload']) for r in frames('rest')]
    ws_records = list(frames('ws'))

    for t_rest, payload in rest_by_time:
        ticker = payload.get('ticker')
        if not ticker or 'error' in payload:
            continue
        for rec in ws_records:
            if rec['t'] > t_rest:
                break
            event = parse_frame(rec['payload'], rec['t'])
            if event is not None:
                cache.apply(event)
        ladder = cache.ladder(ticker)
        if ladder is None or cache.gapped(ticker):
            continue
        want_yes, want_no = rest_levels(payload)
        assert ladder.yes == want_yes, f'{ticker} YES diverged at {t_rest}'
        assert ladder.no == want_no, f'{ticker} NO diverged at {t_rest}'
        checked += 1
    assert checked >= 3, f'only {checked} comparisons; capture a longer window'
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend/trader && pytest tests/test_stream_kalshi.py -n 0 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.stream.kalshi'`

- [ ] **Step 3: Implement**

```python
"""Kalshi's `orderbook_delta`, normalized.

The handshake reuses `KalshiClient._headers` verbatim: that method takes the
already-prefixed path and signs `timestamp + METHOD + path`, while `_request` is
what adds `/trade-api/v2`. So the correct WS signature is
`_headers('GET', '/trade-api/ws/v2')` with no change to the signing code.

`Content-Type: application/json` is dropped from the handshake headers — it
describes a body a WebSocket upgrade does not have.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional, Sequence

from data_collection.stream.base import BookEvent, Level

logger = logging.getLogger('kalshi-stream')

WS_URL = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
WS_PATH = '/trade-api/ws/v2'
SNAPSHOT_TYPES = {'orderbook_snapshot'}
DELTA_TYPES = {'orderbook_delta'}


def _levels(raw) -> list[Level]:
    out: list[Level] = []
    for entry in raw or []:
        try:
            out.append((float(entry[0]), float(entry[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(out)


def rest_levels(book: dict) -> tuple[list[Level], list[Level]]:
    """The same ladder from `GET /markets/{t}/orderbook`, for cross-checking."""
    ladder = book.get('orderbook_fp') or book.get('orderbook') or {}
    yes = _levels(ladder.get('yes_dollars') or ladder.get('yes'))
    no = _levels(ladder.get('no_dollars') or ladder.get('no'))
    return ([lv for lv in yes if lv[1] > 0], [lv for lv in no if lv[1] > 0])


def parse_frame(payload: dict, received: float) -> Optional[BookEvent]:
    """A book message as a BookEvent, or None for anything else.

    Never raises. A heartbeat, a subscribe acknowledgement and an error frame
    all arrive on this socket, and a parser that crashed on them would take the
    whole stream down for a message that carries no book.
    """
    kind_raw = (payload or {}).get('type')
    if kind_raw in SNAPSHOT_TYPES:
        kind = 'snapshot'
    elif kind_raw in DELTA_TYPES:
        kind = 'delta'
    else:
        return None

    msg = payload.get('msg') or {}
    ticker = msg.get('market_ticker') or msg.get('ticker')
    if not ticker:
        return None

    seq = payload.get('seq')
    absolute = True
    if kind == 'snapshot':
        yes, no = _levels(msg.get('yes')), _levels(msg.get('no'))
    else:
        # A delta names one price on one side. `delta` is a SIGNED CHANGE on the
        # wire; `BookEvent` carries absolute size, so this cannot be normalized
        # without the current size — which the cache holds, not us. So a delta
        # is emitted as the signed change under a reserved marker and the cache
        # is not asked to guess: see _apply_signed below.
        side = (msg.get('side') or '').lower()
        try:
            price, change = float(msg['price']), float(msg['delta'])
        except (KeyError, TypeError, ValueError):
            return None
        levels = [(price, change)]
        yes, no = (levels, []) if side == 'yes' else ([], levels)
        absolute = False
    try:
        seq = int(seq) if seq is not None else None
    except (TypeError, ValueError):
        seq = None
    return BookEvent(venue='kalshi', market_ticker=str(ticker), kind=kind,
                     received=received, seq=seq, yes=yes, no=no,
                     absolute=absolute)


class KalshiStream:
    """Connect, subscribe, and yield normalized events."""

    def __init__(self, client, url: str = WS_URL) -> None:
        self._client = client
        self._url = url
        self._ws = None
        self._next_id = 1

    async def connect(self) -> None:
        headers = self._client._headers('GET', WS_PATH)  # noqa: SLF001
        headers.pop('Content-Type', None)
        self._ws = await self._client._session.ws_connect(  # noqa: SLF001
            self._url, headers=headers, heartbeat=10)

    async def subscribe(self, tickers: Sequence[str]) -> None:
        if not tickers:
            return
        await self._ws.send_json({
            'id': self._next_id, 'cmd': 'subscribe',
            'params': {'channels': ['orderbook_delta'],
                       'market_tickers': list(tickers)}})
        self._next_id += 1

    async def events(self) -> AsyncIterator[BookEvent]:
        import time
        async for msg in self._ws:
            try:
                payload = json.loads(msg.data)
            except (ValueError, TypeError):
                continue
            event = parse_frame(payload, time.time())
            if event is not None:
                yield event

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
```

**On the signed delta.** Kalshi's `delta` field is a change, not a resulting
size, so `parse_frame` emits the delta event with `absolute=False` and lets the
cache add it to what is resting. This is why `absolute` exists on `BookEvent`
rather than being special-cased in the cache.

- [ ] **Step 4: Run to verify they pass**

Run: `cd backend/trader && pytest tests/test_stream_kalshi.py tests/test_stream_book.py -n 0 -v`
Expected: PASS. If `test_the_folded_book_matches_a_rest_snapshot_at_the_same_instant`
fails, **the fixture is right and the parser is wrong** — the field names or the
delta convention differ from what the docs implied. Correct `parse_frame` against
the capture and record the correction in the spec's Phase 0 findings.

- [ ] **Step 5: Commit**

```bash
git add backend/trader/data_collection/stream/kalshi.py \
  backend/trader/core/stream_book.py backend/trader/tests/test_stream_kalshi.py \
  backend/trader/tests/test_stream_book.py
git commit -m "Fold Kalshi's deltas, and prove it against the book itself"
```

### Task 7: The append-only spool and `venue_book_events`

**Files:**
- Create: `core/spool.py`
- Modify: `core/datastore.py` (add `venue_book_events` to `SCHEMAS`)
- Test: `tests/test_spool.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (raw payloads only).
- Produces:
  - `FrameSpool(root: Path, venue: str)` with `.append(record: dict) -> None`,
    `.flush() -> None`, `.closed_files() -> list[Path]`
  - `compact(spool_root: Path, store: ResearchStore, *, keep_days: float) -> int`
    returning rows written.

**Why not `ResearchStore.write`.** That path reads a partition, concatenates,
sorts and rewrites it under zstd — `run_live.py` names it as *the* latency
threat, and it is O(partition) per flush against a stream appending thousands of
rows a minute. The spool is O(appended) and never reads back.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.spool import FrameSpool, compact
from core.datastore import ResearchStore


def rec(t=1756200000.0, ticker='K'):
    return {'t': t, 'venue': 'kalshi', 'symbol': 'BTC-USD',
            'market_ticker': ticker, 'seq': 1, 'kind': 'snapshot',
            'payload': {'type': 'orderbook_snapshot'}}


def test_append_never_reads_the_file_back(tmp_path):
    spool = FrameSpool(tmp_path, 'kalshi')
    for i in range(500):
        spool.append(rec(t=1756200000.0 + i))
    spool.flush()
    written = list(tmp_path.rglob('*.jsonl'))
    assert len(written) == 1
    assert sum(1 for _ in written[0].open()) == 500


def test_frames_roll_into_separate_hourly_files(tmp_path):
    spool = FrameSpool(tmp_path, 'kalshi')
    spool.append(rec(t=1756200000.0))
    spool.append(rec(t=1756200000.0 + 3600))
    spool.flush()
    assert len(list(tmp_path.rglob('*.jsonl'))) == 2


def test_compaction_moves_closed_files_into_the_store_and_removes_them(tmp_path):
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append(rec(t=1756200000.0))
    spool.flush()
    spool.close()

    store = ResearchStore(store_root)
    assert compact(spool_root, store, keep_days=0.0) == 1
    got = store.read('venue_book_events', min_quality=None)
    assert len(got) == 1
    assert json.loads(got.iloc[0]['payload'])['type'] == 'orderbook_snapshot'
    assert not list(spool_root.rglob('*.jsonl')), 'compacted files are removed'


def test_the_open_file_is_never_compacted(tmp_path):
    """Compacting a file still being appended to would truncate the stream."""
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append(rec(t=1756200000.0))
    spool.flush()
    assert compact(spool_root, ResearchStore(store_root), keep_days=0.0) == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend/trader && pytest tests/test_spool.py -n 0 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.spool'`

- [ ] **Step 3: Add the schema, then implement**

In `core/datastore.py` `SCHEMAS`, after `pm_ladder`:

```python
    # **The raw stream, stored verbatim.** One row per WebSocket frame, with the
    # message as JSON text rather than a parsed table of level changes.
    #
    # The reason is ONE APPLIER. The function that folds a message into a ladder
    # is the same function the live cache runs and the same one a replay runs.
    # Store a parsed projection and there are two interpretations of one stream,
    # free to drift; store the message and the archive cannot disagree with the
    # live book about what the stream meant. Same argument as one `decide()`,
    # and the same argument `record_ladder` makes for storing levels rather than
    # the 1c/5c buckets: a projection chosen for one question forecloses the
    # rest.
    'venue_book_events': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'market_ticker', 'seq', 'kind', 'payload',
    ),
```

And in `EVENT_KEY_EXTRA`, because many frames share one millisecond:

```python
    # Many frames can carry the same `event_time`. Without `seq` and the ticker
    # in the key, `read` would keep exactly one of them and silently discard the
    # rest of the stream.
    'venue_book_events': ('market_ticker', 'seq'),
```

`core/spool.py`:

```python
"""An append-only sink for the raw frame stream, and its compaction.

**This deliberately does not use `ResearchStore.write`.** That path reads a
partition, concatenates, sorts and rewrites it under zstd. `run_live.py` names
it as the latency threat the whole live process is arranged around, and it is
O(partition) per flush — against a stream appending thousands of rows a minute
it would grow without bound. Appending is O(what was appended) and never reads
back.

Compaction is the other half: closed hourly files are folded into the research
store by the `store_sync` SUBPROCESS, which already exists because heavy Parquet
work must not share the trading event loop.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger('spool')


def _hour_key(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y%m%dT%H')


class FrameSpool:
    def __init__(self, root: Path, venue: str) -> None:
        self.root = Path(root) / venue
        self.root.mkdir(parents=True, exist_ok=True)
        self.venue = venue
        self._hour: Optional[str] = None
        self._handle = None

    def _path(self, hour: str) -> Path:
        return self.root / f'{hour}.jsonl'

    def append(self, record: dict) -> None:
        hour = _hour_key(record['t'])
        if hour != self._hour:
            self.close()
            self._hour, self._handle = hour, self._path(hour).open('a')
        self._handle.write(json.dumps(record) + '\n')

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._hour = None

    def closed_files(self) -> list[Path]:
        """Every hourly file except the one currently open for appending."""
        return sorted(p for p in self.root.glob('*.jsonl')
                      if self._hour is None or p != self._path(self._hour))


def compact(spool_root: Path, store, *, keep_days: float) -> int:
    """Fold closed hourly files into `venue_book_events` and remove them.

    A file still open for appending is skipped: compacting it would archive a
    prefix of the stream and then delete the rest.
    """
    spool_root = Path(spool_root)
    cutoff = time.time() - keep_days * 86400.0
    written = 0
    open_hour = _hour_key(time.time())
    for path in sorted(spool_root.rglob('*.jsonl')):
        if path.stem >= open_hour:
            continue
        rows = []
        for line in path.open():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            stamp = pd.Timestamp(rec['t'], unit='s', tz='UTC')
            rows.append({
                'venue': rec['venue'], 'symbol': rec['symbol'],
                'event_time': stamp, 'available_time': stamp, 'quality': 'valid',
                'market_ticker': rec['market_ticker'], 'seq': rec.get('seq'),
                'kind': rec['kind'], 'payload': json.dumps(rec['payload']),
            })
        if rows:
            written += store.write('venue_book_events', pd.DataFrame(rows))
        if path.stat().st_mtime < cutoff or keep_days <= 0:
            path.unlink()
        logger.info('compacted %s (%d rows)', path.name, len(rows))
    return written
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd backend/trader && pytest tests/test_spool.py -n 0 -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/trader/core/spool.py backend/trader/core/datastore.py \
  backend/trader/tests/test_spool.py
git commit -m "Archive the stream verbatim, off the rewrite path"
```

### Task 8: Run the stream as a supervised component

**Files:**
- Create: `scripts/record_stream.py`
- Modify: `scripts/run_live.py:170-176` (`COMPONENTS`), `:236-268`
  (`build_factories`)
- Test: `tests/test_record_stream.py`

**Interfaces:**
- Consumes: `KalshiStream`, `BookCache`, `FrameSpool`.
- Produces: `run(args, gate=None, cache=None) -> int`, and
  `scripts.record_stream.CACHE` — the process-wide `BookCache` that Task 9 and
  Phase 3 read.

**Two deliberate departures from every other recorder.** The stream **never
waits on `TradingGate`** — every other recorder does, because their work is
bursty Parquet writes, but a stream reader that pauses is a stream reader that
goes stale, which is the failure this design exists to prevent. Only the spool
flush takes the gate. And `--disable stream` reverts everything to REST, so the
feature is one flag from today's behaviour.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import asyncio

import pytest

from scripts import record_stream, run_live


def test_stream_is_a_known_component():
    assert 'stream' in run_live.NAMES


def test_disabling_the_stream_is_accepted():
    assert 'stream' not in run_live.Component.selected(
        run_live.NAMES, disable=['stream'])


def test_the_stream_does_not_wait_on_the_trading_gate():
    """A gated stream goes stale, which is the failure the cache exists to avoid."""
    import inspect
    source = inspect.getsource(record_stream.consume)
    assert 'gate.idle()' not in source


def test_settled_markets_are_forgotten_rather_than_sampled_as_live():
    from core.stream_book import BookCache
    from data_collection.stream.base import BookEvent
    cache = BookCache(now=lambda: 0.0)
    cache.apply(BookEvent(venue='kalshi', market_ticker='K', kind='snapshot',
                          received=0.0, seq=1, yes=[(0.3, 1.0)], no=[]))
    record_stream.retire(cache, keep={'OTHER'})
    assert cache.ladder('K') is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend/trader && pytest tests/test_record_stream.py -n 0 -v`
Expected: FAIL — `ImportError: cannot import name 'record_stream'`

- [ ] **Step 3: Implement**

```python
"""Hold the venue books open, and archive every frame that builds them.

The cache this maintains is process-wide by design: `record_ladder` samples it
(Task 9) and, in Phase 3, so does the trading loop. One book, read by everyone
who needs one, is the whole point — two samplers of one object was the defect.
"""
from __future__ import annotations

import argparse, asyncio, logging, os, time
from pathlib import Path

from core.config import series_to_symbol
from core.stream_book import BookCache
from core.spool import FrameSpool
from data_collection.stream.kalshi import KalshiStream, parse_frame

logger = logging.getLogger('stream')

CACHE = BookCache()


def retire(cache: BookCache, keep: set[str]) -> None:
    """Drop books for markets that are no longer open."""
    for ticker in [t for t in cache._books if t not in keep]:  # noqa: SLF001
        cache.forget(ticker)


async def open_tickers(client) -> dict[str, str]:
    """ticker -> symbol, for every open market on the traded series."""
    out: dict[str, str] = {}
    for series, symbol in series_to_symbol().items():
        payload = await client._request(  # noqa: SLF001
            'GET', '/markets',
            params={'series_ticker': series, 'status': 'open', 'limit': 5})
        for market in payload.get('markets', []):
            if market.get('ticker'):
                out[market['ticker']] = symbol
    return out


async def consume(stream, cache, spool, symbols, gate=None) -> None:
    """Read frames forever. **Never awaits the gate** — see the module docstring
    of run_live: a stream that pauses goes stale, and staleness is the failure
    this design removes. Only the flush below is gated."""
    since_flush = time.monotonic()
    async for event in stream.events():
        cache.apply(event)
        spool.append({'t': event.received, 'venue': event.venue,
                      'symbol': symbols.get(event.market_ticker, 'UNKNOWN'),
                      'market_ticker': event.market_ticker, 'seq': event.seq,
                      'kind': event.kind,
                      'payload': {'yes': event.yes, 'no': event.no}})
        if time.monotonic() - since_flush >= 5.0:
            if gate is not None:
                await gate.idle()
            await asyncio.to_thread(spool.flush)
            since_flush = time.monotonic()


async def run(args, gate=None, cache=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    cache = cache or CACHE
    spool = FrameSpool(Path(args.spool_root), 'kalshi')
    pem = (os.getenv('KALSHI_PRIVATE_KEY')
           or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read())
    async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                            private_key_pem=pem) as client:
        symbols = await open_tickers(client)
        retire(cache, set(symbols))
        stream = KalshiStream(client)
        await stream.connect()
        await stream.subscribe(list(symbols))
        logger.info('streaming %d markets', len(symbols))
        try:
            await consume(stream, cache, spool, symbols, gate=gate)
        finally:
            await stream.close()
            spool.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--spool-root', default='data/spool')
    parser.add_argument('--resubscribe-seconds', type=float, default=300.0)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
```

In `scripts/run_live.py`, add to `COMPONENTS` (phase is irrelevant — it is
continuous, not periodic):

```python
    Component('stream', phase=0.0),         # continuous; never gated
```

and to `build_factories`, importing `record_stream` alongside the others:

```python
    async def stream():
        return await record_stream.run(
            _recorder_args(record_stream.build_parser), gate=gate)
```

adding `'stream': stream` to the returned dict.

**`supervise` already gives the stream reconnect-with-backoff for free**, and
`run` returning is treated as a failure and restarted — which is exactly right
for a socket that closed.

- [ ] **Step 4: Run to verify they pass, then the whole suite**

Run: `cd backend/trader && pytest tests/test_record_stream.py -n 0 -v`
Expected: PASS

Run: `cd backend/trader && pytest -m "not slow"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/trader/scripts/record_stream.py backend/trader/scripts/run_live.py \
  backend/trader/tests/test_record_stream.py
git commit -m "Hold the books open under the supervisor, ungated"
```

### Task 9: Run both samplers, and compare them on real data

**Files:**
- Modify: `scripts/record_ladder.py:60-115`
- Create: `research/validate/_validate_transport.py`
- Test: `tests/test_ladder_transport_rows.py`

**Interfaces:**
- Consumes: `record_stream.CACHE`.
- Produces: `record_ladder` writes **two rows per market per minute** — the
  existing REST row with `transport='rest'`, and a cache-sampled row with
  `transport='ws'` and its `book_age_ms`. Task 3 is what lets both survive a
  read.

This is the phase that earns the migration. Nothing flips until the two agree.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pandas as pd

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent
from scripts.record_ladder import ws_row


def test_a_ws_row_carries_its_transport_and_its_age():
    cache = BookCache(now=lambda: 100.0)
    cache.apply(BookEvent(venue='kalshi', market_ticker='K', kind='snapshot',
                          received=99.5, seq=1, yes=[(0.3, 10.0)],
                          no=[(0.69, 4.0)]))
    row = ws_row(cache, ticker='K', symbol='BTC-USD',
                 now=pd.Timestamp('2026-08-26 12:00:25', tz='UTC'),
                 open_time=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                 minute=0.417)
    assert row['transport'] == 'ws'
    assert row['book_age_ms'] == 500.0
    assert row['yes_total'] == 10.0


def test_no_book_means_no_row_rather_than_an_empty_ladder():
    assert ws_row(BookCache(now=lambda: 0.0), ticker='K', symbol='BTC-USD',
                  now=pd.Timestamp('2026-08-26 12:00:25', tz='UTC'),
                  open_time=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                  minute=0.417) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend/trader && pytest tests/test_ladder_transport_rows.py -n 0 -v`
Expected: FAIL — `ImportError: cannot import name 'ws_row'`

- [ ] **Step 3: Implement**

Add to `scripts/record_ladder.py`:

```python
def ws_row(cache, *, ticker, symbol, now, open_time, minute):
    """The same ladder, sampled from the stream cache instead of a REST call.

    Returns None when the cache holds no book for the ticker. An empty ladder
    and no ladder are different claims, and writing an empty one would record a
    dead subscription as a market with nothing resting in it.

    A STALE book is still recorded, with its age. For the archive a four-second
    -old book is data, not a fault — a feature can filter on `book_age_ms`
    later, and discarding it forecloses that. The trading path makes the
    opposite choice, in Phase 3, and that asymmetry is deliberate.
    """
    ladder = cache.ladder(ticker)
    if ladder is None:
        return None
    yes = [[p, s] for p, s in ladder.yes]
    no = [[p, s] for p, s in ladder.no]
    return {
        'venue': 'kalshi', 'symbol': symbol,
        'event_time': pd.Timestamp(now).floor('min'),
        'available_time': pd.Timestamp(now), 'quality': 'valid',
        'market_ticker': ticker, 'window_open': open_time,
        'minute_into_window': round(minute, 3),
        'yes_levels': json.dumps(yes), 'no_levels': json.dumps(no),
        'yes_total': sum(s for _, s in yes),
        'no_total': sum(s for _, s in no),
        'transport': 'ws',
        'book_age_ms': round(ladder.age_seconds * 1000.0, 1),
    }
```

In `run()`, tag the existing REST row and append the WS row beside it. The
existing `rows.append({...})` call gains two keys:

```python
                                'transport': 'rest',
                                'book_age_ms': 0.0,
```

and immediately after that block:

```python
                            from scripts.record_stream import CACHE
                            paired = ws_row(CACHE, ticker=market['ticker'],
                                            symbol=symbol, now=now,
                                            open_time=open_time, minute=minute)
                            if paired is not None:
                                rows.append(paired)
```

`research/validate/_validate_transport.py`:

```python
"""Do the two samplers describe the same book?

The migration does not flip on an argument. `venue_ladder` carries both a REST
row and a WS row for the same minute — `transport` is in the event key so both
survive a read — and this prints where they disagree.

A disagreement is not automatically the stream's fault: the REST call and the
cache sample are seconds apart and a real book moves. What matters is whether
the disagreement is small, symmetric and shrinking with `book_age_ms`, or
structural.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from core.datastore import ResearchStore


def compare(store: ResearchStore | None = None) -> pd.DataFrame:
    store = store or ResearchStore(os.getenv('RESEARCH_STORE'))
    rows = store.read('venue_ladder')
    if rows.empty or 'transport' not in rows:
        return pd.DataFrame()
    key = ['symbol', 'market_ticker', 'event_time']
    rest = rows[rows['transport'] == 'rest'].set_index(key)
    ws = rows[rows['transport'] == 'ws'].set_index(key)
    both = rest.join(ws, how='inner', lsuffix='_rest', rsuffix='_ws')
    if both.empty:
        return both
    both['top_bid_rest'] = both['yes_levels_rest'].map(_top)
    both['top_bid_ws'] = both['yes_levels_ws'].map(_top)
    both['top_bid_diff'] = (both['top_bid_ws'] - both['top_bid_rest']).abs()
    both['total_ratio'] = both['yes_total_ws'] / both['yes_total_rest'].replace(0, pd.NA)
    return both


def _top(raw):
    levels = json.loads(raw or '[]')
    return max((p for p, _ in levels), default=float('nan'))


def main() -> int:
    both = compare()
    if both.empty:
        print('no paired minutes yet; let both samplers run')
        return 1
    print(f'paired minutes: {len(both)}')
    print(f'exact top-of-book agreement: '
          f'{(both["top_bid_diff"] == 0).mean():.1%}')
    print(f'median |top-of-book diff|: {both["top_bid_diff"].median():.4f}')
    print(f'median size ratio ws/rest: {both["total_ratio"].median():.3f}')
    print(both.groupby(pd.cut(both['book_age_ms_ws'], [0, 500, 2000, 10000, 1e9]),
                       observed=True)['top_bid_diff'].agg(['size', 'median']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests, then deploy and let it run**

Run: `cd backend/trader && pytest tests/test_ladder_transport_rows.py -n 0 -v`
Expected: PASS

Run: `cd backend/trader && pytest -m "not slow"`
Expected: PASS

```bash
docker compose up -d --build live
docker compose logs -f live --tail 40
```

Expected within a few minutes: `streaming N markets`, then `wrote N ladder rows`
at roughly double the previous count.

- [ ] **Step 5: Commit**

```bash
git add backend/trader/scripts/record_ladder.py \
  backend/trader/research/validate/_validate_transport.py \
  backend/trader/tests/test_ladder_transport_rows.py
git commit -m "Write both samplers side by side, and measure the disagreement"
```

---

# Phase 2 — Flip the archive, once the evidence is in

### Task 10: Retire the REST sampler and the probe

**Files:**
- Modify: `scripts/record_ladder.py`
- Delete: `scripts/probe_ws.py`, `scripts/analyse_ws_capture.py`
- Modify: `CLAUDE.md` (the market data pipeline section)

**Gate — do not start this task until `_validate_transport` reports, over at
least 24 hours of paired minutes:** exact top-of-book agreement above 99% at
`book_age_ms < 500`, a median size ratio within 0.99–1.01, and no trend in
disagreement across `book_age_ms` buckets other than the one a moving book
explains. **If it does not, that is a finding, not an obstacle** — record it in
the spec and stop. A stream that cannot reproduce the book must not become the
archive.

- [ ] **Step 1: Write the failing test**

```python
def test_the_recorder_no_longer_fetches_the_orderbook_over_rest():
    import inspect
    from scripts import record_ladder
    source = inspect.getsource(record_ladder.run)
    assert '/orderbook' not in source, (
        'the cache is the sampler now; a second REST fetch reintroduces the '
        'two-sampler defect this whole change exists to remove')
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend/trader && pytest tests/test_ladder_transport_rows.py -n 0 -v`
Expected: FAIL — the REST fetch is still there.

- [ ] **Step 3: Remove the REST path**

In `record_ladder.run`, delete the `client._request('GET', f".../orderbook")`
call and the `rows.append({...'transport': 'rest'...})` block that consumes it.
Keep the `/markets` call — it is how open markets and `window_open` are
discovered, and it is one request per series per minute, not per market.

Then delete the throwaway probe files:

```bash
git rm backend/trader/scripts/probe_ws.py backend/trader/scripts/analyse_ws_capture.py
```

The fixtures under `tests/fixtures/ws/` **stay** — they are what
`test_stream_kalshi.py` tests against.

- [ ] **Step 4: Run the tests, then verify live**

Run: `cd backend/trader && pytest -m "not slow"`
Expected: PASS

```bash
docker compose up -d --build live
docker compose logs live --tail 30 | grep ladder
```

Expected: ladder rows still being written, now roughly half the Task 9 count and
all `transport='ws'`.

- [ ] **Step 5: Update CLAUDE.md and commit**

In the "market data pipeline" section, correct the table: `venue_ladder` is
written from the stream, not a per-minute REST call, and `venue_book_events` is
the new raw tier beneath it. Note that `transport` is part of the event key.

```bash
git add -A
git commit -m "The archive reads the same book the trader will"
```

---

## What this plan does NOT do

Deliberately deferred to a second plan, written after Phase 0's findings land:

- **Phase 3 — the trading path.** `live.fetch_quotes` reading the cache with a
  REST fallback past `max_book_age_seconds`, and `_record_touch` reading the
  same book the decision read. This requires preserving `exchange_index` and
  `status` from the REST market resolution, since `orderbook_delta` carries no
  market metadata — getting that wrong reproduces the sharding failure where
  every order was refused `insufficient_balance`.
- **Phase 4 — Polymarket.** A second adapter against the interface Task 4
  defines, moving the YES→NO conversion out of `record_pm_ladder._no_levels` and
  the window-open slug convention into the adapter.
- **`fill`, `market_positions`, balance** stay on REST permanently. The venue is
  the account of record, and an authoritative polled read beats a push feed we
  would have to trust never dropped a message.
