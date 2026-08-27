# Streaming venue books over WebSocket

**Status:** design approved 2026-08-26, not yet implemented.
**Scope:** Kalshi and Polymarket order books. Ledger reads stay on REST.

## Why

The venue feeds are being collected so they can become live model inputs. That
raises a requirement polling cannot meet, and it is not the one it looks like.

**It is not latency.** `project-latency-budget` measured directly from the trade
tape — microsecond timestamps, 360 settled windows — that below 30 seconds the
market gains no information distinguishable from noise. The curve turns clearly
positive only at 45-60s. So "the decision reads a fresher book" is the weakest
available justification, and a design sold on it would be optimising across a
region where nothing measurable happens.

**It is that we sample the same object two different ways.** `record_ladder`
takes a REST snapshot at roughly +25s past the minute. `live._record_touch`
issues its own, separate REST fetch at the decision instant. If a book-derived
feature enters the model, it is fitted on the first sampler and scored on the
second: two clocks, two round trips, one feature vector. That is the same defect
the repository already documents for `levels_bid`/`levels_ask` across backfill
and live, where the measured ratio is 0.579 — a feature that reads as signal and
is actually reporting which pipe the row came through.

A single continuously-maintained book, read by both the recorder and the
decision, collapses the two samplers into one. It is the same argument as one
`decide()`.

**A secondary gain, which is real but not the driver:** a once-a-minute snapshot
discards every intra-minute change. `record_ladder`'s own docstring argues the
book is irrecoverable — no historical endpoint carries resting size at any price
— and by that argument the 59 seconds we currently throw away are as
irrecoverable as the second we keep.

## What the venues offer

| | Kalshi | Polymarket |
|---|---|---|
| endpoint | `wss://api.elections.kalshi.com/trade-api/ws/v2` | `wss://ws-subscriptions-clob.polymarket.com/ws/market` |
| auth | RSA-PSS over `timestamp + "GET" + "/trade-api/ws/v2"` | none (public) |
| book channel | `orderbook_delta` (snapshot then deltas) | `book` + `price_change` |
| other useful | `ticker_v2`, `trade`, `market_lifecycle_v2`, `fill`, `market_positions` | `last_trade_price` |

Kalshi's handshake reuses `KalshiClient._headers` **verbatim**. That method
takes the already-prefixed path and signs `timestamp + METHOD + path`; it is
`_request` that hardcodes the `/trade-api/v2` prefix. So
`_headers('GET', '/trade-api/ws/v2')` is a correct WS handshake with no change
to the signing code — the prefix difference lives entirely in the caller.

**Kalshi publishes no gap-detection contract.** The documentation describes
snapshot-then-delta and mentions a `subscription buffer overflow` error, but
says nothing about sequence numbers or how to detect a dropped message. This is
the single largest unknown in the design and is resolved by measurement, not by
assumption — see Phase 0.

## Phase 0 findings (measured 2026-08-27, live container, 45s + 200s captures)

Four things the documentation did not say, three of which change the design.

### 1. The rate is ~150x the estimate this spec was written around

**777 frames/second across three markets** — BTC 492/s, ETH 190/s, SOL 95/s —
measured mid-window (03:09-03:10 UTC on the 03:00-03:15 window). The spec's
retention arithmetic assumed ~5 msg/s/market and concluded ~200 MB/day was
affordable. The real figure is ~67M frames/day.

**So "store every frame verbatim, forever" is not affordable** and the retention
policy becomes a real decision rather than a knob with a comfortable default.
See "Retention, revised" below.

### 2. `seq` is global per SUBSCRIPTION, not per market

This is the one that would have shipped as a silent, total failure.

Globally the sequence is perfectly contiguous — 1 to 34,956, every step +1,
across all three markets on one `sid`. Per market it reads `1, 9, 10, ...`.

A per-market gap check — which is what "snapshot then incremental updates"
naturally suggests and what this design originally specified — would therefore
flag **every single delta** as a gap, mark every book permanently suspect, and
either resubscribe in a loop or serve a book it believed was corrupt.

**Gap detection is per-connection.** One missed `seq` means every book on that
subscription is suspect, so the repair is to resubscribe all of them and take
fresh snapshots — not to repair one market.

### 3. The field names are not the ones the documentation implies

```
orderbook_snapshot  msg: market_ticker, market_id, yes_dollars_fp, no_dollars_fp
orderbook_delta     msg: market_ticker, market_id, price_dollars, delta_fp,
                         side, ts, ts_ms
both                top-level: type, sid, seq, msg
```

Prices and sizes are **fixed-point strings** (`"0.5400"`, `"-5.00"`), matching
the `_fp`/`_dollars` convention `KalshiClient` already handles for REST. `delta_fp`
is a **signed change**, not a resulting size — confirming that `BookEvent` needs
its `absolute` flag rather than a conversion at the adapter.

Non-book frames on the same socket: `{"type": "subscribed", "id": 1, "msg":
{"channel": ..., "sid": 1}}`, which carries no `seq`. The parser must ignore it
rather than raise.

### 4. The ticker format has gained a suffix

Live now: `KXBTC15M-26AUG262315-15`, event `KXBTC15M-26AUG262315`. `CLAUDE.md`
documents `KXBTC15M-26AUG230030` with no suffix and cites the absence of a
strike suffix as the tell for an up/down market. `strike_type` is still
`greater_or_equal` and `open_time`/`close_time` are still a 15-minute window, so
these are the right markets — but the documented tell is now wrong, which is one
more reason resolution must keep asking the venue rather than matching a shape.

### Retention, revised

The raw stream cannot be kept indefinitely, so the archive has two tiers with
different lifetimes:

* **`venue_book_events` — bounded retention, default 14 days.** The complete
  stream, for research that needs intra-second book dynamics.
* **`venue_ladder` — kept forever, sampled from the cache.** This is the tier
  features are built on, and it is what the live path will read.

**The "store it verbatim" argument survives in a different form.** It was made
to stop a projection chosen for one question foreclosing the rest. A Kalshi
delta is a flat, fully-typed record — ticker, price, signed delta, side, ts_ms,
seq — with no nested structure. Storing those as typed columns is therefore not
a projection; it *is* the message, minus a redundant `market_id`. It also
compresses roughly an order of magnitude better than the same content as JSON
text, which is what makes even 14 days affordable. The one-applier invariant is
untouched: replay still folds the same fields through the same function the live
cache uses.


## Architecture

```
  KalshiStream ─┐                                      ┌─→ live.fetch_quotes  (read, no I/O)
  (auth, WS,    │                                      │
   orderbook_   ├─→ BookEvent ─→ BookCache ─→ ladder ──┼─→ record_ladder      (sample, no I/O)
   delta)       │   (normalized)   (per market)        │
  PolyStream ──┘                                       └─→ event spool        (append-only)
  (public, book/price_change)
```

### Modules

**`data_collection/stream/base.py`** — `BookEvent` and the `VenueStream`
protocol: `connect()`, `subscribe(tickers)`, `events()`. Nothing venue-specific
crosses this line.

**`data_collection/stream/kalshi.py`** — handshake via `KalshiClient._headers`,
subscribes `orderbook_delta` and `market_lifecycle_v2`, emits normalized
snapshot and delta events.

**`data_collection/stream/polymarket.py`** — public connection, `book` and
`price_change`. **Two known traps are fixed here, at the venue boundary, so they
can exist in exactly one place:**

* The YES→NO conversion currently in `scripts/record_pm_ladder._no_levels`.
  Polymarket serves asks YES-denominated; Kalshi's `no_levels` are NO bids.
  Storing them as served put a 0.51 YES ask in the column holding a 0.51 NO bid:
  same name, opposite meaning, no exception anywhere.
* Window identity. A Polymarket slug's trailing unix stamp is the window's
  **open**, not its close. Read as a close it shifts every window fifteen
  minutes and nothing raises, because every window is a valid window and every
  book is a real book.

**`core/stream_book.py`** — `BookCache`: ticker → ladder, per-market update time,
sequence state. Pure, synchronous, no network, no venue knowledge. This holds the
state a wrong answer would come from, so it carries the heaviest unit tests.

### The staleness contract

Every read is `cache.ladder(ticker) -> (levels, age_seconds) | None`, and `None`
is a real answer.

**A cached book that quietly stops updating is worse than a REST call**, because
it looks healthy. Staleness is therefore a first-class field rather than an
implementation detail, and callers state their own tolerance rather than
inheriting a global one — because they do not want the same thing:

* **`live.fetch_quotes` fails closed.** Past `max_book_age_seconds` it falls
  back to a REST fetch and labels the row's transport accordingly. It never
  trades a book it cannot date. **Default 10s**, and the number is derived
  rather than picked: the tape measurement puts the market's information gain
  below the noise floor out to ~30s, so anything under that is free, while a
  book older than a few seconds signals a broken connection rather than a quiet
  market. Ten seconds is inside the free region with room to spare, and a
  breach means "the transport is sick", not "the market is slow".
* **`record_ladder` fails honest.** It records the age into the row rather than
  refusing. For the archive, a book four seconds old at sampling time is data,
  not a fault; a feature can filter on it later, and discarding it forecloses
  that question.

A single global staleness rule would have to pick one of these and be wrong for
the other.

## The archive

### Existing tables keep their shape

`venue_ladder` and `pm_ladder` remain schema twins on the same 1/min cadence,
still joining on `(symbol, window_open, minute)`. Only the source changes:
sampled from the cache instead of a REST round trip. Two columns are added to
both:

* **`transport`** — `'rest'` or `'ws'`. Non-negotiable. Changing how a ladder is
  sampled creates exactly the cross-source incomparability already measured for
  level counts, and an unlabeled row makes it invisible.
* **`book_age_ms`** — cache staleness at sampling time, so a future feature can
  filter on freshness rather than trust it.

**Open verification, not an assumption:** whether DuckDB's `read_parquet` over a
mixed set of old and new partitions unions by name or errors. Resolve before
migrating.

### One new dataset: `venue_book_events`

Identity columns for partitioning and joining — `venue, symbol, event_time,
available_time, quality, market_ticker, window_open, seq, kind` — plus **the raw
message as JSON text**. Not a parsed long-format table of level changes.

**The reason is one applier.** The function that folds a message into a ladder is
the same function the live cache runs and the same one a replay runs. Store a
parsed projection and there are two interpretations of one stream, free to drift.
Store the message and the archive *cannot* disagree with the live book about what
the stream meant. This is the same argument as one `decide()`, and it is what
`record_ladder` means by a projection foreclosing every other question.

### The write path is the constraint

`ResearchStore.write` reads a partition, concatenates, sorts and rewrites it
under zstd. `run_live.py` already names that as *the* latency threat, and it is
O(partition) per flush against a stream appending thousands of rows a minute.
So the tick sink never touches it:

1. The stream writer appends to an hourly, append-only spool file per
   `(venue, hour)` — no read-back, no merge — flushed in `to_thread`.
2. Closed spool files are compacted into `venue_book_events` partitions by the
   **existing `store_sync` subprocess**, already hourly, already out-of-process,
   and already the designated home for blocking Parquet work.

Precedent for both halves is in the tree: `data/iv_ladder.jsonl` is append-only,
and `store_sync` exists precisely because heavy CPU must not share the trading
event loop.

### Retention

Order of magnitude: 6 markets at 5 msg/s is ~2.6M messages/day; at 80 bytes
compressed, ~200 MB/day and ~70 GB/year, against 477 GB free. Affordable — but
the real rate could be several times that near settlement, which is exactly when
the book matters most. Phase 0 measures it, and the spool carries an explicit
retention knob rather than the answer arriving as a full disk.

## Live path integration

**`fetch_quotes` splits into two reads with different lifetimes.**

Market *resolution* stays REST: once per window per symbol, not latency-bound,
and markets are resolved by asking the venue rather than built from a pattern.
Resolution also drives subscription — a resolved market is subscribed, a settled
one unsubscribed.

Two things a naive swap would break, both of which have bitten this repository:

* **`orderbook_delta` carries no market metadata.** `venue_exchange_index()`
  reads `exchange_index` off the quotes to choose the balance shard, and
  `quote.tradeable()` reads `status`. Neither is in the book feed. Metadata
  therefore comes from the REST resolution, cached per window and refreshed on
  `market_lifecycle_v2`; only prices and sizes come from the stream. Getting this
  wrong reproduces the sharding failure where every order was refused
  `insufficient_balance`.
* **`_record_touch` stops fetching the orderbook** and reads the same cache the
  decision read. Today those are two REST calls at two instants, so the recorded
  touch is not the book that was traded against. This is the alignment fix and
  the concrete payoff.

**`run_live` gains a `stream` component**, supervised like the rest — `supervise`
gives it reconnect with backoff for free. Two deliberate departures:

* **The stream never waits on `TradingGate`.** Every other recorder does, because
  their work is bursty Parquet writes. A stream reader that pauses is a stream
  reader that goes stale, which is the failure this design exists to prevent.
  Only its spool flush takes the gate.
* **`--disable stream` reverts everything to REST**, so the whole feature is one
  flag away from today's behaviour. That property is required on anything
  touching the trading path.

## Out of scope, with reasons

* **`fill`, `market_positions`, balance stay on REST.** "The venue is the account
  of record" is better served by an authoritative polled read than by a push feed
  we would have to trust never dropped a message. Reconciliation is not
  latency-bound.
* **`record_implied_vol` stays on REST.** KXBTCD is an hourly series on a
  different cadence and is not on the alignment path. `ticker_v2` would work; it
  is additive, and additive work is not this design.

## Tests

All offline, from fixtures captured in Phase 0:

1. **WS-vs-REST agreement** — replay a captured frame sequence, assert the
   reconstructed ladder equals a REST snapshot taken at the same instant. This is
   what proves delta application is *correct* rather than plausible; the same
   idea as `research/validate/_validate_depth.py`.
2. **Staleness** — injected clock; assert `fetch_quotes` falls back to REST past
   tolerance and never returns an undated book.
3. **Gap/resync** — inject a sequence gap; assert resubscribe-and-resnapshot
   rather than continuing on a corrupt book.
4. **Denomination** — assert the Polymarket adapter emits NO-denominated
   `no_levels`.
5. **Metadata preservation** — assert a cache-served `Quote` carries
   `exchange_index` and `status` from the REST resolution.

## Rollout

Trading is currently halted on the daily drawdown breaker; scoring and recording
continue with no new entries. Every phase below is therefore observable against
real books with real credentials and no order risk.

| phase | what runs | what it answers |
|---|---|---|
| 0 (throwaway) | connect, subscribe, log | message rate; does `seq` exist; subscription limits; WS-vs-REST agreement — and the fixtures every test needs |
| 1 | stream + cache + spool; `record_ladder` still REST | do the two samplers agree, minute by minute, on real data |
| 2 | `record_ladder` reads the cache | archive migrates once agreement is demonstrated |
| 3 | `fetch_quotes` reads the cache, REST fallback | trading path, last and behind a flag |
| 4 | Polymarket adapter | second venue against a proven interface |

**Phase 0's output is an answer and a fixture set, labeled throwaway** — not code
that is kept. Its findings may change the schema, and a schema chosen before the
measurement would be a guess wearing a spec's clothes.
