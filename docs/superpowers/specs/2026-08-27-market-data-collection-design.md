# Market data collection: ledger-driven, two venues, maximum fidelity

**Status:** design approved 2026-08-27, not yet implemented.
**Supersedes:** the ad-hoc collection in `research/collect/_collect_book.py`,
`research/collect/_collect_pm.py` and the `_gap_fill*.sh` family.

## Why this exists

The previous collection layer produced a dataset nobody could state the
coverage of. Over one night, four separate claims about what data exists were
measured, believed, and then disproved:

| claim | reality | cause |
|---|---|---|
| Kalshi book starts 2026-06-19 | 2026-01-08 | probe omitted the required `start_time`/`end_time`, so real books read as empty |
| Polymarket book starts 2025-12-31 | ~2026-01 | binary search assumed coverage is monotone |
| Kalshi SOL book starts 2026-06-03 | same as BTC/ETH | same, on a market thin enough that a 3-window sample missed it |
| Kalshi density falls through May–June | the dip is real, but the first measurement of it was not | failed requests counted as "no book" |

Every one conflated **"I got no data"** with **"no data exists"**, and every one
erred toward understating what is available. The cost was concrete:
`BOOK_COVERAGE_START` sat five months late, capping the Kalshi book at ~70 days
of a retrievable ~230. That constant is the source of the "70 days of ONE
venue" figure that `CLAUDE.md` calls the project's binding constraint.

This design's central claim is that the confusion was **structural, not
careless**: nothing in the old pipeline could represent "we never asked", so
coverage had to be inferred from absence, and absence has three causes that
look identical from the outside. The ledger below makes that distinction a
column.

## Scope

Kalshi and Polymarket, BTC/ETH/SOL, 15-minute up/down markets, from
2026-01-08 to present. Coinbase `minute_bars` is already held (5 years,
~2.63M bars/symbol) and is out of scope for collection — it is in scope for
the feature map only.

Polymarket is treated as **trading-capable from day one**. It is used first
for cross-venue features, may later become a live signal in the decision path,
and may eventually be traded. The schema must not have to change for that.

Two consequences follow, and only the second is in scope here. Cross-venue
features require exact window alignment, which this design must get right now
(see Alignment). Polymarket entering the *live* decision path would make it
inherit that path's latency budget — the loop already decides at +76s against
a ~3s edge budget — but no change to `scripts/live.py` is proposed here. This
spec covers historical collection only; the live path is a separate piece of
work that this schema is designed not to obstruct.

## Ground truth (measured 2026-08-26/27)

Sampled 6 windows per month per asset, with request failures counted
separately from empty books. `err` was 0 across every Kalshi cell, so the
OK/EMPTY split is trustworthy.

### Kalshi — complete

| month | ok / 18 | note |
|---|---|---|
| 2025-11, 2025-12 | 0 / 18 | before book coverage |
| 2026-01 | 15 / 18 (83%) | coverage begins |
| 2026-02 | 18 / 18 | |
| 2026-03 | 18 / 18 | |
| 2026-04 | 16 / 18 (89%) | |
| 2026-05 | 9 / 18 (50%) | **real Predexon gap** |
| 2026-06 | 7 / 18 (39%) | **real Predexon gap** |
| 2026-07 | 18 / 18 | |
| 2026-08 | 18 / 18 | |

Overall **119/144 = 82.6%** from 2026-01-08 onward. All three assets behave
identically; SOL is not late.

### Polymarket — complete

0/18 in 2025-11 and 2025-12. **18/18 in every month from 2026-01 through
2026-08**, all three assets, with zero errors and zero empties.

Polymarket's book coverage is therefore *better* than Kalshi's — 100% against
82.6% — and it does **not** share the May/June gap. That gap belongs to
Predexon's Kalshi ingestion specifically, which is worth knowing: it means
cross-venue windows in May and June will be bounded by Kalshi alone (50% and
39%), not by both venues degrading together.

The practical consequence for the plan is that Polymarket's ~17-point
advantage in yield does not translate into more *paired* windows. Cross-venue
features exist only where both venues are `ok`, so the joint coverage is
Kalshi's.

### Other measured facts

* Kalshi markets exist from **2025-12-10** (pagination exhausted — a real venue
  floor, matching Kalshi's December 2025 launch). Books start about a month
  later. Our previous store began 2026-01-06, which was our collector's limit,
  not the venue's.
* Kalshi's own API purges markets past ~2 months; a ticker held in our own
  store 404s there. **Predexon is the only source for Kalshi history.**
* Polymarket TWAP markets exist from **2025-10-10** (BTC/ETH) and
  **2025-10-28** (SOL), with real volume ($42,033 traded on BTC's first day)
  and zero retrievable order book. Market existence and book availability are
  different boundaries.
* Polymarket ran an **earlier, different instrument** before the current one:
  `{asset}-up-or-down-15m-{ts}` from 2025-09-12 settles on the Chainlink spot
  stream read *at the end* of the range; the current
  `{asset}-updown-15m-{ts}` settles on the Chainlink **TWAP-60s** stream — the
  average *over* the range. An endpoint reading and a 60-second time-average
  are different random variables (a time-average carries a third of its
  endpoint's variance), so the eras must never be pooled. Only the TWAP era is
  in scope.

## Architecture

One ledger, one storage schema, one orchestrator, one monitoring surface; two
thin venue fetchers, because the APIs differ genuinely rather than
incidentally. Kalshi returns paginated forward tick deltas requiring
exhaustion; Polymarket returns a snapshot list in a single call. Forcing one
control flow onto both buys nothing.

```
              ┌──────────────────────────────┐
  catalogs ──▶│  collection_ledger (SQLite)  │◀── coverage report
              │  one row per (venue,symbol,  │
              │  window_open)                │
              └──────────────┬───────────────┘
                             │ WHERE status IN ('pending','error')
                             ▼
                     ┌───────────────┐
                     │ orchestrator  │  rate limit · retry · breaker · lockfile
                     └───┬───────┬───┘
                         │       │
              ┌──────────▼─┐   ┌─▼───────────┐
              │ kalshi     │   │ polymarket  │   fetch(market_id, window)
              │ fetcher    │   │ fetcher     │     -> (ticks, status, meta)
              └──────────┬─┘   └─┬───────────┘
                         └───┬───┘
                             ▼
              archive layer (full ladders, compressed)
                             │
                             ▼
              derived layer (13-field packed summary)
```

### The ledger

SQLite, consistent with how `scripts/scrape.py` already holds mutable working
state before `sync_store` converts to Parquet. Parquet is append-oriented and
wrong for rows that get updated; the research store remains the output, not the
bookkeeping.

| column | meaning |
|---|---|
| `venue`, `symbol`, `window_open` | identity; primary key |
| `market_id` | the venue's own ticker or slug, **resolved, never constructed** |
| `status` | `pending` · `ok` · `empty` · `error` · `skipped` |
| `attempts`, `last_attempt_at`, `last_error` | retry bookkeeping |
| `snapshots`, `bytes` | what was actually retrieved |

Three invariants:

1. **Seeded before fetching.** Every window in range gets a `pending` row up
   front, so "never asked" is queryable rather than inferred.
2. **`empty` and `error` never merge.** `empty` means the venue answered and
   there was no book — a result, recorded, never retried. `error` means the
   request failed — retried with backoff, capped. Collapsing these caused every
   failure in the table at the top of this document.
3. **Resume is a query, not a cursor.** Restart selects
   `status IN ('pending','error')`. There is no saved position to corrupt and
   nothing to reconcile after `kill -9`. The cursor added to `_collect_pm.py`
   on 2026-08-26 becomes redundant and should be removed with it.

### Storage: two layers

**Archive** — the full ladder at every book change, stored as Parquet with
zstd compression, partitioned by `venue/symbol/month` to match the existing
research store layout. Raw ladders run ~1.1 MB/window uncompressed but are
highly repetitive (a tick is usually one level changing), so columnar
compression should hold the ~133k-window corpus in the tens of GB against
477 GB free. Kalshi destroys books on settlement, so
anything not captured now is gone permanently, and Predexon — the only source —
already has holes. This layer answers questions not yet asked: queue position,
ladder slope beyond 5c, replenishment shape.

**Derived** — the existing 13-field packed summary (`ts`, `best_bid`,
`best_ask`, `bid_at_touch`, `ask_at_touch`, `bid_1c`, `ask_1c`, `bid_5c`,
`ask_5c`, `bid_levels`, `ask_levels`, `bid_vol`, `ask_vol`), computed at write
time. Features read this; a training run never touches the archive. Fully
regenerable, so a new summary field is a recompute rather than a re-scrape.

### Alignment

Both venues run the same quarter-hour grid, so `window_open` (UTC) is the join
key. They encode it differently:

* Kalshi `KXBTC15M-26JAN061730-30` → **close** time, in **Eastern**
* Polymarket `btc-updown-15m-1767268800` → **open** time, **unix UTC**

Both normalise to one `window_open`. **The derived value is cross-checked
against the venue's own `open_time`/`end_time`/`title`, and a mismatch is
recorded as `error` rather than accepted.** Reading the Polymarket slug as a
close shifted every window fifteen minutes and nothing raised — every window
was valid and every book was real; it surfaced only as a 49.85% settlement
agreement. A parse that can be wrong-but-plausible needs a second opinion.

Cross-venue features are defined only where both venues are `ok` for the same
window. The ledger makes that a join, and the coverage report states how many
windows qualify before training rather than after.

### Orchestration

* **Fetcher contract:** `fetch(market_id, window_open) -> (ticks, status, meta)`.
  Fetchers know nothing about retries, rate limits, resume or storage.
* **One global rate limiter, enforced by a lockfile.** The 1 req/s Predexon
  bucket is org-wide. On 2026-08-26 concurrent probes and a backfill competed
  for it and the resulting 429s were silently counted as "no book". Only one
  collector process may run; this is enforced, not remembered.
* **Retry policy:**

  | condition | status | retried |
  |---|---|---|
  | 429, 5xx, timeout, connection reset | `error` | yes, backoff, ~5 attempts |
  | 4xx other than 429 | `error` + code | no |
  | venue answered, no book | `empty` | **no** |
  | venue answered with ticks | `ok` | — |

* **Circuit breaker.** If more than 25% of the trailing 40 attempts end in
  `error`, the orchestrator pauses and exits non-zero rather than continuing.
  Without it, a two-hour venue outage marches through thousands of `pending`
  rows converting them to `error` at full speed, and "the venue was down"
  becomes indistinguishable from "these windows are broken". The threshold is
  a starting value, not a measured one; it should be loose enough that the
  measured ~17% `empty` rate never trips it, which it is, because `empty` is
  not `error`.
* **Monitoring:** progress line with done/total, rate, ETA and error rate per
  venue; the ledger is queryable mid-run (`GROUP BY status`).

## Feature map

Today every feature in `core/features.py` is computed from Coinbase bars
alone. No feature group reads venue book data. The collection below is
therefore almost entirely new capability.

| source | held? | feeds |
|---|---|---|
| Coinbase `minute_bars` | yes, 5y | the **label** (strike/settlement as 1-min means), `sigma_remaining`, and all five existing groups: `vol_state`, `microstructure`, `cross_asset`, `geometry`, `clock` |
| Kalshi book | **to collect** | `book_state` (spread, mid, depth at touch, depth 1c/5c, imbalance, level counts, quote age); `order_flow` from the archive (tick rate, sweeps, queue depletion, replenishment) |
| Kalshi book — market price | **to collect** | unblocks `market_windows` and `model_minus_market`, two of eighteen promotion gates that currently read NaN because a backtest has no book |
| Polymarket book | **to collect** | the same `book_state` fields, mirrored; `cross_venue` (mid difference, spread difference, depth ratio, lead-lag) |
| `venue_implied_vol` | yes, live, unused | `implied_sigma_per_min` from the KXBTCD strike ladder (measured R² > 0.95) — a **forward-looking** estimate of `sigma_remaining`, the one quantity the barrier framing says must be forecast. Every existing vol feature is backward-looking realised vol. |
| settlements, both venues | **to collect** | label validation only (~3% noise). Deliberately not a feature. |

`clock` remains as the labelled control regardless of what else is added: the
previous project's best-scoring grid cell was its own control, and that was
its most useful result.

## Execution

**Phase 0 — rebuild catalogs (~1.3 h).** Kalshi settlements per series (~700
requests) for tickers, window opens and results. Polymarket discovery (~3,550
pages) for slugs, token IDs and `winning_side`. Both yield the venue's own
identifiers.

**Phase 1 — seed the ledger.** Local, instant.

**Phases 2–3 — fetch books (~47 h): ~28 h Kalshi, ~19 h Polymarket,
interleaved by month.** Running all of one venue then the other means stopping
at 60% yields zero cross-venue windows — the signal of most interest arrives
last. Interleaving makes each completed month usable on both venues
immediately, and any stop leaves a clean aligned prefix. Same total cost.

Cost is driven by windows **attempted** (~66,500 per venue: 231 days × 96
windows × 3 assets), not windows with data, since a request is spent either
way to learn which it is.

**Phase 4 — derived layer + coverage report.**

### Validation checkpoints

1. **After Phase 0** — both venues' window grids land on the same quarter-hour
   boundaries and counts agree. Catches the slug-shift class of error before
   47 hours are spent.
2. **~500 windows into Phase 2** — compare backfilled depth against the
   **live-recorded ladders preserved from the old store** (`venue_ladder`,
   `pm_ladder`) on overlapping windows. This is the only independent evidence
   the backfill describes the same object.
3. **After each month** — actual density against the census forecast (82.6%
   Kalshi, with the May/June dip expected). A material deviation is a signal,
   not a number to accept.

## What was deleted, and what was kept

Deleted 2026-08-27 (all re-scrapable): `data/book_full.jsonl` (1.29 GB),
`data/pm_prices.jsonl` (627 MB), `data/pm_markets.jsonl`, and the
`venue_quotes`, `venue_settlements` and `venue_depth` tables.

Kept, because it cannot be re-created: `minute_bars` (Coinbase, 5 years),
`venue_ladder` and `pm_ladder` (live-recorded raw ladders),
`venue_implied_vol` and `data/iv_ladder.jsonl` (live-recorded). `venue_depth`'s
live portion rebuilds from the ladders via `scripts/build_depth.py`.

## Open questions

* Whether the May/June Predexon gap is permanent or backfilled by the provider
  later. If it is ever filled, the ledger's `empty` rows for those months would
  need a deliberate re-sweep — `empty` is not retried by design.
* Archive-layer compression ratio is estimated, not measured. Phase 2's first
  month settles it; if the corpus trends past ~100 GB the archive should be
  reconsidered before continuing.

---

## As built (2026-08-27)

Implemented, validated and running. What the design got wrong, and what the
measurements said instead.

### Estimates that were wrong

| estimate | actual | why |
|---|---|---|
| catalog by pagination | replaced entirely | pagination degraded with cursor depth — 1.8 days of history a minute at the start, 0.55 four hundred pages in, extrapolating past 6h. Targeted `market_slug=` lookup (50 per request) does the same corpus in 26 min. |
| archive "tens of GB" | 202GB raw, ~13GB gzipped | raw ladders are 97% of the bytes and compress 19.5-33x. Uncompressed would have been the largest thing in the repo by two orders of magnitude. |
| ~47h to collect | ~52h | the trial measured August, and January windows are 2-4x heavier (Kalshi 2,762 snapshots a window against 1,341; Polymarket 1,463 against 358). Kalshi's January average exceeds the 2,000-per-page cap, so those windows legitimately cost two requests. |
| 1 req/s is the constraint | transfer is | a window with a large book takes 1.5-5.4s while issuing ONE request — 0.19 req/s against a 1 req/s budget. Eight concurrent fetches behind one thread-safe limiter hold 0.84 req/s and cut the run from ~77h to ~52h. |

### Validation results

**Checkpoint 1 — window grid.** Every decoded Kalshi window lands on the
quarter-hour boundary, evenly across :00/:15/:30/:45. Zero off-grid.

**Cross-check on discovery.** Pagination and grid construction are independent
methods; on the 10,665 markets both found they agree exactly on token_id,
window_open and result — zero mismatches. The 288 pagination-only markets are
all at or after the grid's deliberate end boundary.

**Checkpoint 2 — collected book vs live-recorded book.** The raw agreement
percentage turned out to be the wrong statistic: the live recorder stamps an
observation when its request returned and the backfill stamps a tick when the
venue published it, so a naive minute-level match compared instants up to 45
seconds apart and reported 26% agreement with a 3c median gap. Matched
as-of on the actual instant:

```
kalshi        tol      n   exact    <=1c    <=2c  median
               5s  1,042  47.9%   88.6%   94.4%    0.0c
              20s  1,150  44.3%   83.4%   90.4%    0.1c
              90s  1,216  42.2%   80.3%   87.6%    0.2c

polymarket     5s    336  46.1%   77.1%   87.8%    0.1c
              20s    546  36.8%   65.0%   76.7%    1.0c
              90s    627  35.2%   65.1%   76.1%    1.0c
```

The SHAPE is the test, not the level. Agreement improving as the match
tightens is what two views of one book look like when their clocks differ; a
shifted window, the wrong market or mangled units would disagree structurally
and tightening would not help. The verdict now tests that directly rather than
against a fixed threshold, which would otherwise have been tuned until it
passed.

**Independent corroboration of the packing.** The derived layer's average
spread over 53,427 real snapshots is **1.05 cents**, reproducing the
one-cent spread `CLAUDE.md` records from a live order book by a completely
separate path. A units error would have shown a 2x or 100x spread.

### Sizes, measured

| layer | per window | corpus (130,624 windows) |
|---|---|---|
| gzipped JSONL, during collection | ~119 KB | ~15 GB |
| derived Parquet (what features read) | ~56 KB | ~7.4 GB |
| derived + archived ladders | ~200 KB | ~26 GB |
