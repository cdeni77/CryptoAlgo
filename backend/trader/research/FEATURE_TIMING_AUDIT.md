# FEATURE_TIMING_AUDIT.md — what instant each feature is built from

Audited 2026-10-21. For a decision at **offset `m`**, taken at `m + 3s`
(`decision_latency_seconds`). Every number below is measured on the real
store unless marked INFERRED.

The question this answers: **is the value the model trains on built from the
same instant as the value it is handed live?**

## Summary

| group | n | training instant | live instant | parity |
|---|---:|---|---|---|
| `geometry` | 9 | close of bar `m-1` | close of bar `m-1` | **exact** |
| `vol_state` | 11 | close of bar `m-1` | close of bar `m-1` | **exact** |
| `microstructure` | 7 | close of bar `m-1` | close of bar `m-1` | **exact** |
| `cross_asset` | 6 | close of bar `m-1` | close of bar `m-1` | **exact** |
| `offset` | 2 | clock | clock | **exact** |
| `book_quote` | 4 | ~`m` (candles, ~3s early) / `m+3.2s` (live_touch) | `m+3s`, book ~34ms old | live_touch **exact**; candles ~3s early |
| `book_flow` | 6 | minute grid, as-of `m-1` and `m-3` | minute grid, <=3s past each minute | near |
| `implied_vol` | 5 | fit at 50/35/20/10 min before the hour | same fits | **exact** |
| `cross_venue` | 4 | peer book, max age 5s | peer book, max age 5s | near |

33 of 48 features are bar-derived and have exact parity. The mismatch is
confined to the book snapshot.

## The bar features have exact parity, and it is enforced

`core/windows.py:259` reads `filled[:, offset - 1]` -- the close of bar
`m-1` -- in BOTH paths, because the backtest and the live loop call the same
builder. The data is 3-63s old at the decision, but identically so in
training and live, which is what parity means.

It is enforced rather than assumed: `core/trade_bars.py:145` marks a minute
TAINTED when a trade arrives after it closed, and stops serving that minute
from the stream. Measured, 17 such refusals in 6 hours. Without it the live
bar could include a late print the training bar never saw.

## The book snapshot does NOT have parity

| source | captured at | the book it holds |
|---|---|---|
| `candles` | stamp `m` | close of `[m-60s, m)` -- about 3s BEFORE the decision |
| `live_touch` | `m + 3.20s` | a book 0.88s old -- **the decision's own book** |
| `live_ws` / `live` | `m + 25.3s` | a book as of `m + 25.3s` -- 22s AFTER the decision |

**"Late", not "stale".** A book captured at `m+25s` is not old, it is in the
FUTURE relative to a decision at `m+3s`. `decision_age` returns a NEGATIVE
number for it, which the code reads as "seen after the decision". Refusing
these rows guards against LOOKAHEAD, not against noise -- the opposite
failure from the one the word "stale" suggests.

Live scores from the in-process stream cache at `m + 3s`, with a book ~34ms
old. Training scores from whichever source the selector picked: `candles`
before late August, `live_touch` after -- and `live_touch` IS that same
decision book, so the live era has exact parity.

Measured against `live_touch`, which is the only honest reference because it
is the book the decision used:

```
                  median   within 0.5c   >1.5c (the edge gate)
candles            1.00c      41.1%         31.9%
backfill           2.00c      16.8%         58.8%
live_ws            3.00c      15.4%         66.8%
```

A candle is one cent from the decision book. Comparing candles against
`live_ws` instead -- a book 22 seconds later -- makes them look twice as bad
and is the wrong comparison.

Measured on one window (BTC, 2026-10-20 12:00, offset 3) the four sources
disagree materially, so this is not a rounding question:

```
candles     as-of 180s   0.59 / 0.60
live_touch       182.8s  0.60 / 0.61     <- what the decision saw
live_ws          205.1s  0.66 / 0.67     <- +25s, six cents higher
live             232.0s  0.64 / 0.65
```

**The +25s recorders hold a different book, not the same book written down
late.** `decision_age` refuses them: 97.8-99.8% score a negative age, which
means "seen after the decision". They contribute 1.2% of usable rows while
accounting for 140,646 rows.

## Two defects this audit found

### 1. `live_touch` writes two rows and the selector keeps the wrong one

Every group -- 8,013 of 8,013 at decision offsets since 2026-10-15 -- has
exactly two `live_touch` rows: one carrying sizes, one not.

```
median quote_age   sized 0.423s   unsized 0.285s
```

The selector ranks on `_unusable`, then source, then age. Both rows are
`live_touch`, so **age decides, and the unsized row wins by 140ms**. After
selection only **53.2%** of live-era decision rows carry sizes.

So half the live era loses `yes_bid_size`, `depth_*` and `bid_at_touch` --
including `market_state`'s complete-case column -- in exchange for 0.14
seconds of freshness. The tie-break should prefer the row that carries more
information when the age difference is below the noise.

### 2. The one sampler that IS point-in-time is thrown away

`scripts/record_book_minutes` samples the stream cache <1s past every minute,
with sizes, at every offset. Its docstring says nothing is written to the
store because "`venue_depth` already records the same minutes" -- but the
thing `venue_depth` records is the +25s ladder recorder, which is 98%
refused.

So the pipeline persists a sample it cannot use and discards one it can.
Persisting it is the only route to depth data at non-decision offsets that a
decision could actually have used, and it accrues only forward.

## What follows

* **Training quotes should come from the venue's own publication**
  (`candles`), which is point-in-time correct at every offset, plus
  `live_touch` where it exists. The three per-minute recorders contribute
  nothing and should be dropped from `--quote-sources`.
* **Nothing should be deleted.** `venue_ladder` and the stream archive are
  the replay and transport-validation material and cannot be re-fetched.
* **Going forward**, persist the <1s sampler so the live era accumulates
  quotes captured the way the decision captures them.

## The other 15 features, audited

### `offset` (2) — exact
`elapsed_fraction`, `remaining_minutes` are clock arithmetic. No data source,
no parity question.

### `book_quote` (4) — sound, ~3s early
From candles across the whole history, 1.00c from the decision book, 100%
usable at the freshness bound. The 3-second gap is a noise question, not a
validity one: a candle at `m` is a book a decision at `m+3s` could have seen.

### `implied_vol` (5) — recoverable, but BTC-only
100% BTC / 29% ETH / 4% SOL in the backfilled era, against 96/95/24 live.
Keeping it under `--complete-cases` SKEWS the sample toward BTC rather than
merely shrinking it. See DATA.md.

### `cross_venue` (4) — fine, and the aggregate number lies

Peer freshness looks like it collapses across eras:

```
2026-01  median 0.5s   91.2% <=5s     sources: backfill
2026-05         2.5s   65.1%
2026-08         5.7s   47.0%          backfill + live
2026-10        14.1s   34.8%          backfill + live + live_touch
```

It does not. Disaggregated by source in 2026-10:

```
live_touch   median 0.3s   95.7% <=5s   <- the peer's decision-instant book
backfill            5.4s   48.1%
live               31.5s    4.7%        <- the +30s recorder
```

Polymarket has its OWN `live_touch` and it is fresh. The apparent collapse is
the aggregate being dragged down by the 31.5s recorder appearing in later
months -- the same mistake as comparing candles against `live_ws`. The peer
selector sorts on `_unusable` then age, so it takes `live_touch` where it
exists.

## Open: the bar parity is UNMEASURED on real data

33 of 48 features rest on the claim that a live bar equals the training bar.
The rule is shared code, and `tests/test_trade_bars.py::
test_rest_then_stream_equals_the_rest_candles_it_replaces` asserts the stream
bar equals the REST candle it replaces -- but that test builds BOTH from the
same synthetic trade list. It proves the aggregation is equivalent given
identical trades. It does not prove we saw the trades Coinbase counted.

`reliable()` bounds the risk: a minute is served from the stream only when it
is final, untainted by a late print, and fully inside one continuous coverage
interval. Anything else falls back to REST, which IS the training source. But
bounded is not measured.

**It is directly testable.** The spot trade archive holds the tape, so the
stream bar for past minutes can be rebuilt and compared against the stored
`minute_bars` for the same minutes -- the bar-side equivalent of the replay
validation that proved the ladder rebuild bit-for-bit on 8,169 of 8,169
minutes. Until that is run, "33 of 48 are identical" is true of the RULE and
unproven of the DATA.
