# DATA.md — how the training set is built, and what every number in it is

Companion to `CLAUDE.md` (why the strategy is shaped the way it is) and
`research/RESEARCH_PROTOCOL.md` (how a result is allowed to be believed). This
file is about the **data itself**: where each number comes from, what is wrong
with it, what cannot be recovered, and what the pipeline does from raw bytes to
a scored row.

Everything stated as measured here was measured against the live venue, and the
date is given. Where something is an assumption, it says so.

---

## 0. The unit of data is a WINDOW, not a row

A Kalshi 15-minute market opens on a quarter-hour, records the price there as
its **strike**, and settles on whether the price at the next boundary is at or
above it. We score four decision offsets (+3, +6, +9, +12 minutes) per window,
on three symbols.

```
1 window  ->  3 symbols  ->  4 offsets  =  12 rows
```

Those 12 rows are **not 12 observations**. The four offsets share one
settlement; the three symbols are ~0.7 correlated inside a window; consecutive
windows chain, because a window's strike is the previous window's settlement.

**All inference resamples windows.** Quoting a row count as a sample size is a
protocol violation (`RESEARCH_PROTOCOL` §0). This bit us directly on
2026-10-21: 642 live settlements looked like a comfortable sample against the
500-trade decision rule, but they were 275 windows over **4 days**, and the
window-clustered interval on realised edge was −5.23 .. +2.14 pp — useless in
either direction.

---

## 1. The datasets, what they are, and what is wrong with each

### `minute_bars` — the price series

| venue | span | what it is |
|---|---|---|
| `coinbase_spot` | 2021-08-24 → now, 8.1M rows | one-minute OHLCV from Coinbase |
| `cfbenchmarks` | being backfilled | CF Benchmarks' own index (BRTI / ETHUSD_RTI / SOLUSD_RTI) |

**The problem Coinbase has:** the markets settle on BRTI, and the market prices
on BRTI. We compute displacement, `sigma_remaining` and **35 of the deployed
artifact's 55 features** from Coinbase.

Measured 2026-10-21 over 400 consecutive BTC windows:

```
sd of the Coinbase-vs-BRTI basis          0.96 bp
autocorrelation at 15 minutes            +0.094      <- white noise
sd of the change over 15 minutes          1.29 bp
fraction that cancels between strike and decision   ~5%
```

`CLAUDE.md` assumed *"both ends use the same approximation, so most of its bias
cancels in the comparison."* **That is wrong.** The basis is white noise at the
window scale, so it compounds rather than cancels. ~1.29 bp reaches
displacement, which at +12m against ~8 bp of remaining sigma is **~6 pp of
probability error** — against a target edge of **1.5 pp**. The signal is
smaller than the noise in its own input.

The fix is to run the pipeline on `venue='cfbenchmarks'`. `load_minute_bars`
reads `minute_bars` with `venue=config.venue`, so this is a config change, not
a rewrite — and it gives a clean A/B.

**Volume is carried over from Coinbase on the index bars, deliberately.** An
index has no volume, but `microstructure` reads `volume_z_15`, `vwap_gap_15`
and `signed_volume_15`, and under `--complete-cases` a null column drops every
row. Volume is also the one quantity the basis cannot distort. The prices are
the venue's; only the activity columns are Coinbase's, and `venue` says so.

**Validated on write:** BRTI bars agree with Coinbase to a median +0.03 bp, and
the minute alignment was confirmed by shift test — mean |diff| 4.15 at shift 0
against 34.15 and 31.81 at ±1 minute. A bar stamped at the wrong end would
shift every window and nothing would raise.

### `venue_depth` — the order book at each minute

One row per `(venue, symbol, window, offset, source)`. `source` says which pipe
produced it, and **the sources are not interchangeable**:

| source | span | prices | sizes/depth |
|---|---|---|---|
| `backfill` (Predexon) | 2026-01-07 → 2026-10-14 | whole-cent in the tails | present |
| `candles` (Kalshi) | 2026-08-13 → now (+ history being added) | **true deci-cent** | **absent** |
| `live` / `live_ws` / `live_touch` | 2026-08-25 → now | true deci-cent | present |
| `live_ws_replay` | 2026-09-30 → 2026-10-18 | true deci-cent | present |
| `polymarket` | 2026-01-07 → now | — | cross-venue feature only |

**Predexon's defect is quantisation, not staleness-of-price-level.** Kalshi's
tick is `tapered_deci_cent` — a tenth of a cent below 10c and above 90c — and
Predexon rounds to whole cents there.

Measured 2026-10-21 on the 2026-08-25 .. 2026-10-14 overlap where all three
sources exist, as the share of quotes NOT on a whole cent:

```
             above 90c   below 10c   mid band
backfill          0%          0%         0%     <- fully quantised
candles          93%         68%         0%
live_ws          93%         90%         0%     <- ground truth
```

Candles reproduce the live book's tick behaviour almost exactly above 90c and
substantially below it; Predexon reproduces none of it. Mid-band 0% is correct
for all three — the tick genuinely is a whole cent inside [0.10, 0.90].

**The price-error comparison does NOT discriminate**, and should not be quoted
as if it did: median |error| against the live book is 2.200c for Predexon and
2.000c for candles, because both are MINUTE-BOUNDARY observations compared
against live samples taken at arbitrary instants, and these markets move ~2c in
a minute. That is timing common to both, not accuracy.

Caveat worth carrying: candles are 68% off-grid below 10c where the live book
is 90%, so candle closes land on whole cents more often than the real book does
in the deep tail — and 41% of live contracts are bought under 15c.

**Predexon's sizes are a separate and worse problem.** Measured 2026-10-21 on
110,555 rows overlapping the live book:

```
total depth ratio        1.0140      <- usable
bid size ratio (median)  1.2182      <- 22% too big
bid size correlation     0.2623      <- weak
median |price diff|      2.000 cents
```

The 2-cent price gap explains the size gap: Predexon is a **minute-stale**
snapshot, so its "best bid" sits at a different price level and "size at the
touch" is a different quantity, not the same one noisily. **Do not build
imbalance features from Predexon.**

### `venue_strikes` — the venue's own strike and settlement

`floor_strike` is the number the market actually settles against; it and
`expiration_value` come from `/historical/markets/{ticker}` or `/markets/{ticker}`.

Measured 2026-10-21 over the whole table, 85,879 markets, 2025-12-10 to now
(a full census, not the 88-window sample this entry used to quote):

```
month      markets   floor_strike   expiration_value   result
2025-12       3,123      83.2%             0%          100%
2026-01       8,142      99.2%            63.7%        100%
2026-02 on  ~8,500/mo    99.8%+          100%          100%
```

**`result` is 100% populated in every month, and `result` is the LABEL.**
`expiration_value` is the settlement VALUE: it adds magnitude, not sign, so
its absence before 2026-02 does not block `--label-source venue`. 99.1% of
markets carry both a `floor_strike` and a `result`.

The oldest weeks are the thin ones: 2025-12 is missing `floor_strike` for
16.8% of markets and `expiration_value` for all of them. The venue begins
serving `expiration_value` around 2026-01-14.

This is the **exact** strike and the **exact** label. It removes two standing
caveats outright rather than reducing them.

**Gap-filling:** re-running the script fills holes — it skips windows already
held. It was ~3 weeks short at the start until 2026-10-21 (it began
2026-01-07 while candles reached 2025-12-16); running it from 2025-12-01
closed a 3,692-market gap to zero.

### `venue_implied_vol` — the inverted strike ladder

A fit per (symbol, hour, offset) from the KXBTCD/KXETHD/KXSOLD ladders.
`source` names the writer: `live` reads the resting book, `ladder_candles` is
rebuilt from each strike's own candlesticks, and the Predexon-derived rows
were purged 2026-10-21.

The rebuild is VALIDATED against the live recorder — 10/10 fits reproduced at
a median 0.01 bp/min — but its COVERAGE is BTC-only:

```
              live era    backfilled era
BTC-USD         96.0%         100.0%
ETH-USD         95.3%          29.2%
SOL-USD         23.6%           3.8%
```

SOL is thin live too, which is the venue. **ETH is not, and the gap is a
source limit rather than a defect.** The live recorder reads the resting
book, so every strike is quoted every minute; a candle exists only for a
minute with ACTIVITY. ETH's hourly ladder is coarse relative to its
volatility — 20 wide on a ~2,326 price is 0.86% a step against ~0.19% hourly
sigma, so a step is ~4.5 sigma and only two or three rungs trade in a given
minute. Usable rungs per fit instant, where a fit needs four: **ETH median
3.5, BTC median 14**.

Widening the strike window does not help; 22, 40 and 80 rungs all produce
zero fits on the same hour. The constraint is how many strikes carry a
two-sided candle at that instant.

**So selecting `implied_vol` under `--complete-cases` does not just cap the
sample, it skews it to BTC.**

`available_time` must be the FIT INSTANT. `core/book_features` joins with a
backward `merge_asof` on it, so stamping it with the backfill's run time puts
every row after every historical decision: nothing matches and the feature is
silently NaN. Provenance goes in `source`. Repairing such rows needs
`overwrite=True` — `REVISION_KEY` includes `available_time`, so a plain
rewrite appends a revision and the read returns the broken one.

### `venue_candles` — the venue's own per-minute candle

Traded-price OHLC, bid OHLC, ask OHLC, volume, open interest. `price_low` is
the cheapest print in the minute, which bounds the maker-fill question over
every market the endpoint serves.

**A candle exists only for a minute in which that strike had activity.** A
market with no candle is usually a market nobody traded, not a fetch that
failed. The two are indistinguishable downstream, so coverage has to be
measured carefully.

**Join on `market_ticker`, never on `event_time`.** The two tables are at
different grains: `venue_strikes.event_time` is the window open, while
`venue_candles.event_time` is the candle MINUTE. Joining them on
`event_time` silently asks "did minute 0 of this window trade", which is a
liquidity question, and returns a plausible-looking coverage number that
means something else entirely. Both tables carry `market_ticker` and
`window_open`; use those.

Measured 2026-10-21 on `market_ticker`:

```
strike row but NO candles      179 of 81,568   0.22%   Jan/Feb, genuine no-trade
candles but NO strike row    3,692 of 85,081   4.34%   2025-12 (2,523), 2026-01 (1,136)
```

The second number was a `backfill_markets` gap, not a candle gap, and it is
now CLOSED (3,692 -> 0): `venue_strikes` began 2026-01-07 while candles
reached 2025-12-16. It now runs from 2025-12-10.

**Two endpoints, and the gap-fill needs the right one.** `--historical` reads
the per-ticker tier, which reaches 2025-12 to 2026-08-21. The default batch
path reaches only ~69 days back. That edge hid a real casualty: **2026-08-13,
both halves of the day, all three symbols** had no candles at all, and sits
exactly on the 69-day boundary, so a default re-run could no longer see it.
`--historical` recovered it — 285 markets, 132,345 rows. Anything older than
~69 days must be filled with `--historical` or the run will report nothing to
do and look clean.

### `available_time` means different things in the two backfills

Measured 2026-10-21, median lag past the offset minute:

```
backfill (Predexon)        0.0s      available_time == event_time
live_touch                 3.0s
live / live_ws            31.9s
candles            3,057,014.6s      available_time == when the BACKFILL RAN
```

`scripts/backfill_candles` stamps the fetch time deliberately for
`venue_candles`, where open interest is a revised series the venue restates.
The **quote** rows it writes into `venue_depth` inherit that stamp, where the
justification does not hold: a bid and ask at minute `m` were knowable at
minute `m`.

**The research path is unaffected**, and not by luck: `core/quotes.py` uses
`available_time` only for `live_touch` and judges everything else on
`quote_age_seconds`, which candle rows carry as 0.0 and 100% populated. They
are 100% usable at the 5-second bound.

**Anything that joins on `available_time` sees none of them.** The campaign
harness does exactly that, with an as-of join on
`available_time <= decision_instant`, so an experiment run through it would
silently score zero candle quotes and report that as the data.

The two backfills therefore disagree with each other, and the comment at
`core/quotes.py:120` describes the candle convention rather than Predexon's.

### Candle and live quotes are ALIGNED; the residual is movement, not error

Comparing a candle to the live book at the same offset is a fair comparison,
and that was checked rather than assumed. Shifting the candle offset against
the live one, median absolute difference:

```
shift  -2      -1       0      +1      +2
     10.0c    7.0c    2.0c    3.0c    7.0c
```

Offset 0 is the minimum and it degrades symmetrically, so there is no
off-by-one. The ~2c residual is genuine price movement between the candle's
minute boundary and the live sample 32 seconds later — the live recorders
stamp at +31.9s, `live_touch` at +3s. **So a candle is not "wrong" by 2c; it
describes a different instant.** Any claim that a minute-grid quote is
inaccurate has to separate that movement from real error.

### A minute-grid quote is not an executable price — on EITHER source

Measured 2026-10-21 against the live book, on the rows where each overlaps it:

```
                      |d|>0.5c   |d|>1.5c   |d|>3c   median
candles                 72.1%      51.2%     36.0%    1.70c
backfill (Predexon)     80.1%      60.3%     43.2%    2.30c
```

**A 1.5pp edge gate is 1.5c**, so the quote is off by more than the entire
edge the strategy trades on in 51% of candle rows and 60% of Predexon rows.

Candles are strictly better — essentially unbiased (+0.014c against
Predexon's −0.256c), on the venue's true deci-cent grid, and closer on every
cut — so they are the right quote source and Predexon can be retired. But the
improvement does not change the category. The dominant error is SHARED: both
are minute-grid, and a candle close is a 60-second boundary while a decision
happens at an instant.

So:

* **for features** (`spread`, `mid_change_1/3`) candles are a real upgrade;
  bias and grid matter and noise partly averages out over a rolling window;
* **for economics** — backtest P&L, fill simulation, realised edge — candles
  inherit the disqualification already recorded for the Predexon backfill.
  Only the live WS book is a decision-instant price.

### `venue_trades` — the public tape

One row per `(market, minute, aggressor side)` with `n_prints`, `contracts`,
`min_price`, `max_price`, `vwap`. 27.4M prints, 16,269 markets, 2026-08-25 → now.

**`taker_book_side='ask'` is the SELLING aggressor** — its prints land on the
bid (vwap − yes_bid = −0.0001 over 188,451 rows), while `'bid'` prints land on
the ask. The natural reading is backwards, and a fill simulation built on it
inverts every result while looking entirely plausible.

**The minute aggregation is too coarse for its own purpose.** A minute carries
a median 724 selling prints spanning 11 cents, so "the minute's low was below
our limit" is true ~99.9% of the time. Answering the fill question properly
needs per-print timestamps, which `backfill_trades._rows` discards.

### Everything else

| dataset | span | note |
|---|---|---|
| `venue_ladder` | 2026-08-25 → now | recorded live book; **cannot be backfilled** |
| `venue_book_events` | 2026-09-30 → now | raw WS frames, 1.5bn rows, **60-day retention** |
| `venue_implied_vol` | 2026-01-07 → now | fitted off the KXBTCD ladder; `source` names the writer (`live` / `ladder_candles`). The Predexon-derived rows were purged 2026-10-21 |
| `venue_settlements` | 2026-01-06 → now | market results, both venues |
| `pm_ladder` | 2026-08-25 → now | Polymarket book |
| `index_proxy_1s` | 2026-10-14 → now | home-made 4-exchange median; superseded by BRTI |
| `spot_trades` | 2026-10-14 → now | Coinbase tape, 60-day retention |

---

## 2. What cannot be recovered, ever

**The venue serves no historical order book.** Sizes, depth buckets and level
counts before **2026-08-25** do not exist at any price. Predexon's stale-touch
sizes were the only candidate and failed validation (§1).

That is why these **7 of 55** features cannot extend back:

```
book_flow     imbalance_change_3, depth_trend_3, imbalance_persistence
market_state  imbalance_touch, imbalance_5c, depth_ratio, book_convexity
```

The other **48 extend to 2026-01-14**, which is where the exact label starts.

**Retention clocks that are running:**

* `venue_book_events` — 60 days. The WS replay window closes ~2026-11-29.
* `spot_trades` — 60 days.
* The live `/markets/trades` tier — ~3 months; older prints answer on
  `/historical/trades`, which returned empty for 2026-08 markets when tested,
  so the fallback is **unverified**, not confirmed.

---

## 3. Enumeration was the wall, not the data

The single most important structural fact found on 2026-10-21:

> `/markets?status=settled` and `/historical/markets` only **list** about 69
> days back. The **per-ticker** endpoints serve far older markets. The limit
> was the listing, never the data.

A ticker is `series + close in New York local + minute suffix`, fully
determined by the window, so tickers are **constructed**, not enumerated:

```
KXBTC15M  first market ~2025-12-16
KXETH15M  first market ~2025-12-16
KXSOL15M  first market ~2025-12-31
```

This is what makes the long history reachable at all. It applies to market
metadata, candlesticks and trades alike.

**The two candle endpoints TILE rather than overlap:**

```
/historical/markets/{ticker}/candlesticks   2025-12 .. 2026-08   (empty after)
/markets/candlesticks (batch)               ~69 days back from today
```

---

## 4. Traps that produce silence, not errors

Every one of these was found by running something, not by reading it.

1. **The two candle endpoints disagree on field names.** Batch sends
   `yes_bid.close_dollars` and `volume_fp`; per-ticker historical sends
   `yes_bid.close` and `volume`. `rows_for_market()` returned **0 of 15**
   candles on a historical payload — it would have written nothing for 69,120
   markets without an error.
2. **The websocket and REST disagree the same way.** Stream sends
   `msg.yes_dollars_fp`; REST sends `orderbook_fp.yes_dollars`.
3. **`taker_book_side` is inverted** against the natural reading (§1).
4. **`--quote-sources` filters BOTH venues.** Excluding Predexon from Kalshi
   also discarded Polymarket's gamma backfill — 77,759 windows — and under
   `--complete-cases` every `cross_venue` row with it: 6,465 candidate windows
   became 956 and no fold could form. Fixed by `peer_quote_sources`.
5. **A Polymarket slug's trailing stamp is the window OPEN, not its close.**
6. **`no_levels` is NO-denominated on Kalshi and YES-denominated on Polymarket**
   before conversion.
7. **`seq` is global per SUBSCRIPTION, not per market**, so one gap condemns
   every book on the connection.
8. **`delta_fp` is a signed change**, so levels driven to zero accumulate float
   residue and keep a dead price at the touch.
9. **Minute-grid sources are aged FORWARD** (`latency − raw`), so a book
   sampled more than `decision_latency_seconds` after its own minute goes
   negative and is dropped. A stream row stamped 0s old yields 6/6 `book_flow`
   features; the same row at 7s yields 0/6.

---

## 5. The pipeline, end to end

```
Coinbase REST/WS ─┐
Kalshi REST/WS   ─┼─> spool / recorders ─> ResearchStore (Parquet, partitioned)
Polymarket       ─┘                              │
CF Benchmarks ───────────────────────────────────┘
                                                 │
                        core/windows.py ── the 15-minute grid, strike, label
                                                 │
                        core/vol.py ── HAR + intraday seasonality -> sigma_remaining
                                                 │
                        core/baseline.py ── F(x/sigma), THE NULL
                                                 │
                        core/quotes.py ── attach the market quote (priority ladder)
                                                 │
                        core/features.py ── 9 groups + the clock control
                                                 │
                        core/model.py ── LightGBM on an init_score
                                                 │
                        core/backtest.py ── walk-forward, 7-day folds, 1-day embargo
                                                 │
                        core/metrics.py ── 24 gates ─> core/promotion.py
```

### The quote priority ladder

```
QUOTE_SOURCE_PRIORITY = live_touch, live_ws, live, live_ws_replay, candles, backfill
```

An **unlisted source ranks LAST**, below the backfill — a new producer that is
not named here is quietly distrusted rather than quietly trusted.

### The init score

`init_score_source='market'` with `init_coefficient=1.0` means the prediction is

```
market_logit x 1.0  +  residual_scale x (tree correction)
```

So the **anchor is the market's own probability** — which is BRTI-correct by
construction — and only the **correction** rides on our price-derived features.
That bounds how much the Coinbase basis can hurt: it degrades the correction,
not the anchor.

### Point-in-time discipline

* A decision at offset `m` sees the close of bar `m-1` and nothing after.
* Cross-validation splits on the **window**, never the row.
* The embargo is a **day**, not fifteen minutes, because `log_rv_1440` reaches
  back that far.
* `available_time` is when a row became knowable, and a row whose
  `available_time` precedes its `event_time` is refused at write time.
* Standard errors come from **fold dispersion**, not a breadth formula.

---

## 6. The target

```
strike      = mean over [t0 - 1min, t0)        <- 60s of BRTI, per the venue's rules
settlement  = mean over [t1 - 1min, t1)
outcome     = settlement >= strike             <- `strike_type` is greater_or_equal
```

Today `core/windows.py` builds both from Coinbase OHLC means. Measured against
56,284 of the venue's own settlements, that proxy agrees **96.98%** of the
time, and the disagreement lives entirely in near-ties:

```
move from strike   windows   disagree
     <1bp             2806     34.85%
     1-2bp            2826     13.91%
     2-5bp            7860      3.18%
    10-25bp          18147      0.05%
     >25bp           13472      0.01%
```

**This is ALREADY FIXED in the deployed configuration, and the fix is not
`expiration_value`.** `--label-source venue` swaps `outcome` for
`venue_outcome`, which is the venue's own `result`; measured 2026-10-21 it is
present on 100% of scored rows. The 3.12% above is the Coinbase PROXY's error
against the venue label, not an error the training set still carries.

What `expiration_value` adds is the settlement **value** — the number, not the
yes/no. That gives an exact next-window strike and an exact distance from
strike, which the proxy only approximates. It does not remove label noise that
has already been removed.

---

## 7. The intended end state

| quantity | today | after |
|---|---|---|
| strike | Coinbase OHLC mean, ~0.55bp median error | venue `floor_strike`, exact |
| label | **already venue-sourced** (`--label-source venue`) | unchanged; `expiration_value` adds the settlement VALUE |
| price series | Coinbase | CF Benchmarks index |
| quotes | Predexon whole-cent in tails | venue candles at true tick |
| span (Kalshi-only) | 2026-08-13 → now, 6,465 windows | 2026-01-14 → now |
| windows actually usable | 6,465 | **~16,600 with `implied_vol`, ~27,000 without** |
| features available | 55 | 48 over the long span, 55 from 2026-08-25 |
| folds | 5 (`min_folds` FAILS) | ~20 with `implied_vol`, ~35 without |

**`implied_vol` coverage is the binding constraint over the long span, not the
quote source.** Measured 2026-10-21, the fraction of (symbol, window) with an
IV fit in the same hour:

```
2026-01  22%   2026-04  28%   2026-07  28%
2026-02  25%   2026-05  19%   2026-08  43%
2026-03  21%   2026-06  13%   2026-09  68%   2026-10  61%
```

Under `--complete-cases` a NaN in any selected feature drops the whole row, so
keeping `implied_vol` caps the recoverable sample at roughly a quarter of the
pre-August windows. Whether its 5 features are worth ~10,000 windows is an
ablation, not a judgement.

**That coverage is an artifact of the recorder, and the rebuild is VALIDATED.**
The live recorder only ever saw markets with `status=open`; the venue still
serves every strike of every past hour. Rebuilding the fit from each strike's
own candlesticks reproduces a live-recorded fit **10/10 at a median 0.01
bp/min** (mean −0.03, same `n_strikes` in 9 of 10) — the same number by
another route, not an approximation of it.

Three endpoints **tile**, and missing the third is what made this look
unverifiable for a day:

```
/historical/markets/{t}/candlesticks   2025-12  ->  2026-08-21
/markets/{t}/candlesticks              nothing, ever, for a 1-hour strike
/markets/candlesticks   (PLURAL)       ~69 days back  ->  now
live recorder                          2026-08-26  ->  now
```

The first two stop **five days before** the live recorder starts. With only
those wired there is no era holding both a candle fit and a live fit, which
reads as "no overlap, cannot validate" when the cause is a missing endpoint.
The plural path is keyed on `market_tickers`, nests one level deeper
(`markets[].candlesticks[]`), and names the ticker `ticker` or `market_ticker`
by tier — so it cannot share the per-ticker parse. Because it takes a list, a
response can carry markets you did not ask for; returning the first block
regardless fits one strike's quotes onto another strike's price, a wrong
ladder that raises nothing and fits fine. Matched on ticker, with a test.

**The eras are not equally dense, and the thinness is REAL rather than an
endpoint artifact.** A candle exists only for a minute in which that strike
had activity, so a quiet ladder yields few candles, few strikes quoting at a
given fit offset, thinner `n_strikes` and lower R2.

The tempting explanation was that the older tier is lossy. It is not. In the
**Aug 13-21 overlap**, where both the per-ticker historical endpoint and the
plural batch endpoint serve the same market, the two return the **same count
every time** — 16/16, 12/12, 39/39, 40/40, 7/7 across three strikes on each
of four hours. The earlier "9-14 against 60" reading compared *different
dates*, not the same market, and measured the venue getting busier over 2026
rather than one endpoint dropping rows.

So old-era coverage cannot be raised by querying differently. It is what the
market was.

**A second source-comparability trap sits in `cross_venue`.** Polymarket rows
exist at 96-99% of decision offsets across the whole span, but the share
passing a 5-second freshness gate runs 91% in January against 28% in
September — because backfilled rows have their age counted MINUTE-BACKWARD
while live recorders count FORWARD (`core/quotes.MINUTE_BACKWARD_SOURCES`).
The backfill can report a small age by construction. Extending the span
therefore makes old rows pass the peer gate far more often than recent ones,
which is a selection artifact and not better data.

### The constraint that governs any switch

Live **already fetches `floor_strike`** and uses it for settlement, but
deliberately **not** for features. An earlier version overwrote `strike` and
recomputed only `z_score`, leaving `abs_z_score`, the three excursion-z
columns, `path_efficiency`, `displacement_vs_elapsed`, `touched_opposite` and
`peer_displacement` on the bar basis — so `abs_z_score == |z_score|`, which
holds in every training row, was violated at inference. A tree ensemble does
not degrade gracefully off its training manifold.

**So the strike must be changed in TRAINING first, and live switched to match.**
Not the other way round, and never piecemeal.

**The same rule binds the price series, and this is MEASURED, not argued.**
Tested 2026-10-21 on 16,344 windows graded against the venue's own settlement,
fitting one calibration scale per variant so neither gets more freedom:

```
Coinbase price / Coinbase strike   log loss 0.482983
Coinbase price / venue strike      log loss 0.488026   WORSE
window-clustered gain -0.0063 nats, 90% CI -0.0118 .. -0.0018
```

Adopting the venue's exact strike while the price stays on Coinbase makes the
barrier null **significantly worse**, because it injects the cross-index basis
rather than removing it. The strike and the price series move together, or
neither moves. The end-state cell — BRTI price with the venue strike — is not
yet testable; it needs index bars over the same windows.

---

## 7b. What `--complete-cases` costs, per feature group

**Measured 2026-10-21, AFTER the candle backfill landed.** An earlier version of
this section carried numbers from evaluate runs started before it, over a span
beginning 2025-10-01 — months in which there was no Kalshi book at all, so the
book features were NaN for reasons unrelated to sizes. Those numbers said the
book groups cost 51% and `implied_vol` 15%. **Both were wrong, and backwards.**

Window ceilings at the four decision offsets, Kalshi-only quote sources:

```
feature set kept                      windows   vs all   earliest
everything (quote+sizes+peer+IV)       20,839       —    2026-01-07
drop implied_vol                       58,569    +181%   2026-01-07
drop book (sizes)                      23,198     +11%   2026-01-07
drop book AND implied_vol              74,648    +258%   2026-01-07
drop book, IV and cross_venue          84,795    +307%   2025-12-15
```

Per month, what each requirement admits:

```
month     windows    +quote     +sizes    +peer5s +impl_vol
2025-12     2,555     2,555          0          0         0
2026-01     8,126     8,056      6,295      6,563     1,424
2026-03     8,772     8,772      7,803      8,688     1,664
2026-05     8,743     8,743      2,577      8,258     1,511
2026-07     8,793     8,775      7,932      8,169     2,292
2026-09     8,544     8,544      8,544      7,419     5,792
2026-10     5,868     5,868      5,862      5,269     3,552
```

### What this means

* **`implied_vol` is the binding constraint by a wide margin** — it alone caps
  the set at 20,839 of 84,795 quoted windows. It is also RECOVERABLE: the
  KXBTCD strike ladder is constructible per hour (75 strikes listed for a
  January 2026 event), so the 13-28% coverage is an artifact of the live
  recorder only seeing `status=open` markets, not a limit of the venue.
  **Confirmed 2026-10-21** — the rebuild reproduces live fits to 0.01 bp/min,
  so this is a recovery rather than a substitute, and the keep-or-drop
  question is no longer "is it worth ~10,000 windows".
* **The book-shape features cost only 11%.** Whether those 7 unrecoverable
  features earn their place is therefore an 11% decision, not a 51% one.
* **Keeping every feature exactly as deployed, the candle backfill took the
  Kalshi-only sample from 6,465 to 20,839 windows — about 3.2x**, with no
  feature changes and no trade-off. Folds go from 5 (`min_folds` FAILED) to
  roughly 27.

### The consequence, which is NOT symmetric

`--complete-cases` binds at TRAINING time. Live scoring does not drop rows — it
substitutes LightGBM's learned defaults for a NaN, which is the `book_flow`
failure in §4. So the model is fitted on the rows where everything was present
and then scored on rows where things are missing.

### What is NOT yet known

Whether the book-shape groups earn their 11%. Both available tests are
unusable: the long span computes those features from Predexon's minute-stale
touch (ratio 1.22, correlation 0.26) for most of its sample, and the live-book
era yields a 369-row inner holdout against a 500-row minimum. Four attempts are
recorded in `research/campaign/experiments/`. **It needs calendar time since
2026-08-25, not another configuration** — and at 11% it is not urgent.

## 8. Rate limits and cost

```
usage tier   advanced (granted 2026-10-21, free, permanent, Advanced is the ceiling)
read         300 tokens/s, bucket 900   -> 30 req/s at the default 10-token cost
write        300 tokens/s, bucket 900   -> 30 req/s
/cfbenchmarks/*   50 tokens             ->  6 req/s
```

**Backfills must be SEQUENCED, not parallel.** They share one bucket, and the
429 retry holds its semaphore slot while it sleeps, so once throttling starts
throughput collapses — measured on the tape backfill at **~11 markets/min at
concurrency 4 against ~96/min at concurrency 2**, about 9x.

---

## 9. How to rebuild from nothing

```bash
cd backend/trader
python -m scripts.scrape --backfill-days 1825      # Coinbase bars
python -m scripts.scrape --fill-gaps               # see the 1-in-301 note in CLAUDE.md
python -m scripts.sync_store

python -m scripts.backfill_markets --start 2025-12-16   # exact strike + label
python -m scripts.backfill_candles --historical \
       --start 2025-12-16 --end 2026-08-13              # true-tick quotes, old
python -m scripts.backfill_candles                      # true-tick quotes, recent
python -m scripts.backfill_index --start 2026-01-07     # CF Benchmarks index
python -m scripts.backfill_trades --concurrency 2       # the tape
python -m scripts.build_depth                           # one table from every book source

python -m scripts.baseline                              # read the NULL first
python -m scripts.evaluate                              # walk-forward + gates
```
