# Research campaign ledger — opened 2026-10-17

Rules: every experiment pre-registered before results are inspected. Negative
results recorded. No re-running a disproven hypothesis without new evidence.
Budget: 12 main experiments. Compute budget stated per experiment.

## Diagnostics (not experiments — no thresholds tuned)

| id | question | status |
|---|---|---|
| D1 | Is live +$17.53 distinguishable from noise? | CLOSED — no, t=0.17 |
| D2 | Do >90c favourites carry the P&L? | CLOSED — no, 23.8%, t=0.27 |
| D3 | How does quote age distort model_minus_market? | CLOSED — monotonic, does not rescue backfill |
| D4 | Cent-quantised backfill false edge? | CLOSED — not quantisation; noise + 1-min misalignment |
| D5 | Backfill legitimate as PIT feature / executable price? | CLOSED — NO as price, yes as m-1 feature |
| D6 | Features consistent across eras? | CLOSED — NO; 14/49 depend on a book with 3 provenances |

## D1 — RESULT: the live P&L is noise. CLOSED.

1,289 settlements, 6,692 contracts, $3,138 deployed, $83.83 fees, **+$17.53**.
- per contract **+0.262pp** against a pre-registered +1.5pp
- 936 independent 15-min windows, **cluster-robust t = 0.171**
- block bootstrap 95% CI **[-$181.36, +$219.84]**
- **excluding the single largest winner the total is -$0.85**; top-5 |pnl| = $48.50 vs total $17.53
- months: Aug +8.19 (t=0.11), Sep -8.22 (t=-0.13), Oct +17.56 (t=0.52)

Verdict: no live edge is demonstrated. The live record cannot confirm or refute
anything at this sample size and must not be used to select a model.

## D2 — RESULT: >90c favourites do NOT carry the P&L. Hypothesis rejected.

| price | n | contracts | pnl$ | pp/ct | win% (ct-wtd) | breakeven% | excess_pp | t |
|---|---|---|---|---|---|---|---|---|
| [0.00,0.10) | 43 | 379 | -9.34 | -2.465 | 5.01 | 7.48 | -2.465 | -0.78 |
| [0.10,0.30) | 284 | 1,873 | 15.94 | 0.851 | 22.21 | 21.36 | 0.851 | 0.27 |
| [0.30,0.50) | 304 | 1,496 | -0.15 | -0.010 | 40.51 | 40.52 | -0.010 | -0.00 |
| [0.50,0.70) | 283 | 1,207 | -12.40 | -1.027 | 59.32 | 60.35 | -1.027 | -0.27 |
| [0.70,0.80) | 161 | 661 | -22.05 | -3.335 | 73.24 | 76.57 | -3.335 | -0.83 |
| [0.80,0.90) | 128 | 586 | 41.37 | **7.065** | 92.48 | 85.42 | **7.065** | **2.79** |
| [0.90,0.95) | 83 | 475 | 3.58 | 0.754 | 93.05 | 92.30 | 0.754 | 0.23 |
| [0.95,1.01) | 3 | 15 | 0.58 | 3.887 | 100.00 | 96.11 | 3.887 | 11.34 |

**>=90c is 23.8% of P&L on t=0.27** — not the driver. The only bucket with a
t-stat is [0.80,0.90) at +7.07pp (n=128, t=2.79), which is 1 of 8 buckets tested
and does not survive a Bonferroni correction (p~0.005*8=0.04 marginal, n small).
Realized outcomes beat break-even essentially nowhere.

Repro: `python research/campaign/diagnostics/d1_d2_live_pnl.py <vs.tsv>` where
vs.tsv is `select ticker,market_result,yes_contracts,no_contracts,yes_cost,no_cost,revenue,fee_cost,pnl,settled_time from venue_settlements order by settled_time`.

## D3/D4/D5 — RESULT: the Predexon backfill is not an executable price, and its
## offset stamp is one minute late. CAMPAIGN-DEFINING. CLOSED.

Paired test: 143,134 observations where the live stream recorder and the Predexon
backfill describe the SAME (symbol, window, offset), 2026-08-25 -> 2026-09-16.

**D4 — quantisation is NOT the problem, and the apparent bias was an artifact.**
Conditioning on the live mid showed a +6.7pp/-5.8pp mean-reverting gap. Conditioning
on the BACKFILL mid instead collapses it to +0.44/-0.00/+0.25/-0.24/-0.35/+0.60pp.
The signs do not agree, so the D4 pattern was regression to the mean from independent
noise, not bias. Overall gap -0.056pp (t=-1.70). The backfill is unbiased.

**It is, however, enormously noisy, and that is worse.**

**D4b — the backfill mid is a far worse forecast of settlement than the live mid.**

| price | log loss | brier | calib err |
|---|---|---|---|
| live mid (stream) | **0.454078** | 0.150825 | 0.00654 |
| backfill mid (Predexon) | 0.496603 | 0.166357 | 0.00642 |

Gap **+0.042525 nats**, paired bootstrap 95% CI [+0.0414, +0.0470] over 1,965 windows.
That is **~54x the deployed model's entire claimed log_loss_skill of +0.00079.**

**D3 — quote age degrades it monotonically but does NOT rescue it.**

| backfill age (s) | n | ll_backfill | ll_live | gap |
|---|---|---|---|---|
| [0,5) | 109,638 | 0.502554 | 0.478452 | +0.024103 |
| [5,15) | 7,585 | 0.404813 | 0.360460 | +0.044353 |
| [15,30) | 4,109 | 0.453185 | 0.403692 | +0.049493 |
| [30,60) | 4,528 | 0.475072 | 0.414501 | +0.060571 |
| [60,120) | 6,866 | 0.492099 | 0.395663 | +0.096436 |
| [120,300) | 8,716 | 0.517638 | 0.353194 | +0.164444 |
| [300,inf) | 1,692 | 0.595390 | 0.279357 | +0.316034 |

Even rows the backfill calls FRESH (age<=5s, n=109,652 / 1,963 windows) are
**+0.024107 nats worse, CI [+0.0216, +0.0258]**. No age cut recovers the live book.

**D5 — the backfill's `offset_minutes` is stamped ~1 minute late.** Per-row argmin
over the live book across the whole window (n=81,263):

| matched live minute | share |
|---|---|
| offset-1 | **30.60%** |
| offset+0 | 20.88% |
| offset-2 | 10.37% |
| offset+1 | 5.37% |

median relative offset **-1.0**, mean -1.338. Mean |error| minimises at live@t-1
(5.939pp) against t+0 (7.498pp); restricted to age<=5s rows, 4.703pp at t-1 against
5.843pp at t+0. **`quote_age_seconds` does not reveal this** — it reads <=5s on
exactly the rows that are a minute stale.

**Consequences, binding on the rest of the campaign:**
1. Backfill is **NOT an executable price**. Pricing off the backfill ask while
   treating the live mid as fair makes **34.34%** of rows clear a 1.5pp gate; traded
   at the true live ask those rows realise **-0.705pp per contract**.
2. Backfill is **not a valid market benchmark**. Comparing a forecast built at
   offset m against a price from offset m-1 hands the model a one-minute information
   advantage, which inflates `model_minus_market` in the model's favour.
3. The direction is BACKWARD (stale), not forward, so this is **not label leakage** —
   a shifted backfill quote is still legitimately point-in-time *as an m-1 feature*.
4. **The only trustworthy market price in this repository is the live-recorded book**
   (`source in ('live_ws','live')` for Kalshi), 2026-08-25 -> 2026-09-30.

Repro:
```
python research/campaign/diagnostics/d4_quantisation.py
python research/campaign/diagnostics/d4b_which_price_is_real.py
python research/campaign/diagnostics/d3_quote_age.py
python research/campaign/diagnostics/d5_backfill_alignment.py
```

## D5b — RESULT: `live_touch` cannot be validated in the era that matters. BLOCKER.

`live_touch` is the trading loop's own quote at the decision instant, at offsets
3/6/9/12, spanning 2026-08-27 -> 2026-10-17 (14,178 symbol-windows). It would more
than double the trustworthy sample if it were sound.

Paired against the stream ladder, n=43,992:
- **unbiased**: mean (touch - stream) mid gap **-0.0257pp**
- but noisy: mean |gap| **5.2651pp**, median 3.0000pp; only 22.8% within 1pp
- alignment is roughly correct: t+0 5.265pp vs t-1 4.892pp (near tie, unlike the
  backfill's clear 5.939 vs 7.498 win for t-1)
- **+0.018424 nats worse** than the stream at predicting settlement

**The blocker: every one of those 43,992 paired rows carries `transport = null`,
i.e. all of them predate the 2026-09-16 fix** for live_touch pricing off `/markets`'
cached touch rather than the live book. The stream ladder stops 2026-09-17, so the
overlap window and the contaminated era are the same window. **The post-fix
live_touch (2026-09-16 -> 2026-10-17) has no independent source to validate against.**

Per the three-attempts rule this is documented and the affected data excluded from
the benchmark role. live_touch may be used as a FEATURE and for execution realism
(it is what the loop actually saw), never as the fair-value benchmark.

## Consequence: the campaign's evaluation universe

| role | source | span | sample |
|---|---|---|---|
| **benchmark + executable price** | stream ladder (`live`,`live_ws`) | 2026-08-25 -> 09-16, **23 days** | **6,488 symbol-windows** (2,163 windows x 3 symbols), 15 offsets |
| feature only (noisy, shift to m-1) | `backfill` | 2026-01 -> 2026-10-14 | ~759k rows |
| execution realism only | `live_touch` | 2026-08-27 -> 10-17 | 14,178 symbol-windows, 4 offsets |

This is a **power-limited** campaign and will be reported as one. CLAUDE.md's own
power table needs ~456 trades to confirm a +2.6pp edge and ~5,200 for ~1pp; 2,163
independent windows yielding a fraction of that in trades can only detect a large
effect. A null result here is "underpowered", not "no edge", and will be labelled so.

## E1 / E2 — RESULT: the Kalshi mid is already calibrated. Both FAIL.

Dataset: stream ladder x venue settlement, 2026-08-25..09-16.
**84,513 rows, 2,161 windows, 6,482 symbol-windows.** Base rate 0.4947.
Dev 54,509 / val 30,004. 5 purged window folds, 1-day embargo.

Raw market mid (R0, the thing to beat): log loss **0.451929**, calibration error
**0.00776**, max deviation 0.0147.

| candidate | OOF log loss | market | model_minus_market | 95% CI | p(<=0) | verdict |
|---|---:|---:|---:|---|---:|---|
| E1 isotonic | 0.457621 | 0.455700 | **-0.001740** | [-0.00285, -0.00068] | 0.9997 | FAIL |
| E1b Platt | 0.456237 | 0.455700 | -0.000379 | [-0.00135, +0.00052] | 0.7822 | FAIL |
| E2 isotonic by offset | 0.463782 | 0.455700 | **-0.007650** | [-0.01070, -0.00494] | 1.0000 | FAIL |

Every recalibration makes the price WORSE, and calibration error rises from the
market's 0.00461 to 0.0073-0.0115. E2's much larger loss is overfitting: fourteen
separate curves on 2,161 windows.

The market's reliability table does show a visible pattern — mid prices overpriced
by ~1.1-1.2pp, prices above 0.72 underpriced by ~0.8-1.1pp — but it is **not stable
enough across folds to exploit**: fitting it out-of-fold loses money against simply
believing the quote.

**Consequence: there is no free correction. Any edge must come from information the
price does not contain.** Every later candidate is measured against the RAW mid.

NOT RETRIED. A different recalibrator is the same hypothesis.

## Correction to the campaign's own method (recorded, not hidden)

The first dataset design precomputed a feature panel and cached it. **That was
wrong and the artifact was discarded.** In this repository features are built
INSIDE the fold by `core.dataset.fit_fold`, because three things are fitted — the
volatility model, the intraday seasonality factor, and the baseline's scale/tail —
and a panel computed once over the whole span would leak all three into every test
block. `Dataset.build().windows` carries no feature columns at all, which is what
exposed the mistake: the first leakage test compared **0 features over 0 rows** and
passed vacuously.

Two lessons kept:
1. A passing leakage test proves nothing unless it reports how many comparisons it
   made. The test now asserts on a non-zero comparison count.
2. The expensive part of `Dataset.build` is the depth attachment (0.5s without it,
   >20 minutes with 2.4M depth rows). Since the campaign joins the clean stream
   price itself, depth is only needed for the book-feature candidates (E7, E8).

## Causality tests — status

| test | status |
|---|---|
| decision instant inside its window | pass |
| features vary with the offset | pending panel |
| **market price is not from the future** (|p-0.5| must rise as the window resolves) | **PASS** |
| label is the venue's, not the proxy | pending panel |
| replay vs truncated rebuild | **rewritten — the first version was vacuous** |

## D6 — RESULT: the deployed model's features are NOT consistently populated. CLOSED.

The artifact declares 49 features in 8 groups. **14 of them depend on the venue's
book**: `cross_venue` (4), `market_state` (5), `implied_vol` (5).

Book-source availability by month (distinct symbol-windows):

| month | kalshi stream | kalshi backfill | kalshi live_touch | polymarket |
|---|---:|---:|---:|---:|
| 2026-06 | **0** | 3,618 | 0 | 8,603 |
| 2026-07 | **0** | 8,046 | 0 | 8,662 |
| 2026-08 | 1,971 | 7,750 | 1,089 | 7,140 |
| 2026-09 | 4,518 | 8,510 | 8,538 | 8,640 |
| 2026-10 | **0** | 3,725 | 4,557 | 4,628 |

**A trustworthy Kalshi book exists in August and September only.** The deployed
artifact was trained on data through 2026-10-14, so its `market_state` features came
from the backfill (noisy, stamped a minute late — see D5) or from `live_touch`
(unvalidatable post-fix — see D5b) across most of its training span. Three different
provenances under one set of column names.

**`implied_vol` is worse.** Fits per month by symbol:

| month | BTC | ETH | SOL |
|---|---:|---:|---:|
| 2026-07 | 2,448 | 1,257 | 1,129 |
| 2026-08 | 8,910 | 2,094 | 1,210 |
| 2026-09 | 35,590 | 31,606 | 4,261 |
| 2026-10 | 18,133 | 14,268 | **28** |

SOL's implied vol is **effectively absent in October** (28 fits against BTC's 18,133).
And the share of fits clearing the live `r2 >= 0.90` gate jumps discontinuously:

| month | BTC | ETH | SOL |
|---|---:|---:|---:|
| 2026-07 | 0.869 | 0.839 | 0.847 |
| 2026-08 | 0.981 | 0.924 | 0.850 |
| 2026-09 | **1.000** | **1.000** | **1.000** |
| 2026-10 | **1.000** | **1.000** | **1.000** |

A quality metric that goes from 0.85 to exactly 1.000 and stays there is a change in
the data-generating process, not an improvement in the market. Either the collector
changed or the gate is being applied upstream. **Feature distributions are not
comparable across the training span**, which is one more reason this campaign
evaluates only on the Aug-Sep stream-ladder era.

Repro: `python research/campaign/diagnostics/d6_feature_population.py`

# ============================================================================
# D7 — CORRECTION. D3/D4/D5 WERE WRONG IN THEIR MAIN CONCLUSION.
# E1, E2, E3 AND BATCH 1 ARE INVALIDATED AND MUST BE RERUN.
# ============================================================================

Prompted by the user's challenge: "Predexon IS the true market vs something like
Coinbase". Correct, and it forced the right measurement.

**What I failed to check: WHEN each source's snapshot was actually taken.**

| source | event_time - stamp | available_time - stamp | own quote_age |
|---|---:|---:|---:|
| backfill | 0s | **+0s** | 0.33s |
| live / live_ws | 0s | **+32s** | 32.6s |
| live_touch | 0s | +3s | NaN |

**The live ladder row labelled offset m describes the book at m + 32 seconds.**
`record_ladder` samples at ~+32s past the minute. The backfill is stamped at the
minute exactly.

So for a decision taken at offset m (+3s latency):
- `backfill@m` **was available** (available_time = m)
- `live@(m-1)` **was available** (available_time = m - 28s)
- `live@m` **was NOT available** — it is **32 seconds of lookahead**

### The decisive test (n=255,773 over 1,965 windows)

| price | log loss |
|---|---:|
| backfill @ m (available at m) | 0.484792 |
| live @ m-1 (available at m-28s) | 0.486307 |
| live @ m (available at **m+32s** — lookahead) | 0.439771 |

Paired bootstrap over windows:

| comparison | diff | 95% CI |
|---|---:|---|
| **backfill@m minus live@m-1** (both available) | **-0.000040** | **[-0.00286, +0.00276]** |
| backfill@m minus live@m (live = lookahead) | +0.046777 | [+0.04377, +0.04986] |
| live@m-1 minus live@m (pure 60s of freshness) | +0.046817 | [+0.04557, +0.04807] |

**The backfill's entire apparent handicap is TIMING.** Its gap to live@m (+0.0468) is
statistically indistinguishable from the gap a pure 60-second freshness difference
produces (+0.0468), and against the genuinely contemporaneous price it is
**indistinguishable from zero** (-0.000040, CI straddling 0).

### What is overturned

- ~~"the backfill is not an executable price"~~ — **WRONG.** It is the correct
  point-in-time price for a decision at offset m.
- ~~"the trustworthy universe is 2,161 windows over 23 days"~~ — **WRONG.** The
  backfill is usable 2026-01 -> 2026-10, ~28,000 symbol-windows.
- ~~"model_minus_market measured against the backfill is inflated"~~ — **WRONG, and
  backwards.** Measuring against `live@m` makes the benchmark unfairly HARD by giving
  the market 32 seconds the model never had.

### What survives

- D4's symmetry control: the backfill is **unbiased** (±0.5pp conditioning on its own
  mid). Still valid.
- D5's argmin "matches offset-1": **reinterpreted** — a 60-second grid resolving a
  ~32-second offset, not a stamping error.
- D3's quote-age gradient: high-age backfill rows are genuinely staler, but the
  magnitudes were measured against a lookahead reference and are inflated.
- D1, D2, D5b, D6: unaffected.

### The bug in my own harness

`harness.stream_prices()` joined `live@m` as both benchmark and executable price.
**Every experiment run so far used a market price 32 seconds in the future.** That is
why E1, E2 and E3 lost to "the market" so decisively — the market had 32 extra
seconds of information in a 15-minute window.

Per RESEARCH_PROTOCOL.md section 1: the bug is fixed, the dependent experiments are
**invalidated**, and **the same pre-registered specifications are rerun**. No new
specifications are invented at this point.

| experiment | status |
|---|---|
| E1 / E1b / E2 (market recalibration) | **INVALIDATED — rerun** |
| E3 (baseline only) | **INVALIDATED — rerun** |
| batch 1 (E4, E5, E5c, E6, E6b, E9, E10) | **INVALIDATED — killed mid-run, rerun** |
| negative controls | **INVALIDATED — rerun** |
| D1, D2, D5b, D6 | unaffected |

Repro: `python research/campaign/diagnostics/d7_timing_correction.py`

## Operational finding — why the evaluation universe ends 2026-10-14

Prompted by the user asking why data stops on the 14th when today is the 17th.

**The pipeline is updating correctly.** Every live collector is current:

| dataset | last event_time |
|---|---|
| minute_bars | 2026-10-17 01:58 |
| venue_depth | 2026-10-17 02:12 |
| venue_settlements | 2026-10-17 01:45 |
| venue_implied_vol | 2026-10-17 02:11 |
| pm_ladder | 2026-10-17 02:10 |
| venue_book_events | 2026-10-17 00:59 (1.18B rows) |

**The one stale producer is the `backfill` SOURCE inside venue_depth**, which is a
manual batch job (`scripts/build_depth.py` over the Predexon archive), last run
2026-10-14:

| venue | source | last window |
|---|---|---|
| kalshi | **backfill** | **2026-10-14 05:45** |
| kalshi | live_touch | 2026-10-17 02:00 |
| kalshi | live / live_ws | **2026-09-16 20:15** |
| polymarket | backfill | 2026-10-14 00:45 |
| polymarket | live | 2026-10-17 02:00 |

Since D7 established the backfill as the primary point-in-time price, the evaluation
universe ends where the backfill ends. Beyond 2026-10-14 only `live_touch` exists,
at four offsets, from the source D5b could not validate. **Running
`python -m scripts.build_depth` would extend the universe by three days**; not done
here because the campaign must not write to production data stores.

**Second finding, and it is a real operational bug: the Kalshi ladder summariser
stopped on 2026-09-16 and nobody noticed.** `venue_ladder` has been dead for a month
while `venue_book_events` — the raw frames it is built from — is current to today
with 1.18 billion rows. The full contemporaneous ladder for 2026-09-16 -> 2026-10-17
is therefore **recoverable by replaying the raw stream**, which is what makes the
23-day "trustworthy window" an artifact of a dead summariser rather than of missing
data. Not needed for this campaign (D7 made the backfill the benchmark), but it
should be fixed, and it explains the gap.

# ============================================================================
# D8 / D9 — THE BASELINE "BEATING THE MARKET" IS QUOTE CARRY-FORWARD.
# BATCH 1 INVALIDATED AGAIN. THIRD CORRECTION.
# ============================================================================

E3 (the barrier baseline ALONE, no ML) scored `model_minus_market` **+0.007963**,
CI [+0.00623, +0.00966], p=0.0000 over 7,932 windows, and made money at t=5.22.
RESEARCH_PROTOCOL.md section 5: an unusually strong result is a reason to audit the
pipeline. CLAUDE.md already records this exact failure — "a fresh forecast against a
stale price manufactures edge ... the baseline 'beating' the market by 0.246 nats at
t = 8.4, which was entirely the carry-forward."

## D8 — the edge is entirely in stale quotes

Apparent baseline edge by the SOURCE's own `quote_age_seconds` (time since the last
recorded book change):

| book age | n | windows | model_minus_market | 95% CI |
|---|---:|---:|---:|---|
| <1s | 215,289 | 7,715 | **-0.006185** | [-0.00772, -0.00468] |
| 1-5s | 13,816 | 4,228 | **-0.005807** | [-0.01011, -0.00149] |
| 5-15s | 17,691 | 4,111 | +0.005683 | [+0.00062, +0.01070] |
| 15-30s | 11,957 | 3,375 | +0.011859 | [+0.00516, +0.01874] |
| 30-60s | 11,784 | 3,187 | +0.025879 | [+0.01805, +0.03375] |
| 60-120s | 8,289 | 3,038 | +0.069291 | [+0.05874, +0.07985] |
| >120s | 17,429 | 2,801 | **+0.172620** | [+0.15841, +0.18662] |

**On genuinely fresh quotes (age <= 5s, n=229,105 / 7,737 windows) the baseline
LOSES to the market: -0.006208, CI [-0.00776, -0.00470].** On stale rows (>30s) it
"wins" by +0.096081. Monotonic across seven bands — not noise.

By source: backfill +0.008606, live -0.000085, live_touch -0.005606. **Backfill-only.**

By offset the edge rises monotonically (-0.0026 at offset 1 to +0.0382 at offset 14),
because later in a window the feed is more likely to have gone quiet while the
baseline's spot input stays fresh.

## D9 — and a stale backfill quote is a FEED HOLE, not a quiet book

If a large book age meant a genuinely static market, the live book would also be
still. It is not. n=273,819 paired rows:

| backfill book age | n | mean abs(backfill - live) | **mean live move per minute** |
|---|---:|---:|---:|
| <1s | 198,117 | 5.861pp | **8.374pp** |
| 5-15s | 14,582 | 6.720pp | 8.551pp |
| 30-60s | 8,748 | 10.185pp | 8.838pp |
| 60-120s | 13,376 | 13.466pp | 9.179pp |
| >120s | 19,878 | **19.582pp** | **9.235pp** |

**The live book moves just as much (~8.4-9.2pp/min) no matter what the backfill's
book age says**, and 72-77% of rows show the live book moving >2pp/min in EVERY band
including >120s. Backfill-vs-live disagreement meanwhile grows 5.86pp -> 19.58pp.

**When Predexon reports "no change for 120 seconds", the Kalshi book was moving
normally. The price is missing data carried forward, not a resting quote.** It is not
tradeable: you cannot buy at a price that does not exist.

## Consequence

`available_time` is necessary but **not sufficient**. The as-of join correctly
answers "could we have known this row", but the row's PRICE can still be a frozen
carry-forward. A second filter on the source's own `quote_age_seconds` is required.

Cap set at **5 seconds**, which is where the baseline's apparent edge crosses from
negative (correct) to positive (artifact). Rows above it are retained in the dataset
and FLAGGED, never silently dropped, and both filtered and unfiltered counts are
reported.

| experiment | status |
|---|---|
| E1/E1b/E2 (rerun on corrected price) | **INVALIDATED again — stale-quote contamination** |
| E3 (corrected price) | **INVALIDATED — it IS the artifact** |
| batch 1 (E4 onward) | **INVALIDATED — killed mid-run** |

Repro:
```
python research/campaign/diagnostics/d8_baseline_beats_market.py
python research/campaign/diagnostics/d9_stale_or_gap.py
```

## D10 — what `quote_age_seconds` ACTUALLY measures, per source. My 5s filter was wrong.

Read from `scripts/build_depth.py`, the one writer of `venue_depth`:

| source | line | formula | **what it means** |
|---|---|---|---|
| `backfill` | 246, 270 | `age = (minute_mark - stamp)/1000` where `stamp` is the newest Predexon tick at or before the mark | **time since the last RECORDED book change** |
| `live`, `live_ws`, `live_touch` | 177 | `age = observed - event_time` where `observed` is the record's `available_time` | **the sampler's own polling lag** |

**These are incommensurable.** The backfill's age is data recency; the live
recorder's age is sampling delay — a constant ~32s by construction
(`record_ladder` polls at +32s), and it says nothing about whether the book moved.

CLAUDE.md already records this for the *selector* case: "`quote_age_seconds` is NOT
comparable across sources, so it cannot be the ranking key ... Rank on that and
backfill wins essentially every contested row — not because it is fresher, but
because its clock starts later." **I then made the same mistake as a FILTER.**

Measured consequence: a flat `book_age <= 5s` cut removes almost every live row,
because a live row's age is ~32s *by definition*. Of 271,723 "usable" rows,
**268,639 (98.9%) are backfill** and only 2,616 are `live` — the filter silently
deleted the independent recorder rather than the stale quotes.

**The 5s threshold is withdrawn pending the independent justification below.** It was
also chosen for a second bad reason: it was the value at which the barrier baseline's
apparent edge crossed from positive to negative, i.e. it was selected on an expected
MODEL result. That is threshold selection on the outcome, in the opposite direction
from the usual error but the same mistake.

Batch 1 stopped again; it was running under the withdrawn filter.

## D12 — threshold sensitivity. The model-free criterion I hoped for is CONFOUNDED.

Applied to BACKFILL ONLY (per D10, the live recorders' age is sampler lag, not
staleness, and the as-of join already handles it).

| max backfill age | rows | % kept | windows | symbol-windows | market log loss | market calib | market Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1s | 290,417 | 74.8% | 9,753 | 28,741 | 0.473122 | 0.00749 | 0.158622 |
| 2s | 294,873 | 75.9% | 9,756 | 28,773 | 0.471932 | 0.00755 | 0.158174 |
| 5s | 304,476 | 78.4% | 9,757 | 28,813 | 0.469948 | 0.00765 | 0.157401 |
| 10s | 316,148 | 81.4% | 9,757 | 28,826 | 0.467585 | 0.00795 | 0.156494 |
| 15s | 323,830 | 83.4% | 9,757 | 28,834 | 0.465991 | 0.00834 | 0.155857 |
| 30s | 336,292 | 86.6% | 9,757 | 28,839 | 0.464095 | 0.00851 | 0.155104 |
| 60s | 347,075 | 89.3% | 9,757 | 28,841 | 0.463091 | 0.00844 | 0.154721 |
| 120s | 359,799 | 92.6% | 9,757 | 28,841 | 0.463305 | 0.00845 | 0.154742 |
| none | 388,466 | 100.0% | 9,757 | 28,841 | 0.467599 | 0.00893 | 0.156257 |

**Coverage barely moves**: 28,741 symbol-windows at a 1s cut against 28,841
unfiltered — **0.35%**. The filter removes offsets within windows, not windows. So a
tight threshold is nearly free in window terms, which is the main practical finding.

**But the market-quote-quality criterion does NOT identify a threshold, because it is
confounded by which markets go quiet.** The market's own log loss by age band:

| age band | n | market log loss | market calib | mean abs(p-0.5) |
|---|---:|---:|---:|---:|
| <1s | 254,580 | 0.476804 | 0.00719 | 0.2467 |
| 1-2s | 4,456 | **0.394426** | 0.01388 | 0.3017 |
| 2-5s | 9,603 | 0.409018 | 0.01396 | 0.2938 |
| 5-10s | 11,672 | 0.405930 | 0.01560 | 0.2904 |
| 15-30s | 12,462 | 0.414822 | 0.01543 | 0.2871 |
| 30-60s | 10,783 | 0.431783 | 0.01131 | 0.2777 |
| 60-120s | 12,724 | 0.469149 | 0.01329 | 0.2573 |
| >120s | 28,667 | **0.521490** | 0.01556 | 0.2190 |

Log loss is **non-monotonic**: the freshest band (<1s) scores *worse* than 1-60s.
The reason is visible in the last column — the bands are different populations, with
different distances from a coin flip, so log loss confounds age with how decided the
market already is. The aggregate column in the first table is non-monotonic for the
same reason.

**Conclusion: this criterion cannot set the threshold, and it is reported as a
negative result rather than quietly dropped.** Calibration error is the one
monotone-ish signal (0.00719 at <1s rising to 0.01556 at >120s) but the spread is
small and the populations still differ. The threshold must be justified on the TAPE
evidence (D11), which asks a question about the data alone.

## D11 — the tape says the Kalshi book is NEVER quiet. The backfill's age is a
## SAMPLING GAP, not an unchanged order book. CLOSED.

Ground truth: `venue_book_events`, every Kalshi WebSocket frame the venue sent
(1.18B rows). For each backfill row, the frames the venue actually sent during the
span the backfill reports as unchanged. Full tape/backfill overlap,
2026-09-30 .. 2026-10-14, **n = 60,918**.

| backfill claimed age | n | mean frames in span | median | **% with ZERO frames** | frames/sec |
|---|---:|---:|---:|---:|---:|
| <1s | 57,519 | 350.9 | 209 | 4.78% | 350.85 |
| 1-2s | 427 | 605.2 | 353 | 2.81% | 400.47 |
| 2-5s | 964 | 1,273.5 | 749 | 2.70% | 378.06 |
| 5-10s | 682 | 2,579.5 | 1,620 | 1.47% | 359.54 |
| 10-15s | 378 | 4,785.3 | 2,750 | 4.50% | 390.46 |
| 15-30s | 558 | 8,460.8 | 5,278 | 1.61% | 388.03 |
| 30-60s | 377 | 14,545.2 | 8,447 | **0.00%** | 387.20 |

**The frame rate is FLAT at ~350-400/sec in every band.** There is no age at which
the venue goes quiet; a claimed age of N seconds corresponds to roughly 380xN frames
the backfill did not record. Zero of the 30-60s rows had a silent venue.

**So `quote_age_seconds` on the backfill measures a PREDEXON SAMPLING GAP, not an
unchanged book.** The three candidate readings in the brief resolve as:
genuinely stale observation — **yes**; unchanged but still valid order book — **no**;
missing source update — **yes**.

**One honest limit:** a "frame" is any book event at any level, not necessarily a
top-of-book move, so D11 alone does not prove the PRICE moved. D9 supplies that
half independently: the live mid moves **8.4-9.2pp per minute regardless of the
backfill's claimed age**, and backfill-vs-live disagreement grows monotonically with
it. The two together are what justify the cut.

## The threshold, chosen on MEASUREMENT ERROR and nothing else

From D9, mean abs(backfill - contemporaneous live) by claimed age. The <1s figure is
a **floor of 5.86pp** that is not backfill error at all — it is the 32-second live
sampler offset established in D7. The incremental error attributable to age is the
excess over that floor:

| claimed age | abs error | **incremental over the 5.86pp floor** |
|---|---:|---:|
| <1s | 5.861pp | — |
| 1-5s | 6.303pp | +0.44pp |
| 5-15s | 6.720pp | +0.86pp |
| 15-30s | 8.359pp | +2.50pp |
| 30-60s | 10.185pp | +4.32pp |
| 60-120s | 13.466pp | +7.61pp |
| >120s | 19.582pp | +13.72pp |

**Rule, fixed before looking at any model result: keep ages whose incremental
measurement error stays below 1pp**, which is the order of the venue's own 1-cent
tick and twice the measured 0.5c half-spread. That gives **15 seconds**.

| threshold | rows kept | % | symbol-windows | vs unfiltered |
|---|---:|---:|---:|---:|
| 1s | 290,417 | 74.8% | 28,741 | -0.35% |
| 5s | 304,476 | 78.4% | 28,813 | -0.10% |
| **15s** | **323,830** | **83.4%** | **28,834** | **-0.02%** |
| 30s | 336,292 | 86.6% | 28,839 | -0.01% |
| none | 388,466 | 100.0% | 28,841 | — |

**The previous 5s value is superseded.** It was chosen for two bad reasons: it was
applied to all sources despite D10, and it was the point at which the barrier
baseline's apparent edge changed sign — selection on an expected model result. 15s
is set by the measurement-error rule above, applied to **backfill rows only**, and
every experiment will be reported with the 1s / 5s / 30s / unfiltered sensitivity
alongside.

Repro: `python research/campaign/diagnostics/d11_multiday.py`

# ============================================================================
# D11 (REBUILT) — actual book reconstruction. VALIDATES THE BACKFILL and
# independently confirms D7. The first D11 was frame-counting and is superseded.
# ============================================================================

The first D11 counted WebSocket frames and inferred staleness from the count. That
was invalid: a frame is any event at any level, and 1,707 frames is not 1,707
economically meaningful book moves. **Superseded, not deleted.**

This version RECONSTRUCTS the book: newest snapshot at or before the instant, plus
the signed deltas with `seq > snap_seq` and `event_time <= t`, levels at or below
`MIN_SIZE=1e-3` dropped (a delta is a signed change, so a level driven to zero
accumulates float residue). `best_yes_bid` = max surviving yes price;
`yes_ask = 1 - best_no_bid`, because Kalshi's book is two BID stacks.

## Validity controls — clean

| day | frames | snapshots | tickers | seq span | distinct seqs | **seq gaps** | missing | bad quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-02 | 79,935,750 | 89,222 | 291 | 1,284,234 | 1,284,234 | **0** | 0 | 0 |
| 2026-10-05 | 76,731,121 | 88,969 | 291 | 1,266,646 | 1,266,646 | **0** | 0 | 0 |
| 2026-10-08 | 74,836,380 | 83,489 | 267 | 1,474,210 | 1,474,210 | **0** | 0 | 0 |
| 2026-10-11 | 55,933,460 | 85,502 | 291 | 927,943 | 927,942 | **1** | 1 | 0 |
| 2026-10-13 | 81,726,726 | 84,003 | 291 | 1,731,378 | 1,731,378 | **0** | 0 | 0 |

369M frames, **one missing sequence number in total**. Continuity checked GLOBALLY —
`seq` runs 1..N across every market on one subscription, so a per-ticker check would
flag every delta (CLAUDE.md records that trap).

**Coverage: 19,343 targets, MATCHED 19,325 (99.91%), UNMATCHED 18 (0.09%).**
Unmatched = no snapshot at or before the instant; reported, never folded.

**Limitation, stated: `available_time - event_time` is 0.0ms on every row**, so this
table cannot separate venue event time from local receive time — the recorder stamps
both identically. The timestamp-alignment requirement is therefore satisfied only
for backfill-vs-reconstruction, not for venue-vs-local.

## THE RESULT — reconstructed book vs backfill quote at the decision instant

| backfill claimed age | n | mean abs | **median abs** | p90 | % exact | % <=1pp |
|---|---:|---:|---:|---:|---:|---:|
| <1s | 18,051 | 1.295pp | **0.000pp** | 2.00 | **54.38%** | **83.49%** |
| 1-2s | 145 | 1.947 | 0.500 | 5.00 | 23.45% | 63.45% |
| 2-5s | 337 | 2.287 | 1.000 | 6.00 | 17.21% | 48.96% |
| 5-10s | 244 | 2.580 | 1.500 | 5.50 | 8.61% | 31.56% |
| 10-15s | 148 | 4.402 | 2.000 | 12.00 | 6.76% | 31.08% |
| 15-30s | 247 | 4.549 | 3.000 | 10.00 | 5.26% | 18.62% |
| 30-60s | 153 | 7.509 | 5.000 | 18.62 | 1.31% | 9.80% |

**At age <1s the backfill mid is EXACT 54.4% of the time, median error 0.000pp.**

**This independently confirms D7 and retires the D9 "5.86pp floor".** D9 measured
backfill against the live LADDER and found 5.86pp even at <1s; against the book
reconstructed at the ACTUAL decision instant the error is 1.295pp mean / 0.000pp
median. The 5.86pp was the live sampler's 32-second offset, exactly as D7 said —
now confirmed by a second, independent method.

Stable across symbols (BTC 1.597, ETH 1.374, SOL 1.373), dates (1.100-2.092) and
offsets (1.201-1.979, no trend). By price band, error is largest mid-book
(0.7-0.9: 2.089pp) and smallest in the tails (<0.1: 0.734pp; >0.9: 0.756pp).

## The eligibility rule — tick-referenced, model-free, fixed before any model result

**Criterion: the median reconstruction error must not exceed ONE VENUE TICK.**
Kalshi's tick is 1 cent = 1.0pp in [0.10, 0.90]. Median error by band: <1s 0.000,
1-2s 0.500, **2-5s 1.000**, 5-10s 1.500, 10-15s 2.000, 15-30s 3.000, 30-60s 5.000.

| label | rule | justification |
|---|---|---|
| **verified** | backfill age <= **5s**; or source in (`live`, `live_ws`) | median error <= 1 tick |
| **questionable** | backfill age 5-60s; or `live_touch` any age | median 1.5-5pp; live_touch unvalidatable post-2026-09-16 (D5b) |
| **unsuitable** | backfill age > 60s | median >= 5pp |

**This lands on the same 5s as the withdrawn value, for an entirely different and
valid reason.** The withdrawn version was a global cut across incommensurable
quantities (D10) chosen at the point a model result changed sign. This one is
source-specific and set by reconstruction error against the venue's own tick, with
no model metric involved. The coincidence is noted so it cannot be mistaken for the
earlier reasoning surviving.

Repro: `python research/campaign/diagnostics/d11_rebuild.py`

## NC1 (first version) — INVALID CONTROL, my design error. Superseded by NC1c.

NC1 permuted the LABEL within 2-day blocks and left the market price in place, then
asked whether the model still beat the market. It read **model_minus_market
+0.359158, CI [+0.3468, +0.3715]**, which the script flagged as "LEAK SUSPECTED".

**That is not a leak. The control is wrong for a market-relative model.** Once the
label is shuffled the market quote no longer predicts it either, so the market
becomes a confidently-WRONG forecast; any model that merely shrinks its predictions
toward 0.5 beats it by a large margin. The number measures the destruction of the
benchmark, not skill in the model.

**Corrected design (NC1c):** permute the FEATURES against the (label, market price)
pair, keeping label and price together. The market retains exactly its real
predictive power and the only destroyed relationship is features -> outcome. A model
that still beats the market under that permutation is reading something it should not.

NC2 (noise-only features) and NC3 (shuffled price) are unaffected and are retained.
NC1's number is preserved here as an invalid result, not deleted.

## METHOD NOTE — the two runners fit differently. Cross-arm comparison is INVALID,
## but the difference is itself a controlled result on overfitting.

`runner._lgbm` (used by the registered batch E4-E10 and the book batch) fits
`m.fit(Xtr, ytr, init_score=itr)` with **no eval set and no early stopping** — all
300 trees, always. `boosting_audit.lgbm_fit_predict` (arms A/B/C) uses a chronological
inner split with 50-round early stopping, which is the correct protocol.

That makes E5 vs A an accidental but clean controlled comparison: same features, same
market init score, nearly the same hyperparameters, differing mainly in early stopping.

| arm | early stopping | min_child | trees | model_minus_market |
|---|---|---:|---|---:|
| **A** (boosting runner) | **yes**, 50 rounds | 500 | stops at [10, 1, 26, 41] | **+0.000195** |
| **E5** (campaign runner) | **no** | 200 | all 300 | **-0.003791** |

**Without early stopping the model overfits by ~0.004 nats** — four times the size of
any effect under study. This answers the underfitting-vs-overfitting question with a
measurement rather than an inference: **both failure modes are reachable on this
data, and early stopping is doing real work.** It also means the deployed
configuration's early stopping is not a formality.

**Consequence: E4-E10 and the book candidates are fitted under a weaker protocol than
A/B/C and must not be ranked against them.** They are reported in their own block,
labelled, rather than silently merged into one leaderboard. Rerunning the registered
batch with early stopping is left undone deliberately — the pre-registration fixed
the specifications and the budget, and changing the fitting protocol mid-campaign
after seeing results is exactly the move the protocol forbids. It is recorded as the
single most informative follow-up.

# ============================================================================
# STAGE 9/11/12/14 ARE INVALID — LOOKAHEAD IN MY OWN FEATURE PIPELINE.
# NOT PROMOTED. Found while investigating the edge-magnitude inconsistency.
# ============================================================================

The user asked for a check before promotion. The check found a disqualifying bug,
and it is mine, not the repo's.

`build_snapshot_flow()` and `book_and_peer()` both end with:

    d.sort_values('available_time').drop_duplicates([...], keep='last')

`keep='last'` takes the LATEST revision of each (symbol, window, offset) **regardless
of whether it was knowable at the decision instant**. The repo's own `attach_quotes`
applies an as-of rule — the newest row with `available_time <= window_open + offset
+ latency` — and my pipeline bypassed it entirely by reading `venue_depth` directly.

Measured, against a decision instant of `window_open + offset + 3s`:

| venue | rows | **rows whose quote was NOT yet available** | median lead | max |
|---|---:|---:|---:|---:|
| kalshi | 433,748 | **109,685 (25.3%)** | 25s | 75s |
| polymarket | 451,905 | **216,721 (48.0%)** | 30s | 57s |

The offending rows are `live` (89,957) and `live_touch` (19,031) for Kalshi and
`live` (216,709) for Polymarket — exactly the sources whose `available_time` is the
sampler's +32s lag (D7). `pm_mid` and `pm_spread` were in the feature set, and
**48% of Polymarket rows carried future information.**

## What this invalidates

| stage | reported | status |
|---|---|---|
| 9 full training | +0.000534 row, $102.75 @t=3.33 | **INVALID** |
| 11 neural / tree weekly | tree +0.001050 row, $240.78 @t=6.21 | **INVALID** |
| 12 silence analysis | $249.52 in last 5 folds | **INVALID** |
| 14 isolation | base +0.001227 | **INVALID** |

**Stage 13 is the clean measurement**, because it builds features through the repo's
`attach_quotes`, which enforces availability:

    row      +0.000100  CI [-0.000001, +0.000199]   does NOT beat
    sym-win  +0.000239  CI [+0.000002, +0.000469]   marginal
    window   +0.000201  CI [-0.000095, +0.000495]   does NOT beat

That 10x gap between Stage 12 and Stage 13 was never six features. It was 25-48%
future-dated quotes.

## Why it was not caught earlier

Every warning sign was present and each had an innocent reading:
- **+4.9pp per contract from +0.001 nats** -- attributed to trade selection.
- **t=6.21**, which trips the repo's own `sharpe_implausible` instinct.
- **realised edge EXCEEDING predicted** (+5.97 vs +4.53pp) -- read as underconfidence;
  it is the signature of scoring against prices the model partly already saw.
- **a monotone edge gradient with offset** -- tested for carry-forward (D8's
  mechanism), which it was not, and then accepted. The lookahead hypothesis was not
  tested because D7 had already "settled" the availability question for the CAMPAIGN
  harness, and these overnight scripts never used that harness.

**The lesson: `harness.market_prices()` enforces the as-of rule; the overnight
feature builders did not, and nothing made them share it.** The availability rule
must live in ONE function that every feature path calls.
