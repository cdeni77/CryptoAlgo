# Diagnostics report — data risks resolved before training

Run 2026-10-17. Seven diagnostics against local data. Every number is a query result.

**This report contains a correction to itself.** D3/D4/D5 reached a conclusion that
D7 overturned. Both are kept: the superseded findings are marked, not deleted,
because the reasoning that produced them is the reason the final answer is trusted.

---

## Summary

| id | question | verdict |
|---|---|---|
| D1 | Is the live +$17.53 distinguishable from noise? | **No.** t=0.171, CI [-$181, +$220] |
| D2 | Do >90c favourites carry the P&L? | **No.** 23.8% of total, t=0.27 |
| D3 | How does quote age distort the market benchmark? | Monotonic; magnitudes inflated by the D7 error |
| D4 | Do cent-quantised prices create false edge? | **No.** The backfill is unbiased |
| D5 | Is the backfill a legitimate point-in-time price? | ~~No~~ → **YES** (corrected by D7) |
| D5b | Is `live_touch` trustworthy? | **Unvalidatable** in the post-fix era |
| D6 | Are the 49 features consistently populated? | **No.** 14/49 depend on a book with 3 provenances |
| D7 | Which source was actually sampled WHEN? | **The live ladder is +32s. It was the lookahead.** |

---

## D1 — The live +$17.53 is not distinguishable from noise

Venue's own ledger: 1,289 settlements, 6,692 contracts, $3,138.48 deployed, $83.83 fees.

| | |
|---|---|
| total venue P&L | **+$17.53** |
| per contract | **+0.262pp** (pre-registered: +1.5pp) |
| independent 15-min windows | 936 |
| cluster-robust t | **0.171** |
| block bootstrap 95% CI | **[-$181.36, +$219.84]** |
| **P&L without the single largest winner** | **-$0.85** |

Monthly: Aug +$8.19 (t=0.11), Sep -$8.22 (t=-0.13), Oct +$17.56 (t=0.52). Top five
settlements by |P&L| sum to $48.50 against a $17.53 total. **One trade is the whole
result.** Excluded from model selection.

## D2 — >90c favourites do NOT carry the profitability

| price bucket | n | contracts | P&L $ | pp/ct | win% (ct-wtd) | break-even% | excess pp | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [0.00,0.10) | 43 | 379 | -9.34 | -2.465 | 5.01 | 7.48 | -2.465 | -0.78 |
| [0.10,0.30) | 284 | 1,873 | +15.94 | +0.851 | 22.21 | 21.36 | +0.851 | 0.27 |
| [0.30,0.50) | 304 | 1,496 | -0.15 | -0.010 | 40.51 | 40.52 | -0.010 | -0.00 |
| [0.50,0.70) | 283 | 1,207 | -12.40 | -1.027 | 59.32 | 60.35 | -1.027 | -0.27 |
| [0.70,0.80) | 161 | 661 | -22.05 | -3.335 | 73.24 | 76.57 | -3.335 | -0.83 |
| [0.80,0.90) | 128 | 586 | +41.37 | **+7.065** | 92.48 | 85.42 | +7.065 | **2.79** |
| [0.90,0.95) | 83 | 475 | +3.58 | +0.754 | 93.05 | 92.30 | +0.754 | 0.23 |
| [0.95,1.01) | 3 | 15 | +0.58 | +3.887 | 100.00 | 96.11 | +3.887 | 11.34 |

Break-even% is `(cost + fees) / contracts`. **>=90c is 23.8% of P&L on t=0.27** —
the hypothesis is rejected. The one bucket with a t-stat is 1 of 8 tested on n=128.
Realized outcomes beat the price-implied break-even essentially nowhere.

## D7 — The correction: which source was sampled when

**The question D3/D4/D5 failed to ask.** Every source stamps `event_time` at
`window_open + offset` exactly. `available_time` does not agree:

| source | available_time - stamp | own quote_age |
|---|---:|---:|
| backfill | **+0s** | 0.33s |
| live / live_ws | **+32s** | 32.6s |
| live_touch | +3s | NaN |

`record_ladder` samples at ~32 seconds past the minute it labels. **A `live` row at
offset m describes the book at m+32s and was not knowable at the decision instant.**

For a decision at offset m (+3s latency): `backfill@m` was available; `live@(m-1)`
was available; **`live@m` was not — it is 32 seconds of lookahead.**

### The decisive test (n=255,773 over 1,965 windows)

| price | log loss |
|---|---:|
| backfill @ m (available at m) | 0.484792 |
| live @ m-1 (available at m-28s) | 0.486307 |
| live @ m (available at **m+32s**) | 0.439771 |

Paired bootstrap over windows:

| comparison | diff | 95% CI |
|---|---:|---|
| **backfill@m − live@m-1** (both available) | **-0.000040** | **[-0.00286, +0.00276]** |
| backfill@m − live@m (lookahead) | +0.046777 | [+0.04377, +0.04986] |
| live@m-1 − live@m (pure 60s of freshness) | +0.046817 | [+0.04557, +0.04807] |

**The backfill's entire apparent handicap is timing.** Its gap to `live@m` is
statistically identical to what 60 seconds of pure freshness produces, and against a
contemporaneous price it is indistinguishable from zero.

### Overturned

- ~~"the backfill is not an executable price"~~ — it **is** the correct
  point-in-time price for a decision at offset m.
- ~~"the trustworthy universe is 2,161 windows over 23 days"~~ — it is
  **9,757 windows over 105 days**.
- ~~"measuring against the backfill inflates `model_minus_market`"~~ — **backwards**:
  measuring against `live@m` makes the benchmark unfairly hard.

### Surviving from D3/D4/D5

- **D4: the backfill is unbiased.** Conditioning on the live mid gives a
  +6.7/-5.8pp mean-reverting gap; conditioning on the *backfill* mid collapses it to
  ±0.5pp. The signs disagree, so the pattern was regression to the mean from
  independent noise. Overall gap -0.056pp (t=-1.70). Cent quantisation is real
  (0.00% of backfill prices off the whole-cent grid vs 25.70% of live) but
  second-order.
- **D3: the quote-age gradient is real** (log loss rising monotonically from age<5s
  to age>300s), but the magnitudes were measured against a lookahead reference and
  are inflated.
- **D5's "matches offset-1"** is reinterpreted: a 60-second grid resolving a
  ~32-second offset, not a stamping error.

## D5b — `live_touch` cannot be validated in the era that matters

Unbiased against the stream ladder (mean gap -0.0257pp) but noisy (mean |gap|
5.2651pp). **All 43,992 paired rows carry `transport = null`** — every one predates
the 2026-09-16 fix for `live_touch` pricing off `/markets`' cached touch. The stream
ladder stops 2026-09-17, so the only available overlap *is* the contaminated era, and
the post-fix data has no independent reference. Documented as a blocker; `live_touch`
is used where the as-of rule selects it (6.3% of rows) but is not relied on alone.

## D6 — The deployed model's features are not consistently populated

49 features in 8 groups; **14 depend on the venue's book** (`cross_venue` 4,
`market_state` 5, `implied_vol` 5).

| month | kalshi stream | kalshi backfill | live_touch | polymarket |
|---|---:|---:|---:|---:|
| 2026-07 | 0 | 8,046 | 0 | 8,662 |
| 2026-08 | 1,971 | 7,750 | 1,089 | 7,140 |
| 2026-09 | 4,518 | 8,510 | 8,538 | 8,640 |
| 2026-10 | 0 | 3,725 | 4,557 | 4,628 |

Implied-vol fits, and the share clearing the live `r2 >= 0.90` gate:

| month | BTC | ETH | SOL | BTC r2 | ETH r2 | SOL r2 |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07 | 2,448 | 1,257 | 1,129 | 0.869 | 0.839 | 0.847 |
| 2026-08 | 8,910 | 2,094 | 1,210 | 0.981 | 0.924 | 0.850 |
| 2026-09 | 35,590 | 31,606 | 4,261 | **1.000** | **1.000** | **1.000** |
| 2026-10 | 18,133 | 14,268 | **28** | 1.000 | 1.000 | 1.000 |

**SOL implied vol is effectively absent in October** (28 fits vs BTC's 18,133), and a
quality metric that jumps from 0.85 to exactly 1.000 and stays there is a change in
the collector, not in the market. Feature distributions are not comparable across the
deployed model's training span.

---

## The corrected evaluation universe

As-of join: the usable quote at offset m is the most recent Kalshi row with
`available_time <= window_open + m + 3s`. One rule, every source.

| | |
|---|---|
| span | 2026-07-01 → 2026-10-14 |
| rows | **388,466** |
| windows | **9,757** |
| symbol-windows | **28,841** |
| rows with a quote from the future | **0** (asserted) |
| median quote age at the decision instant | **3.0s** |
| sources selected | backfill 90.8%, live_touch 6.3%, live 2.9% |

**All inference resamples windows, not rows** — four offsets share a settlement, the
three symbols are ~0.7 correlated within a window, and consecutive windows chain.

## Reproduction

```bash
cd backend/trader
docker compose exec -T db psql -U postgres -d trades_db -At -F$'\t' -c \
  "select ticker,market_result,yes_contracts,no_contracts,yes_cost,no_cost,\
revenue,fee_cost,pnl,settled_time from venue_settlements order by settled_time" > vs.tsv
python research/campaign/diagnostics/d1_d2_live_pnl.py vs.tsv
python research/campaign/diagnostics/d4_quantisation.py        # writes d4_pairs.parquet
python research/campaign/diagnostics/d4b_which_price_is_real.py
python research/campaign/diagnostics/d3_quote_age.py
python research/campaign/diagnostics/d5_backfill_alignment.py
python research/campaign/diagnostics/d5b_live_touch.py
python research/campaign/diagnostics/d6_feature_population.py
python research/campaign/diagnostics/d7_timing_correction.py   # THE correction
```

---

## D8 / D9 — a fresh forecast against a carried-forward price manufactures edge

E3 (the barrier baseline alone, no ML) scored `model_minus_market` **+0.007963**
(p=0.0000) on the unfiltered panel. Audited rather than believed, per protocol §5.

Apparent edge by the backfill's own `quote_age_seconds`:

| book age | n | windows | model_minus_market | 95% CI |
|---|---:|---:|---:|---|
| <1s | 215,289 | 7,715 | **-0.006185** | [-0.00772, -0.00468] |
| 5-15s | 17,691 | 4,111 | +0.005683 | [+0.00062, +0.01070] |
| 30-60s | 11,784 | 3,187 | +0.025879 | [+0.01805, +0.03375] |
| >120s | 17,429 | 2,801 | **+0.172620** | [+0.15841, +0.18662] |

Monotonic over seven bands, backfill-only (live -0.000085, live_touch -0.005606),
rising monotonically with offset. **On fresh quotes the baseline LOSES.**

D9 asked whether a stale quote is a quiet book: the live book moves **8.4-9.2pp per
minute regardless of the backfill's claimed age**, with 72-77% of rows moving >2pp/min
in every band including >120s.

## D10 — `quote_age_seconds` means different things per source

| source | formula (`scripts/build_depth.py`) | meaning |
|---|---|---|
| `backfill` | `(minute_mark - last_tick)/1000` | time since the last RECORDED change |
| `live`, `live_ws`, `live_touch` | `available_time - event_time` | the sampler's own polling lag (~32s) |

A single global cut across both deleted **98.9% of live rows** for a reason unrelated
to staleness. CLAUDE.md records this trap for the column used as a *selector*; it was
repeated here as a *filter*. All thresholds are now source-specific.

## D11 (rebuilt) — actual book reconstruction

The first D11 counted frames and inferred staleness from the count. Invalid: a frame
is any event at any level. **Superseded, retained.**

Rebuilt by reconstructing the book — newest snapshot at or before t, plus signed
deltas with `seq > snap_seq` and `event_time <= t`, levels at or below `MIN_SIZE=1e-3`
dropped, `yes_ask = 1 - best_no_bid`.

**Validity:** 369M frames over 5 days, **1 missing sequence number in total**
(continuity checked GLOBALLY — `seq` runs across all markets on one subscription).
Coverage **19,325 / 19,343 matched (99.91%)**; 18 unmatched reported, never folded.

| backfill claimed age | n | mean abs | **median abs** | % exact | % <=1pp |
|---|---:|---:|---:|---:|---:|
| <1s | 18,051 | 1.295pp | **0.000pp** | **54.38%** | 83.49% |
| 2-5s | 337 | 2.287 | 1.000 | 17.21% | 48.96% |
| 5-10s | 244 | 2.580 | 1.500 | 8.61% | 31.56% |
| 15-30s | 247 | 4.549 | 3.000 | 5.26% | 18.62% |
| 30-60s | 153 | 7.509 | 5.000 | 1.31% | 9.80% |

**This confirms D7 by an independent method and retires D9's "5.86pp floor"** — that
floor was the live sampler's 32-second offset, not backfill error. Stable across
symbols (1.37-1.60), dates (1.10-2.09) and offsets (1.20-1.98).

**Limitation:** `available_time - event_time` is 0.0ms on every row, so this table
cannot separate venue event time from local receive time.

## D12 — threshold sensitivity; the model-free quality criterion is CONFOUNDED

| max backfill age | rows | % kept | symbol-windows | market log loss |
|---|---:|---:|---:|---:|
| 1s | 290,417 | 74.8% | 28,741 | 0.473122 |
| 5s | 304,476 | 78.4% | 28,813 | 0.469948 |
| 15s | 323,830 | 83.4% | 28,834 | 0.465991 |
| none | 388,466 | 100.0% | 28,841 | 0.467599 |

**Coverage barely moves** (0.35% of symbol-windows between 1s and unfiltered) because
the filter drops offsets within windows, not windows. But the market quote's own log
loss by age band is **non-monotonic** (<1s 0.4768 vs 1-2s 0.3944) because the bands
are different populations at different distances from a coin flip. **Reported as a
negative result: this criterion cannot set the threshold.**

## The eligibility rule — tick-referenced, no model metric

Kalshi's tick is 1 cent = 1.0pp. Median reconstruction error <= 1 tick holds through
the 2-5s band.

| label | rule |
|---|---|
| **verified** | backfill age <= 5s, or source in (`live`, `live_ws`) |
| **questionable** | backfill 5-60s, or `live_touch` at any age (D5b unvalidatable) |
| **unsuitable** | backfill > 60s |

Frozen dataset `market_v2.parquet`, sha256 `3c729b61cf17967d...`: **388,466 rows =
279,899 verified / 67,176 questionable / 41,391 unsuitable**, 9,701 verified windows,
28,594 verified symbol-windows, 2026-07-01 -> 2026-10-14.
