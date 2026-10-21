# Phase 1 — Local data inventory and coverage

Measured 2026-10-17 directly off `data/research` (DuckDB over Parquet, deduped
on each dataset's own `event_key`), `models/`, and the `trades_db` serving store.
Nothing below is taken from documentation; every number is a query result.
Machine-readable companion: `datasets.json`.

## 1. What exists

| dataset | rows (per symbol) | span | notes |
|---|---|---|---|
| `minute_bars` | 2.70M | 2021-08 → 2026-10-17 | Coinbase 1m OHLCV. Effectively complete now (Sep 2026 = 43,200 = 30x1440 exactly). |
| `venue_depth` | 818–821k | 2026-01 → 2026-10 | **The market price.** Kalshi + Polymarket, every minute 0..15 of the window. |
| `venue_settlements` | 76,074 kalshi / 80,351 pm tickers | 2026-01-06 → 2026-10-17 | Venue-authoritative labels, both venues. |
| `venue_implied_vol` | 79.5k / 51.8k / 7.4k | 2026-01 → 2026-10 | KXBTCD ladder inverted to forward sigma. SOL thin. |
| `venue_ladder` | 59.5k | **2026-08 → 2026-09 only** | Raw Kalshi ladder. Stops 2026-09-30. |
| `pm_ladder` | 73.9k | 2026-08 → 2026-10 | Polymarket raw ladder. |
| `venue_book_events` | 1.18B total | 2026-09-30 → 2026-10 | Raw WS frames, 14-day retention tier. |
| `spot_trades` | 1.31M/586k/289k | 2026-10-14 → | Coinbase tape. ~3 days only. |
| `index_proxy_1s` | 218.8k | 2026-10-14 → | Per-second index proxy. ~3 days only. |
| `venue_quotes` | **absent from disk** | — | Schema declared, no partitions written. ~12 modules still read it. |

Raw, outside the research store: `data/collection/` 47 GB of per-window JSONL/Parquet
(Kalshi + Polymarket, 2026-01 → 2026-10), `data/trading.db` 2.2 GB SQLite scrape
source, `data/iv_ladder.jsonl`, `data/quote_backfill.json`.

## 2. The trainable universe (market-relative), July 2026 → present

Join: `venue_depth` (venue=kalshi) x `venue_settlements` (venue=kalshi, label non-null),
on `(symbol, window_open)`, offsets 1..14.

| month | source | rows | two-sided quote | quote_age<=60s | symbol-windows |
|---|---|---:|---:|---:|---:|
| 2026-07 | backfill | 108,715 | 106,357 | 87,464 | 8,012 |
| 2026-08 | backfill | 108,254 | 106,595 | 94,203 | 7,745 |
| 2026-08 | live / live_ws / live_touch | 26,862 / 19,334 / 4,329 | — | — | 1,971 / 1,416 / 1,089 |
| 2026-09 | backfill | 118,097 | 116,359 | 110,167 | 8,508 |
| 2026-09 | live / live_ws / live_touch | 62,454 / 62,454 / 34,127 | — | — | 4,518 / 4,518 / 8,538 |
| 2026-10 | backfill | 52,150 | 51,051 | 52,138 | 3,725 |
| 2026-10 | live_touch | 18,134 | 18,134 | 2,869 | 4,539 |

**Headline: ~387k backfill rows / ~28k symbol-windows with a real Kalshi quote and a
venue-authoritative label, July 1 → Oct 14 2026.** Label base rate over that span is
0.5024 (n=30,705 symbol-windows).

Earlier months are available too (Jan–Jun 2026, ~351k more backfill rows) but May and
June are thin (17.2k / 32.3k rows vs ~110k for a full month).

## 3. Availability tiers (do not force the intersection)

* **Tier A — bars only.** 2021-08 → now, all symbols. Baseline + vol_state + geometry
  + cross_asset + clock. No market price, so no market-relative target.
* **Tier B — bars + Kalshi backfill quote.** 2026-01 → 2026-10-14. Adds the market
  probability, spread, depth buckets. This is the only tier large enough for a
  market-relative model. Carries `quote_age_seconds` (mean 5–50s by month) and is
  quantised to whole cents outside [0.10, 0.90].
* **Tier C — + implied vol.** BTC 79.5k / ETH 51.8k rows; SOL 7.4k is too thin to train on.
* **Tier D — live-recorded book.** Kalshi `live`/`live_ws` 2026-08-27 → 2026-09-30 only;
  `live_touch` (4 decision offsets) 2026-08 → now. Deci-cent precise, no staleness.
  Small: ~5,000 symbol-windows.
* **Tier E — tape / index proxy.** 2026-10-14 → now, ~3 days. Not trainable yet.

## 4. Defects and hazards found

1. **`venue_quotes` does not exist on disk.** Any module reading it gets nothing.
2. **Polymarket `venue_settlements` is 590x duplicated on disk** — 47.4M raw rows
   collapsing to 80,351 on `event_key`. `read()` dedups correctly, so this is storage
   waste, not a correctness bug, but it makes naive `read_parquet` scans wrong.
3. **Kalshi live ladder recording stops 2026-09-30.** October has only `live_touch`
   (4 offsets) and `backfill` (to Oct 14). The WS-vs-REST retirement gate cannot be
   evaluated on October data.
4. **`quote_age_seconds` is not uniform.** Backfill monthly means swing 5.7s (Apr) to
   49.8s (Feb). Comparing a fresh forecast to a stale price manufactures edge; this
   must be a filter or a feature, never ignored.
5. **Backfill cent-quantisation** outside [0.10, 0.90] is ~0.24c, ~16% of a 1.5pp gate.
6. **Label noise ~3%**, concentrated entirely in near-ties (Coinbase proxy vs BRTI).
7. `transport` is absent from `venue_depth` rows written before 2026-09-16.

## 5. The currently promoted model

`models/promotions/20261014T143301Z.json` — **`passed=false`, `forced=true`,
installed anyway.** Force reason: "first point-in-time model; dry-run forward test".

Failed gates:

| gate | value | threshold |
|---|---|---|
| `model_minus_market` | **-0.000216** | >= 0 |
| `model_minus_market_p` | 0.545 | <= 0.05 |
| `min_folds` | **3** | >= 6 |
| `calibration_vs_market` | +0.001493 | <= 0 |

Passed `log_loss_skill` +0.000790 (p=0.025) — it beats `F(x/sigma)`, it does **not**
beat the Kalshi price, and the sign on `model_minus_market` is negative. Deflated
Sharpe 0.480 after 18 trials (raw 1.836, expected max under the null 1.897).

Realised live P&L from the venue's own settlements (`trades_db.venue_settlements`,
no nulls): Aug +$8.19, Sep -$8.22, Oct +$17.56 — **+$17.53 total on 6,692 contracts
over 1,289 settlements, = +0.26pp per contract** against a pre-registered +1.5pp.
Live predictions: 56,862 rows, 1,028 traded (1.8% coverage).
