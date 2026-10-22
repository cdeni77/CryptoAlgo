# Pre-registration — would the resting orders actually have filled?

Written **before any fill is simulated**, per `RESEARCH_PROTOCOL` §1 and §6.

## The question

The 2026-10-22 backtest returned **+155% and Sharpe +11.49** with
`--maker-entry`, which books **every resting order as filled at the bid**. The
`sharpe_implausible` gate failed on it and was right to. The forecast edge
behind it is `log_loss_skill +0.00074`, so a realised edge of **3.148pp per
contract** cannot come from forecasting — it comes from the fill assumption.

**Which of those orders would a real tape have filled?**

## The data, and why it can answer this

`venue_trades`: 495,025 rows, 2026-08-25 to now, one row per
`(market_ticker, window_open, offset_minutes, taker_book_side)` carrying
`n_prints`, `contracts`, `min_price`, `max_price`, `vwap`.

**`taker_book_side == 'ask'` is the SELLING aggressor**, which is what fills a
resting bid. Established from the data, not assumed: over the 245,431
market-minutes where both sides print, `ask` prints below `bid` 71.6% of the
time, mean vwap 0.4989 against 0.5058. That 0.69c gap is the spread, which is
what it must be if `ask` prints are hitting bids. The literal reading of the
field name inverts every result and has done so before
(`project-taker-book-side-inverted`).

## The fill rule — fixed now

A resting bid of `C` contracts at limit `L`, live over minutes `M`, fills if

```
any minute m in M with taker_book_side == 'ask' and min_price <= L
```

Partial fills are counted by `contracts` available on that side in `m`.

## What this measures, and what it cannot — fixed now

It is an **UPPER BOUND on fills**, so a failure to fill is solid and a fill is
optimistic. Three reasons, all unfixable from this data:

1. **Queue priority is invisible.** Other resting orders at `L` are ahead of
   ours and absorb prints first. The tape is anonymous; it cannot say how much
   of the minute's volume was already spoken for.
2. **The minute is an aggregate.** `min_price <= L` says a print reached our
   level somewhere inside that minute, not that it happened while we rested or
   in what order.
3. **Our own prints are in the tape.** Our taker orders appear as aggressors.
   They are a small share and are not removed; this inflates available volume
   slightly, in the optimistic direction like everything else here.

**So the honest claim is one-directional**: orders this says would NOT have
filled, certainly did not. Orders it says WOULD have, might not.

## The deciding comparison

Re-score the same backtest counting only simulated fills:

* `fill_rate` — the share of resting orders the tape would have filled
* `realised_edge_pp` on filled orders only, against the `3.148` the
  fill-everything run reported
* `total_return` and `Sharpe` against `+155%` and `+11.49`

## What counts as which outcome — fixed now

* **The assumption is broadly sound** if `fill_rate > 0.80` AND the realised
  edge on filled orders stays within 20% of 3.148pp. Then the money numbers
  stand and the `sharpe_implausible` failure needs another explanation.
* **The assumption is the artefact** if `fill_rate < 0.50` OR the realised edge
  on filled orders falls below 1.0pp. Then +155% is not a result, the maker
  strategy is unproven, and `--maker-entry` in production is trading on it.
* **Adverse selection is confirmed** if the realised edge on FILLED orders is
  materially below the edge on all orders — we get the fills we do not want.
  This is the outcome prior work points at: the maker edge flips negative at a
  2c run-through (`project-maker-fills-are-adverse`).

## Stated in advance

I expect `fill_rate` well under 1.0 and the edge on filled orders below the
edge on all orders, because a resting bid fills exactly when someone is keen
to sell into it. If it comes back at `fill_rate ~1.0` with the edge intact,
**suspect the simulation before believing the result** — most likely the side
has been read backwards, which is the error this field has already caused
once.

## What would invalidate this run

* Treating `taker_book_side == 'bid'` as the seller.
* Counting a fill from a minute outside the order's life.
* Comparing filled-order edge against a differently-sampled baseline rather
  than the same run's own all-orders figure.
* Reporting Sharpe from a halted book without saying it halted — the
  fill-everything run halted at the bankroll floor after 12 days, and the
  deflated Sharpe was 0.880 against a raw 11.49.
