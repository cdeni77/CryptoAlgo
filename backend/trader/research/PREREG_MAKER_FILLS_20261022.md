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

---

## AMENDMENT, 2026-10-22 — the premise above is WRONG, the question survives

**"The backtest books every resting order as filled at the bid" is false.**
Verified: `core.decide.as_maker_prices` is **never called** anywhere in
`core/` or `scripts/` -- it is imported by `core/backtest.py` and not used.
Under `--maker-entry` the backtest **lifts the ASK**, deliberately, and says
so in its own comment.

What `--maker-entry` actually does to a backtest is zero the FEE:
`Config.__post_init__` sets `assume_maker` from it (`core/config.py:960`), and
`core/costs.py` then charges the maker rate, which is 0 for these series. The
price is untouched.

```
            fee     price
backtest    zero    lifts the ask   (conservative on the spread)
live        zero    rests at the bid (scripts/live.py:1033)
```

So the backtest pays the WORSE price and no fee. That bounds the maker case
from below on price, which is what the comment in `core/backtest.py` claims
and is correct.

### Therefore the +155% is NOT a fill-pricing artefact

The earlier claim in this file, and what I told the operator, attributed
`Sharpe +11.49` and `realised_edge_pp 3.148` to optimistic maker pricing.
That attribution is withdrawn. The run was priced at the ask. The remaining
explanations are the ones its own output already gave: **12 days** of
continuous book, `HALTED at the bankroll floor`, and a **deflated Sharpe of
0.880** against the raw 11.49.

### The question that DOES survive, restated

The backtest models **no fill process at all**: every decision that clears the
gates becomes a position. Under `--maker-entry` those are orders that would
rest, and a resting order does not always fill.

**So the open question is fill RATE, not fill PRICE:** of the orders the
backtest counts, how many would a real tape have filled, and are the ones
that fill a fair sample of the ones we wanted?

Everything below the original "The data" heading stands: the rule, the
upper-bound caveats, and `taker_book_side == 'ask'` as the selling aggressor,
which was established from the data and is unaffected by this amendment.

### One consequence worth measuring separately

Live prices at the BID and the backtest at the ask, so **live clears the edge
gate on trades the backtest refuses**. The divergence is deliberate and
documented, but it means the backtest understates live's trade COUNT, not
just its P&L per trade. That is a different comparison from the fill rate and
needs its own pre-registration.
