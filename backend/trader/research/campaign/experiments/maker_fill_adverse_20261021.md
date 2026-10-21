# Maker fills are adversely selected — 2026-10-21

**Question.** The maker result (-$77 taker -> +$2,446 maker) assumes every
resting order fills. Does it? Measured against the completed tape: 27.4M prints,
16,269 markets, `taker_book_side` populated.

**First, a convention that inverts the answer.** `taker_book_side='ask'` is the
SELLING aggressor — its prints land on the bid (vwap - yes_bid = -0.0001, n=188,451),
while `'bid'` prints land on the ask. The natural reading is backwards, and
`backfill_trades.py`'s docstring never names which value is the selling side.

**Fill RATE is not the binding constraint.** A minute carries a median 724
selling prints with an 11c price range, so a resting order at the bid is reached
in ~99.9% of minutes. The interesting risk was never whether we are reached.

**Adverse selection, measured.** `margin` = how far the most aggressive print
ran BEYOND our limit. Monotone across all seven buckets:

```
margin through   n     edge/ct    win%       pnl
      <=0        7   +22.429pp   100.0%   $+18.13
     0-1c      117    +8.454pp    91.5%   $+85.03
     1-3c      425    +7.744pp    76.0%  $+225.84
     3-6c      405    +6.133pp    69.1%  $+196.88
    6-12c      465    +1.700pp    60.2%   $+56.54
   12-25c      286    -7.767pp    53.8%  $-207.86
     >25c       85   -26.102pp    49.4%  $-187.22

window-clustered corr(margin, edge/ct) = -0.2506 [90% -0.295 .. -0.202], 1,510 windows
```

**The interaction with the queue is what hurts.** A deep run-through means the
level was CONSUMED, so the fill is near-certain — and those are the losing
buckets. A shallow touch may never clear the queue ahead of us, so the fill is
doubtful — and those are the profitable buckets. **We get the fills we do not
want.**

```
assume we fill only if price ran      n          pnl    edge/ct
>= 0.0c through our limit          1788     $+182.33   +1.801pp
>= 1.0c through our limit          1666      $+84.18   +1.276pp
>= 2.0c through our limit          1430      $-95.85   -0.490pp   <- flips
>= 3.0c through our limit          1252     $-144.79   -0.990pp
>= 6.0c through our limit           851     $-356.52   -4.381pp
```

**The maker edge flips negative between a 1c and 2c requirement.**

## What this does NOT establish

1. **Run-through depth is a PROXY for fill certainty, not a measurement.** We
   have no queue position. The inference "deeper = more certain fill" is
   economic reasoning, not data.
2. **The minute bucket is too coarse.** `backfill_trades._rows` keeps
   min/max/vwap per (minute, side) and discards per-print timestamps. An order
   resting 60s from the decision instant needs sub-minute resolution; an 11c
   intra-minute range cannot be attributed to the 60 seconds we were resting.
3. ~~The margin effect may be a volatility proxy.~~ **TESTED AND CLOSED.**
   Margin barely correlates with any volatility measure (`sigma_remaining`
   -0.103, `rv_15` -0.060, `spread` +0.159, `abs_z_score` -0.196), and
   residualising BOTH margin and edge on `sigma_remaining + rv_15 + spread`
   leaves the relationship intact and slightly stronger:

   ```
   corr(margin, edge) raw                 -0.2506  [90% -0.295 .. -0.202]
   corr(margin, edge) residualised on vol -0.2603  [90% -0.307 .. -0.215]
   ```

   So this is not a fast-market artifact.
4. One arm, 1,791 orders, ~2 months.

## To settle it

Store per-trade rows (timestamp, price, side) rather than minute aggregates, at
least for traded windows, and compare prints inside `[decision, decision+60s]`
against the resting limit. The venue still serves them — `market_trades`
returns individual prints with `created_time`; the backfill aggregated them
away.
