# Decision rule — the 2,000-window verdict

**Written 2026-08-25 00:35 UTC, before the data exists.** At the time of writing:
402 scored symbol-windows carry both a recorded quote and a settled outcome, and
`model_minus_market` reads **−0.098359** on 285 of them. 247 trades have settled.

The point of writing this now is that after the number arrives, every threshold
looks negotiable. This project has a documented history of the failure it
prevents: a 27-cell survey whose highest-scoring cell was its own control, and a
promotion ledger that exists to count trials because a system tuned until its
curve looks good has measured nothing.

---

## The sample

**Forecast test — every scored row.** From 2026-08-23 15:00 UTC to whenever the
count reaches 2,000 distinct `(symbol, window_open)` pairs carrying a recorded
`market_ask_up`/`market_ask_down` and a settled `outcome`. Nothing is excluded.
The scoring path has not changed since trading began, so the whole span is one
configuration.

**Economic test — trades from 2026-08-25 00:00 UTC only.** Two parameters changed
on 2026-08-24: `min_edge_pp` 0.5 → 1.5 and `max_positions_per_window` 2 → 3. Both
change *which* rows trade, so trades either side of that are not one sample.
Everything before it is excluded from the economic test and stays in the record.

This asymmetry is deliberate: the forecast test does not care whether a row
traded, so it keeps the full span; the economic test cares about nothing else, so
it starts over.

## The two tests

**1. Forecast: `model_minus_market`** from `scripts/market_benchmark.py`, the
pooled `all` row. Positive means our probability is a better forecast than the
price we would have to pay.

**2. Economic: realised P&L against the market-is-right null.** For each settled
trade take its actual price, contracts and fee; take the market's de-spread mid
`(ask_up + (1 − ask_down))/2` as the true probability; simulate outcomes with a
Gaussian copula sharing a per-window factor at **rho = 0.7** (the measured
cross-symbol correlation of settle direction is +0.618; 0.7 is the conservative
end). Report `P(net ≥ actual)` one-sided. **Significant means p ≤ 0.05.**

Both numbers get reported whatever they say.

## The decision

| economic | forecast | action |
|---|---|---|
| p ≤ 0.05, positive | `≥ 0` | **Continue, and size up.** Both tests agree. Raise the stake only after `sharpe_implausible` and the promotion gates are re-read on the live sample. |
| p ≤ 0.05, positive | `< 0` | **Continue at current size. Do not size up.** This is "wins bets, loses log loss" — real but unexplained, and an unexplained edge is one you cannot tell has stopped. Priority becomes finding the mechanism. |
| p > 0.05 | `< 0` | **Stop the taker strategy.** Both tests fail. Move to the pivot list below. |
| p > 0.05 | `≥ 0` | **Extend once, to 4,000 windows.** One extension, decided now, no further. If it is still ambiguous at 4,000, treat as stop. |

## What is not allowed

* Re-running either test with a different threshold, sample window, or exclusion
  after seeing the result.
* Adding an exclusion ("ignore the bad week", "ETH was broken") not written here.
* More than the one extension in the bottom row.
* Changing `min_edge_pp`, `max_positions_per_window`, the offsets, the feature
  set, or the model during the economic sample. Any such change **restarts the
  economic sample** and must be recorded here with its date. Bug fixes to the
  execution path are exempt and do not restart it.

## If it stops

Ranked by how much they change, not by how appealing they are:

1. **Make markets instead of taking them.** `maker_fee_rate` is 0.0025 against
   0.07 taker — 28x — and a resting order earns the ~1c spread instead of paying
   half of it. At 30c that is a ~2.4c swing per contract, about 8% of the stake:
   a taker needs ~2pp of edge to break even and a maker needs roughly none. The
   risk is adverse selection, which is severe on a 15-minute binary and could eat
   all of it. Testable against the quote archive before any money moves.
2. **Different information.** Short-dated implied volatility (the barrier needs
   `sigma_remaining`, and implied vol is a market forecast of exactly that);
   sub-minute data (15 one-minute bars is a poor vol estimator for a 15-minute
   window); Kalshi's own book dynamics, which no current feature touches.
3. **Stop.** The perp system was rejected four independent ways and its archive
   exists so it is not re-run. A clean negative on a pre-registered test is the
   method working.

## What is true regardless of the verdict

The execution path, the venue reconciliation, the gate machinery and the
measurement apparatus are strategy-independent and would be needed by any of the
pivots. So is the Kalshi quote archive, which did not exist before 2026-08-23 and
is the input to every option above.

---

# Appendix A — the retroactive forecast test (added 2026-08-25)

**An addition, not an edit. Nothing above this line changes.**

Written before the number exists. Kalshi's own candlestick history turns out to
reach back to the series' origin, so `model_minus_market` can be computed on
~19,353 symbol-windows today instead of the ~2,000 the body of this document
waits for. That is a better test on sample size and a worse one on calendar span,
and both facts are fixed here in advance.

## Why it is a real out-of-sample test, and why that cuts both ways

The deployed artifact is fold 5's model, trained through **2025-12-05**. The three
series began **2026-06-17**. So every backfilled window is unseen by it.

But the artifact was **force-promoted**: `models/promotions/20260823T144827Z.json`
records `forced: true`, `passed: false`, failing `sharpe_implausible`, with the
reason *"live smoke test; edge not established, see AUDIT_REPORT.md"*. So:

* **A negative result confirms what the promotion record already says.** Weak
  news, and no reason to re-examine the pipeline.
* **A positive result is surprising** — a smoke-test artifact nobody claimed an
  edge for, beating the market over 69 days of unseen data. **The first
  hypothesis is a bug in the backfill, not an edge**, and §"Bug or result" below
  fixes what must be checked before it is believed.

## Sample

* **Markets:** every settled market in `KXBTC15M`, `KXETH15M`, `KXSOL15M`,
  enumerated via `GET /markets?series_ticker=…&status=settled` following the
  cursor to exhaustion. `open_time` comes from that response and **never** from
  parsing the ticker — the ticker names the close in US Eastern
  (`KXBTC15M-26JUN172000-00` opens 2026-06-17T23:45Z).
* **Offsets:** 3, 6, 9, 12 minutes after `open_time`. The quote is the
  candlestick whose `end_period_ts == open_ts + offset*60`, `period_interval=1`.
  `end_period_ts` is the *inclusive end*, so that candle's close is the state at
  exactly the offset minute — the same instant a live decision would see, with no
  lookahead. A candle ending at `open + (offset+1)*60` would leak and must not be
  substituted when one is missing.
* **Date bounds:** `open_time` in [2026-06-17T23:45:00Z, the backfill's run
  date). The lower bound is the series origin; there is nothing before it (404 on
  the preceding slot, the preceding day, and May).
* **Market probability:** the de-spread mid,
  `(yes_bid.close + yes_ask.close) / 2`, from the same candle.

## Exclusions, decided now

A row is excluded **only** when a two-sided de-spread mid cannot be computed:

1. the candle at that offset is absent;
2. either side is missing;
3. `yes_ask.close >= 1.0` **and** the implied spread exceeds 5c — the venue's
   no-offer encoding, observed as `ask 1.0000` against a `0.53` bid at a market's
   open. A tight `0.999/1.000` book is a real near-certain market and is **kept**;
4. the resulting mid falls outside (0, 1).

**Nothing else.** In particular **early SOL is not excluded for thinness.** Its
first-day volume was 232 against BTC's 78,128, and its spreads were 3–4c against
BTC's 1–2c. It is reported as its own row in the coverage report and in the
result, and it stays in the pooled number. An exclusion invented after seeing SOL
drag the average is precisely the failure this document exists to prevent.

Total exclusions must be reported as a count and a share. **If exclusions exceed
5% of rows the test is void** and the reason must be diagnosed before re-running.

## Significance

Fold dispersion cannot carry an inference here — over 69 days the six folds span
~11.4 days each and `5 of 6 positive` is a **34.6%** event at ρ = 0.7 (6 of 6 is
22.4%). And the breadth formula `N/(1+(N−1)ρ)` is rejected by name in
`core/metrics.py`. Neither available method works, so:

**Circular block bootstrap over whole UTC days.**

* Resample whole UTC days with replacement to the original day count, pool all
  rows in the drawn days, recompute pooled `model_minus_market`.
* A whole day carries all three symbols and all four offsets together, so
  cross-symbol and cross-offset correlation are absorbed **structurally** rather
  than assumed at some ρ. Intra-day chaining — one window's strike is the
  previous window's settlement — is preserved inside a block; only the midnight
  boundary is cut.
* **Circular**, not moving: a moving-block scheme samples interior days more
  often than the endpoints. With ~69 blocks the bias is small, and it is free to
  avoid by wrapping.
* **Block lengths 1 day and 5 days, both run.** 1 day gives ~69 blocks and may
  understate serial dependence across days; 5 days captures more of it at ~14
  blocks, which is thin and will widen the interval. **The more conservative of
  the two governs** — decided now, so the block length cannot be chosen after
  seeing which is kinder.
* **10,000 resamples, one-sided.** `p` = the share of resamples with
  `model_minus_market <= 0`. **Significant is p <= 0.05.**
* Report the point estimate, both intervals, both p-values, and the day count.
  **Fewer than 30 usable days voids the test.**

## What this test may and may not do

It **substitutes for the forecast test** in the body of this document, on roughly
ten times the sample. If it is decisive, the body's forecast test is redundant and
the two must not be pooled or best-of'd.

It **cannot touch the economic test**, and the reason is stronger than "no book
history". Verified: `GET /markets/{ticker}/orderbook` on a settled market returns
`{"orderbook_fp": {"no_dollars": [], "yes_dollars": []}}` — no depth, no ladder,
no queue position, ever. So the backfill cannot speak to fills, slippage, or
**fill selection**, and that last one is not hypothetical: 30% of intended orders
did not fill live, and the ones that failed carried a *higher* claimed edge than
the ones that filled. A forecast test cannot see any of that. The economic test
stays exactly as the body defines it, on live-recorded data, and its verdict is
not affected by anything here.

One measurement difference, recorded now so it is not discovered later: the
backfilled quote is a **candle close** at the offset minute, while a live-recorded
quote is sampled at the **instant of the decision call**. Both are "the book at
offset m" but they are not byte-identical, and a small systematic difference
between the two samples is expected rather than alarming.

## Outcomes

| result | reading |
|---|---|
| `p <= 0.05`, positive | Surprising, per the force-promotion above. **Verify against the bug checklist before believing it.** Survives that → the forecast leg is met, and the economic test still governs whether it pays. |
| not significant, positive point estimate | No forecast edge demonstrated. The body's economic test still runs to term. |
| negative | Confirms the promotion record. The forecast leg fails; go to the body's pivot list. |

## Bug or result

Before any positive result is believed, all of these must hold, and the check is
part of the test rather than a follow-up:

1. **Timestamp alignment.** Shifting the offset by ±1 minute must *change* the
   answer. If it does not, offsets are not doing what the code thinks.
2. **A deliberately wrong offset must lose.** Scoring the model against the
   candle at `open + 14m` — nearly settled, so the market is nearly always right
   — must produce a strongly negative `model_minus_market`. If that comes out
   positive, the pairing is broken.
3. **Outcome polarity.** The market's own log loss on the same rows must be
   *better* than a 0.693 coin flip and better than `F(x/sigma)`. If the market
   scores worse than a coin flip, outcomes are inverted somewhere.
4. **No row may use a candle later than its offset.**
5. **The base rate must sit near 0.50** per symbol, as it does on Coinbase bars
   (0.5009 BTC / 0.5031 ETH).

## What is not allowed

Everything the body forbids, plus: no changing block length, resample count,
exclusion list or date bounds after the number is seen; and no re-running this
test with a different artifact to find one that passes. If a second artifact is
ever tested, it counts as a second trial and is recorded in the promotion ledger
like any other.
