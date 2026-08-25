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
