# Revised decision gates — set by the user, 2026-10-17

The repository's promotion gates lead with `model_minus_market >= 0`: a candidate
must beat the Kalshi price before its money is discussed. The user has replaced
that objective. The gates that decide now are:

1. **Is it profitable?** Net P&L after the venue fee schedule, at executable prices,
   with a confidence interval and the independent-window count behind it.
2. **Does it trade?** Consistently, across folds — a model that sits out whole weeks
   is not usable regardless of its average.
3. **Is there leakage?** Every feature and every price must have been knowable at the
   decision instant.

Explicitly accepted: **"even if it's slightly worse than the market but profits,
that's fine."**

## Why that is coherent, not a lowered standard

Log loss averages over every row, including the ones a strategy would never touch.
Trading is **selective** — only rows where the model's probability diverges from the
price by more than the fee are acted on. A model can be worse on the global average
and better in the specific states where it chooses to trade. `model_minus_market`
and net P&L are different questions, and for a trading system the second is the one
that pays.

## What this costs, and the compensating requirement

`model_minus_market` is this project's **best leakage detector**, and dropping it as
an objective removes a safety net. It is what exposed the worst bug of 2026-10-17: a
model reporting +4.9pp per contract on only +0.001 nats of forecast improvement. The
magnitude mismatch was the tell, and without the forecast number there would have
been nothing to compare the money against.

**So it is still COMPUTED and REPORTED on every candidate — it just no longer vetoes
one. The PROMOTION decision is made on profitability and trade consistency; the
market-relative gates are information, not a veto, and overriding them with `--force`
plus a written reason is the accepted path.** Specifically:

* a candidate whose realised edge per contract is large while its
  `model_minus_market` is ~0 or negative is **treated as suspect until explained**,
  because that is the signature of lookahead rather than skill;
* every candidate must pass the point-in-time checks (as-of availability on every
  quote, purged chronological folds split on the window, features from the repo's
  own `attach_quotes` path or an equivalent that enforces the same rule);
* negative controls (features permuted against the label/price pair, noise-only
  features) remain mandatory.

## What a promotion looks like under these gates

`scripts/promote.py --force --reason "..."`, where the reason names **every gate
overridden and the measured profitability and trade cadence that justify it**. The
ledger keeps the rejection record either way, which is what makes the trial count
honest. A forced promotion is not a silent one.

The candidate must still show:
* positive net P&L after the venue fee schedule at executable prices, with its CI
  and independent-window count;
* trades in substantially every fold -- and where early folds are silent because
  they trained on almost nothing, the deploy-today analogue is the LAST folds, not
  the average;
* no leakage: as-of availability on every quote, purged folds split on the window.

## The standing hazard this replaces nothing about

Selection bias. Without a market-relative gate, a profitable-looking backtest on
selected trades is easier to produce by chance. Inference still resamples **windows**,
never rows, and the effective sample is ~9,400 windows — not 274,000 rows.
