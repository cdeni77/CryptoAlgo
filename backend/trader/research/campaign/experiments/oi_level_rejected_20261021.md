# OI-level feature — REJECTED before fitting, 2026-10-21

**Hypothesis (pre-registered before measuring).** The baseline's
`sigma_remaining` comes from spot price variance and knows nothing about
positioning in the contract. Open interest is the stock of positions; its
change is net new risk being taken, and participants adding exposure late in a
window — when the outcome is largely determined — is either informed
positioning or liquidity provision. Either carries information the spot-derived
sigma cannot see.

**Why this was a NEW hypothesis and not a re-test.** `core/book_flow.py`
records the trade TAPE as rejected and staying rejected (sixteen cells, four
projections, four offsets, orthogonalised against price and price-squared; not
one significant, largest t=-1.90 on 1,800 windows). Open interest is not
trades: it is the stock, absent from that work entirely. It is also live
available — the Kalshi `ticker` websocket channel carries `open_interest_fp`,
verified off the wire 2026-10-21.

**Measured on `venue_candles`, 19,476 markets, 2026-08-13 .. 2026-10-21.**

```
off   median OI   OI/peak          minute-on-minute steps
  1        3400     0.087          rising   86.1%
  3        8165     0.211          FALLING   7.1%
  6       12992     0.332          flat      0.1%
  9       17246     0.438
 12       23397     0.572
 15       45744     1.000
```

**Result: rejected.** Open interest in a 15-minute binary is a monotone ramp
from near zero — 91% of peak OI turns over inside the window and 0% of windows
are flat, not because positions churn but because they ACCUMULATE and settle at
expiry. Early closing is rare (7.1% of steps), which is the same economics that
makes this project's hold-to-settle policy correct.

So **OI level at offset `m` is predominantly a clock**: it encodes how far into
the window we are. `clock` is the mandatory negative control
(`control_gain_share` gates it at 0.30), and a feature that proxies the control
is the failure this repo has already made once — the 27-cell survey whose
best cell was its own control. Fitting it would have been worse than useless:
it would have looked like skill.

`OI change` is no better placed: it is cumulative volume by another name, which
sits inside the rejected tape family rather than outside it.

**What survives, as a DISTINCT hypothesis requiring its own test.**
Offset-normalised participation: `log(OI_m / median OI at offset m)` and the
same for volume. Deseasonalised by construction, so it cannot proxy the clock,
and it asks a different question — is this window accumulating exposure
unusually fast *for this point in its life*? Not yet tested. If tested, it must
carry `clock` in the ablation and clear the five negative controls in §5.

**No model was fitted. No data was consumed for selection**, so the span stays
available for the narrowed hypothesis.
