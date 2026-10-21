# Overnight campaign — summary

Run 2026-10-17. Dataset: frozen `market_v2` (sha256 `3c729b61…`) plus a **new
order-flow tier built tonight** from 1.18B raw Kalshi WebSocket frames.

---

## What genuinely new signals were discovered?

**An order-flow tier that did not previously exist.** `research/overnight/features/`
now holds 15 daily parquets — one row per (market_ticker, minute 0..15) — carrying
the reconstructed top of book plus per-minute flow: size added, size removed,
event counts, level counts, touch size, and totals on both sides. Built in ~11
minutes, cached, resumable. **No feature group in this repository had ever read the
raw tape.**

**Honest limit on what the tape can say.** `venue_book_events` is `orderbook_delta`
only — there is **no trade channel stored**. A negative delta is *size removed*: a
cancellation or a fill, not distinguishable. Every such column is named
`*_removed_ambig` and no claim of "executed volume" is made anywhere.

**Of 22 derived signals screened model-free against the market residual, 6 cleared
an uncorrected 95% CI** (≈1.1 expected by chance):

| signal | stat | 95% CI |
|---|---:|---|
| `imbalance` (full-ladder, not bucketed) | **+0.010204** | [+0.003667, +0.016660] |
| `d_book_mid_3` | +0.006343 | [+0.000912, +0.011365] |
| `d_imbalance_3` | +0.005617 | [+0.001857, +0.009340] |
| `d_book_mid_1` | +0.005030 | [+0.001391, +0.008467] |
| `d_touch_imb_3` | +0.003059 | [+0.000503, +0.005627] |
| `d_imbalance_1` | +0.002780 | [+0.000356, +0.005209] |

Depletion, net flow, replenishment, event rate and level counts were all **null** —
H1 (touch depletion) and H3 (liquidity withdrawal) are not supported. **H2 (imbalance
and its change) is the only order-flow hypothesis that survived screening.**

## What actually improved relative to the market?

**One candidate, and it is not significant under the conservative statistic.**

LightGBM on the six surviving flow signals, market logit as init score, purged
chronological folds, early stopping on an inner split:

| weighting | model_minus_market | 95% CI |
|---|---:|---|
| row | +0.000479 | **[+0.000179, +0.000780]** |
| symbol-window | +0.000537 | [−0.000160, +0.001217] |
| **window (the honest one)** | **+0.000661** | **[−0.000268, +0.001563]** |

For scale: this project's negative controls sit at **≈ −0.001 to 0.000**, and
tonight's permutation controls returned **+0.000001** and **−0.000094** — tight and
clean. So +0.00066 is real signal, roughly **3.4× the best candidate from the entire
previous campaign** (+0.000195), but its conservative CI still includes zero.

**XGBoost on identical data and folds: +0.000579, CI [−0.000350, +0.001475].**
Statistically indistinguishable from LightGBM — which settles the boosting question.

## Which candidate looks most promising after realistic trading costs?

The same one, and it does **not** yet pay:

| gate | trades | net P&L | pp/contract | t | win rate | max DD |
|---|---:|---:|---:|---:|---:|---:|
| 0.5pp | 752 | −$1.06 | −0.141 | −0.07 | 69.1% | $14.54 |
| 1.0pp | 627 | +$2.10 | +0.335 | 0.16 | 70.2% | $14.40 |
| 1.5pp | 475 | +$6.64 | +1.399 | 0.60 | 71.8% | $8.95 |
| 2.5pp | 186 | +$5.31 | +2.854 | 0.90 | 76.3% | $4.57 |

Venue fee schedule, depth-constrained at the touch, one entry per (symbol, window),
held to settlement. **Every t-statistic is below 1.**

## What evidence supports it, and what remains unproven?

**Supports it:** survives permutation controls with near-zero residue; positive under
all three weightings with consistent sign; LightGBM and XGBoost agree; the underlying
signal (`imbalance`) has an independent economic rationale and was screened before
any model was fitted.

**Unproven:** the conservative CI includes zero. The edge **concentrates** — by symbol
only SOL clears zero (+0.001374 vs BTC +0.000205, ETH +0.000038); by offset only
offset 10; by period the second half (+0.001171) carries it against the first
(+0.000079). **The order-flow tier is 15 days** (2026-09-30 → 10-14), 1,317 windows,
of which 987 are evaluable after the first fold is consumed for training. That is
nowhere near enough to settle a 0.0007-nat effect.

## Two hypotheses were killed, and one of them nearly fooled me

**H6 (entry-time selection) — rejected cleanly.** Buying at the ask has negative
realised net edge at **every** offset, both sides, CIs excluding zero throughout:
−2.16pp at offset 3 improving monotonically to −0.99pp at offset 14. Crossing the
spread costs 1–2.2pp and no entry time recovers it.

**H8 (tight-spread bias) — a Simpson's paradox that passed an out-of-sample test.**
Tight-spread (≤1pp) YES appeared to earn **+4.2212pp on development and +5.5907pp on
validation**, held every month, and survived depth constraints to 50 contracts. It is
an artifact of my own statistic:

| weighting | tight-spread YES edge |
|---|---:|
| **row mean** | **−1.7409pp** |
| symbol-window | +3.4695pp |
| window | +4.6251pp |

Windows holding ≤2 tight rows (mean price 0.5366) show +13.49pp and received the same
weight as windows holding many (mean price 0.2241, −3.10pp). **Every price band is
negative row-weighted, and every offset is negative.** Rejected.

The catch matters beyond this hypothesis: it is why Stage 6 re-ran the order-flow
candidate under all three weightings before reporting it. `model_minus_market` is a
*paired* statistic on identical rows, so it is immune — verified, not assumed, with
`corr(rows_per_window, price) = −0.0247`.

## Verdict

**No candidate meets the success bar.** The strongest emerging signal is full-ladder
order-book imbalance and its short-horizon change, worth ≈ +0.0007 nats against the
Kalshi price — real enough to survive controls, too small and too concentrated to
trade on 15 days of data.

## Reproduce

```bash
cd backend/trader
python research/overnight/features/build_orderflow.py          # ~11 min, resumable
python research/overnight/experiments/stage1_screen.py         # 9s
python research/overnight/experiments/stage2_models.py         # 66s
python research/overnight/experiments/stage34_validate.py      # 135s
python research/overnight/experiments/stage5_policy.py         # 41s
python research/overnight/experiments/stage5b_spread_oos.py    # 12s
python research/overnight/experiments/stage6_reverify.py       # 25s
```
