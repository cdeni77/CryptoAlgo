# Overnight campaign — hypotheses, ranked before any result was seen

Opened 2026-10-17. Budget: 8 hours, <=14 cores. Dataset: frozen `market_v2`
(sha256 3c729b61…) plus a NEW order-flow tier built from `venue_book_events`.

## Data tiers — stated honestly up front

| tier | source | span | sample |
|---|---|---|---|
| **T1 market+bars** | `market_v2` | 2026-07-01 → 10-14 | 28,594 verified symbol-windows |
| **T2 order flow** | `venue_book_events` ∩ market_v2 | **2026-09-30 → 10-14 only** | **~4,300 symbol-windows** |

T2 is 15 days. Every order-flow result is bounded by that and will be labelled
exploratory with an explicit power statement. **Order flow does not exist for
July–September and will not be imputed.**

**What the tape can and cannot say.** `venue_book_events` is `orderbook_delta`:
snapshots and signed per-level size changes. There is **no trade channel stored**,
so a negative delta is *size removed* — a cancellation OR a fill, not
distinguishable. Every feature below that depends on removals is labelled
`ambiguous_removal` and no claim of "trade" is made from it.

## The hypotheses

| # | hypothesis | economic rationale | data | cost | testable | prior |
|---|---|---|---|---|---|---|
| **H1** | **Touch depletion precedes price moves.** Size removed at the best level, net of replenishment, predicts the next move in the Kalshi mid beyond what the mid already implies. | Informed participants pull liquidity before repricing; the book thins on the side about to move. This is the classic queue-depletion effect and it is *not* in any existing feature group. | T2 | low | yes | **high** |
| **H2** | **Order-book imbalance CHANGE (not level) carries signal.** The archive left imbalance open at t=+0.30→+1.84 over 67 days; the level may be priced but its short-horizon derivative may not be. | A level is observable to everyone and arbitraged; the rate of change over 5–30s is harder to monitor across 288 concurrent markets. | T2 | low | yes | **high** |
| **H3** | **Liquidity withdrawal predicts market overreaction.** When depth collapses, the quote moves further than the eventual settlement justifies — i.e. the market overshoots in thin books. | Price impact per unit flow rises as depth falls, so the same information moves the quote further than its fair-value content. Mean reversion after thin-book moves. | T2 | low | yes | **high** |
| **H4** | **Market-relative miscalibration conditional on STATE, not price.** E1/E2 showed the mid is calibrated unconditionally; test whether it miscalibrates conditional on volatility regime, time-to-settle and recent quote velocity. | A single market maker's model cannot be correct in every regime; systematic deviation should concentrate where their model is most stressed. | T1 | low | yes | medium |
| **H5** | **Passive maker fills capture spread net of adverse selection.** Post a resting order inside the spread; measure realised P&L with a QUEUE-AWARE fill model from the reconstructed book. | Kalshi charges makers 0.25% vs 7%·p(1−p) taker; if adverse selection is below the spread captured, passive is profitable where aggressive is not. | T2 | **high** | yes | medium |
| **H6** | **Entry-time selection.** The current policy walks offsets 3/6/9/12 and takes the first clearing a gate. Test whether a *specific* offset or a state-conditional entry time dominates. | Information arrives unevenly through the window; the barrier is most sensitive to σ error at |x|/σ≈1. Cheap to test and never tested against the MARKET. | T1 | low | yes | medium |
| **H7** | **Cross-venue lead–lag with Polymarket.** Whether Polymarket's quote leads Kalshi's at the 15-minute horizon, after aligning the slug-stamp convention. | Different participants, different oracle, no shared matching engine. Genuine disagreement is information rather than noise. | T1 | medium | yes | low-med |
| **H8** | **Selective abstention.** Rather than a better probability, find states where the model's own error is predictably large and refuse those windows. | Converting a zero-edge forecaster into a positive-edge trader by trading only where its error distribution is tight. Does not require beating the market everywhere. | T1 | low | yes | low-med |

## Order of execution

**Stage 1** screens H1–H3 (order flow) and H4/H6 (cheap T1) statistically before any
model is trained. H5 is the most expensive and runs only if H1–H3 show signal. H7/H8
are backstops.

## Rejection rules, fixed now

* A signal must beat the Kalshi mid **out of fold, on windows**, with a bootstrap CI
  excluding zero, before any model is trained on it.
* The noise floor established by this project's negative controls is
  **model_minus_market ≈ −0.001**; a candidate at or below that is noise.
* A strategy profitable only under full-fill assumptions is **rejected** (H5 carries
  a queue model or it does not count).
* No hypothesis is re-tested after falsification without materially new evidence.
