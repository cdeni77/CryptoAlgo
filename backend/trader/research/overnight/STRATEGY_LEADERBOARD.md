# Strategy leaderboard

Ranked by **evidential strength**, not return. Every candidate tested appears,
including failures. `model_minus_market` is window-weighted (the conservative
statistic) unless noted.

Benchmark: the Kalshi mid at the decision instant. Noise floor from this project's
permutation controls: **≈ −0.001 to 0.000**.

## Tier A — order-flow tier (2026-09-30 → 10-14, 1,317 windows, 987 evaluable)

| # | candidate | features | mmm | 95% CI | verdict |
|---|---|---|---:|---|---|
| 1 | **S3 LightGBM flow survivors** | 6 | **+0.000661** | [−0.000268, +0.001563] | best overall; CI includes 0 |
| 2 | S3 XGBoost flow survivors | 6 | +0.000579 | [−0.000350, +0.001475] | tied with #1 |
| 3 | S2d LightGBM all flow | 14 | +0.000334 | [−0.000608, +0.001286] | weaker than the subset |
| — | *NC permuted s0* | 6 | *+0.000001* | *[−0.000290, +0.000299]* | *control — clean* |
| — | *NC permuted s1* | 6 | *−0.000094* | *[−0.000526, +0.000348]* | *control — clean* |
| 4 | S2b logistic, imbalance only | 1 | −0.002964 | [−0.006749, +0.000717] | linear fit hurts |
| 5 | S2a logistic, survivors | 6 | −0.003368 | [−0.007567, +0.000676] | worst of the flow arms |

## Tier B — previous campaign, `market_v2` (2026-07-01 → 10-14, 7,932 windows)

**Fitted under two different protocols — not directly comparable to each other.**
Arms A/B/C use chronological early stopping; E-series does not, which the campaign
measured as worth ~0.004 nats.

| # | candidate | protocol | mmm | 95% CI |
|---|---|---|---:|---|
| 1 | A LightGBM, deployed config | early stop | +0.000195 | [−0.000072, +0.000458] |
| 2 | C XGBoost tuned (6 cfg) | early stop | +0.000195 | [−0.000095, +0.000476] |
| 3 | E4 logistic market residual | none | +0.000122 | [−0.000649, +0.000896] |
| 4 | B LightGBM tuned (6 cfg) | early stop | −0.000057 | [−0.000492, +0.000361] |
| 5 | E1b Platt recalibration | n/a | −0.000379 | [−0.001347, +0.000521] |
| 6 | E8b book-only (venue_depth) | none | −0.000836 | [−0.001334, −0.000341] |
| 7 | E1 isotonic recalibration | n/a | −0.001740 | [−0.002853, −0.000682] |
| 8 | E5 LightGBM bars | none | −0.003791 | [−0.005174, −0.002439] |
| 9 | E8 book features | none | −0.004039 | [−0.005429, −0.002659] |
| 10 | E11 cross-venue (Polymarket) | none | −0.004980 | [−0.006495, −0.003495] |
| 11 | E3 barrier baseline alone | n/a | −0.005150 | [−0.006633, −0.003677] |
| 12 | E7 implied vol | none | −0.006750 | [−0.008612, −0.004960] |
| 13 | E2 isotonic per offset | n/a | −0.007650 | [−0.010698, −0.004942] |

## Rejected hypotheses

| hypothesis | result |
|---|---|
| **H1** touch depletion | all depletion/replenishment signals null at Stage 1 |
| **H3** liquidity withdrawal | `d_total_size`, `d_touch_size`, `d_levels`, event rate all null |
| **H6** entry-time selection | negative realised edge at **every** offset, both sides, CIs excluding 0 |
| **H8** tight-spread bias | Simpson's paradox: **−1.74pp row-weighted** vs +4.63pp window-weighted |
| **H4/H5/H7** | not reached within budget — see NEXT_RESEARCH.md |

## Trading performance of the leader (simulated fills only)

| gate | trades | P&L | pp/ct | t | win% | max DD |
|---|---:|---:|---:|---:|---:|---:|
| 0.5pp | 752 | −$1.06 | −0.141 | −0.07 | 69.1 | $14.54 |
| 1.0pp | 627 | +$2.10 | +0.335 | 0.16 | 70.2 | $14.40 |
| 1.5pp | 475 | +$6.64 | +1.399 | 0.60 | 71.8 | $8.95 |
| 2.5pp | 186 | +$5.31 | +2.854 | 0.90 | 76.3 | $4.57 |

**All simulated.** The only realised P&L in this project is the venue ledger's
+$17.53 over 6,692 contracts, which is indistinguishable from zero (t=0.171,
CI [−$181, +$220]). The two are never summed.
