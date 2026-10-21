# Do the book-SHAPE features earn their place? — pre-registered 2026-10-21

**Why this is asked.** Sizes and depth before 2026-08-25 do not exist and
cannot be recovered (the venue serves no historical order book; Predexon's
touch is minute-stale — size ratio 1.22, correlation 0.26). So 7 of the
artifact's 55 features cannot extend over the long history:

```
book_flow     imbalance_change_3, depth_trend_3, imbalance_persistence
market_state  imbalance_touch, imbalance_5c, depth_ratio, book_convexity
```

If they contribute no measurable skill, the choice between "4.2x sample" and
"all features" disappears.

**Hypothesis, stated before the run.** The book-shape features contribute no
skill distinguishable from zero against the MARKET probability.

**Test.** `scripts.evaluate`, config identical to `ops/retrain_arms2.sh` ARM A
in every respect except `--groups`, which drops `book_flow` and
`market_state` — 11 features, a deliberate SUPERSET of the 7, so the result
bounds the loss from above.

**Metric that decides.** `model_minus_market` and `log_loss_skill`, read
against ARM A2 on the same configuration.

**Success.** Difference within the uncertainty already measured on these runs
(ARM A2's log_loss_skill was +0.000399 with p=0.0657, so fold dispersion is of
order 0.00026). A drop inside that band means the groups are dispensable.

**Prior evidence.** ARM C removed all six `book_flow` features and cost
**0.00004** of log-loss skill — noise — while IMPROVING `model_minus_market`
(+0.000398 vs +0.000383) and `realised_edge_lcb` (+0.170 vs +0.040). The four
`market_state` imbalance features are untested.

**This is a new trial.** 25 were spent before it.

---

## WITHDRAWN and re-run, 2026-10-21

**The testbed was the thing under suspicion.** 67% of the sample (14,743 of
22,099 windows) predates 2026-08-25, where the imbalance features are computed
from the Predexon backfill — measured the same day at size ratio 1.22 and
correlation 0.26 against the live book, because its minute-stale touch sits at
a different price level and "size at the touch" is therefore a different
quantity.

So a null on those features could mean either "they do not help" or "we fed
them noise for two thirds of the run", and the design cannot separate the two.
Testing the value of book-shape features on data whose book shape we had
already discredited is circular.

**Re-run restricted to 2026-08-25 onward**, where every size comes from the
live book, and compared against ARM B2 (same quote sources, same era, all nine
groups, 4,823 windows). The cost is power — ~7,000 windows and far fewer
folds — but a weak measurement of the right quantity beats a precise
measurement of the wrong one.

## Attempt 4, also abandoned — and the reason is the protocol, not the data

Relaunched on the live era with `--peer-max-quote-age-seconds 30` to recover
the rows the 5-second peer cap was cutting. Killed before it produced a number.

The cap was loosened **because the run errored**, not because 30s is the right
cap, and it is not what the account trades. That is `RESEARCH_PROTOCOL` §10
exactly: "Do not keep modifying features, architecture, datasets, evaluation
boundaries or thresholds until something positive appears." A result from a
configuration chosen to make the error go away is not evidence about the
configuration we run.

**Four attempts, no answer. The honest state is in `DATA.md` §7b: this question
needs calendar time, not another configuration.** Do not attempt a fifth
without new live-book history.
