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
