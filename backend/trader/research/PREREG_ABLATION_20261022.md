# Pre-registration — feature-group ablation, 2026-10-22

Written **before any cell is scored**, per `MIGRATION_CHECKS.md` C5 and
`RESEARCH_PROTOCOL` §1 and §6. 25 distinct trials have been spent already;
this adds 9 cells and raises the bar for everything after it.

## The question

Does each feature group earn its place, holding the sample fixed?

It is asked now and not earlier because **C1 has settled the price series**
(venue strike + BRTI beat Coinbase by +0.0033 log loss at t=+5.71, 42 of 50
folds). 33 of 48 features are derived from the price and the strike, so an
ablation run before that would have measured a configuration we are not
shipping.

## Configuration — fixed now

```
--groups  cross_venue,geometry,vol_state,microstructure,cross_asset,offset,
          book_quote,implied_vol
--venue cfbenchmarks --strike-source venue --label-source venue
--start 2025-10-01 --complete-cases --entry-offsets 3 6 9 12
--init-score-source market
--quote-sources live_touch,live_ws,live,live_ws_replay,candles
--peer-quote-sources all
```

The migrated price series, because that is what C1 says we ship. Everything
else matches the deployed retrain.

## Cells — 9, fixed now

`all groups`, plus `all minus G` for each of the 8 groups. Leave-one-out, not
"G alone": the question is whether a group adds something the others do not
already carry. A group can score near zero alone and still matter.

Every cell is scored on the SAME rows — the complete-cases sample of the full
group set — so the comparison isolates the features from the sample they
would otherwise drag along.

## The single deciding metric

**`delta = weighted mean log-loss skill(all groups) − skill(all minus G)`**,
weighted by window count, standard error from fold dispersion. Forecast skill
only. No money: `scripts/ablate` already refuses to report it, and money
numbers from a run this size must not choose a configuration.

## What counts as success — fixed now

A group is **KEPT** only if it clears both:

* `delta > 0` — removing it hurts, and
* `t(delta) > 2.7` — Bonferroni for 8 tests at 0.05 two-sided.

Otherwise it is **DROPPED**. The default is drop: a feature must earn its
place, which is the standing rule in CLAUDE.md and the reason the clock
control exists.

## What each outcome means, written before seeing any of it

* **`cross_venue` fails** — drop it, and the sample goes 164,268 → 341,944
  rows (C4). The cheapest win available and the one we expect most.
* **`implied_vol` fails** — drop it; 222,174 → 341,944, and the BTC skew goes
  with it (recovered coverage is BTC 100 / ETH 29 / SOL 4).
* **Both fail** — 121,355 → 341,944, 2.8x the sample and nine fewer features.
* **A group PASSES** — then its sample cost has to be weighed against its
  skill, which this design does NOT answer. That needs a second comparison on
  differing samples with a common test set, and it will be pre-registered
  separately rather than decided by eye.
* **Everything passes** — the 48-feature set stands, and the parsimony
  argument (optimal tree counts of 1-41, overfitting at 4x the effect under
  study) is wrong for this data. That is a real possible result.

## What would invalidate this run

* Any cell scored on a different row set than the others.
* `--complete-cases` dropped, which would let each arm choose its own sample.
* Reading the money columns.
* Changing the threshold after seeing a t-statistic.

## Recorded before running

Expected sample: ~121,355 rows at offsets 3/6/9/12, 2025-12 onward, the
complete-cases intersection of all eight groups (C4, measured 2026-10-22).
