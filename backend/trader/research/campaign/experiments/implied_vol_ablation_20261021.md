# Does `implied_vol` earn ~10,000 windows? — pre-registered 2026-10-21

**Why this is asked, and why it is NOT the book-shape question.**
`implied_vol`'s five features (`iv_minus_realised`, `implied_sigma_per_min`,
`iv_r2`, `iv_n_strikes`, `iv_staleness_minutes`) are fitted from the KXBTCD
strike LADDER — prices, not touch sizes. So unlike the seven
`book_flow`/`market_state` size features, they are recoverable in principle
over the long history.

The problem is COVERAGE, not recoverability. Measured 2026-10-21, the fraction
of (symbol, window) with an IV fit in the same hour:

```
2026-01  22%   2026-04  28%   2026-07  28%
2026-02  25%   2026-05  19%   2026-08  43%
2026-03  21%   2026-06  13%   2026-09  68%   2026-10  61%
```

Under `--complete-cases` a NaN in any selected feature drops the whole row, so
keeping `implied_vol` caps the long-span sample at roughly a quarter of the
pre-August windows: **~16,600 usable windows with it, ~27,000 without.**

**Hypothesis, stated before the run.** `implied_vol` contributes no skill
distinguishable from zero against the market probability, and therefore does
not justify ~10,000 windows.

**Test.** `scripts.evaluate`, identical to `ops/retrain_arms2.sh` ARM A except
`--groups` drops `implied_vol` ALONE. `book_flow` and `market_state` are kept,
so this isolates the one group; the concurrent book-shape trial isolates the
other two.

**Metric that decides.** `model_minus_market` and `log_loss_skill` against ARM
A2 (+0.000383 and +0.000399 respectively), read against fold dispersion of
order 0.00026.

**Prior evidence.** [[project-iv-live-vs-backfill]]: the implied-vol gate stays
at 0.90 because lowering it was tried and reverted — live and the backfill
never selected the same strikes. So the group already has a known
source-comparability problem, independent of coverage.

**This is trial 27.** 25 were spent before today; the book-shape ablation is 26.
