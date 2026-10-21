# Best candidate — order-flow imbalance residual model

**Status: PROMISING, NOT SUPPORTED.** It does not meet the success bar. It is the
strongest market-relative signal this project has produced and it is worth more data.

## Specification (frozen)

```
dataset      research/overnight/features/flow_*.parquet  (15 files, 2026-09-30..10-14)
             joined to research/campaign/datasets/market_v2.parquet
             sha256 3c729b61cf17967d3a1b4d07d0526dab36affd13487b9c45199f9ceb02facbce
             verified rows only (backfill book_age <= 5s, or live/live_ws)
features     imbalance, d_imbalance_1, d_imbalance_3, d_touch_imb_3,
             d_book_mid_1, d_book_mid_3            (6)
target       settlement (venue's own), with the MARKET LOGIT as init_score
estimator    LightGBM  lr 0.02, num_leaves 7, max_depth 3, min_child_samples 500,
             bagging 0.8/freq 1, feature_fraction 0.8, lambda_l2 20, seed 17,
             500-round cap, 40-round early stopping on a chronological inner split
folds        3 purged chronological, 1-day embargo, split on the WINDOW
```

## Results

| metric | value |
|---|---|
| OOF rows / windows | 34,180 / 987 |
| model_minus_market (window-weighted) | **+0.000661**, CI [−0.000268, +0.001563] |
| model_minus_market (row-weighted) | +0.000479, CI [+0.000179, +0.000780] |
| model_minus_market (symbol-window) | +0.000537, CI [−0.000160, +0.001217] |
| best iterations by fold | [1, 18, 173] |
| XGBoost on identical data | +0.000579, CI [−0.000350, +0.001475] |
| permutation controls | +0.000001 and −0.000094 |
| net P&L @1.5pp gate | +$6.64 on 475 trades, +1.399pp/ct, **t=0.60** |
| max drawdown | $8.95 |

## Why it might be real

The signal is **full-ladder order-book imbalance** reconstructed from the raw tape,
and its short-horizon change. It was screened model-free *before* any fitting and was
the strongest of 22 candidates (+0.010204, CI [+0.003667, +0.016660]). Imbalance has
a standard microstructure rationale — resting size asymmetry precedes repricing — and
the archive left it explicitly open (t=+0.30 → +1.84 over 67 days). It is **not** the
`venue_depth` 1c/5c bucketed version that E8b tested at −0.000836; this is the full
ladder, which that projection discards.

Two estimators agree, every weighting agrees in sign, and the controls are clean.

## Why it might not be

* The conservative CI **includes zero**. `best_iterations [1, 18, 173]` shows one fold
  learning nothing at all.
* **Concentration.** By symbol only SOL clears zero (+0.001374; BTC +0.000205,
  ETH +0.000038). By offset only offset 10 of 14. By period the second half carries it
  (+0.001171 vs +0.000079). With 3 symbols and 14 offsets at 95%, roughly one
  significant cell is expected by chance in each cut.
* **15 days of data.** 1,317 windows, 987 evaluable. The tape has a 14-day retention
  tier, so this cannot be extended backwards — only forwards.
* Every t-statistic in the trading simulation is below 1.
* `d_book_mid_*` is the quote's own recent return, so part of the signal may be
  quote momentum rather than independent information. Not separated.

## What would settle it

Accumulate order-flow data forward. At the current rate (~88 windows/day across three
symbols) reaching ~5,000 evaluable windows takes about **8 weeks**. The specification
above is frozen; it should be scored on that data **without refitting the feature set
or the hyperparameters**.

**Do not trade this.** It is not promoted, not deployed, and the live artifact is
untouched.
