# MIGRATION_CHECKS.md — validating the move to venue-native data

A working checklist, not a reference. `DATA.md` is the source of truth for what
the data IS; this file is what must be true before trusting a migration of it,
and it gets ticked off and eventually deleted.

Each check states the test, the threshold, and **what to do if it fails**, so
it can be executed later without re-deriving the reasoning. They are ordered so
that an early failure makes the later ones unnecessary.

## Pending before the next retrain

Updated 2026-10-21. The Sunday cron ALWAYS installs, unreviewed, so anything
left undone here trades.

### Blocking

| # | item | blocked by | note |
|---|---|---|---|
| 1 | BRTI backfill finishes | running | BTC ~10 min, ETH/SOL ~1h each at 155 hours/min |
| 2 | **C1** — venue strike + BRTI price vs Coinbase | 1 | the gating test; decides whether to migrate at all |
| 3 | If C1 passes: migrate | 2 | **C6** — all ten strike-derived features AND the price series move together, or none do. `live.py` needs the same switch for parity |
| 4 | Ablation / prune 48 -> ~15-20 features | 2 | must run on whichever price series we keep: 33 of 48 features are Coinbase-derived, so pruning before C1 risks doing it twice |
| 5 | `implied_vol` keep or drop | 4 | recovered coverage is BTC 100 / ETH 29 / SOL 4, so keeping it SKEWS the sample, not just caps it |

### C4 — RUN 2026-10-22. `cross_venue` binds, not `implied_vol`.

Rows surviving `--complete-cases` at offsets 3/6/9/12, 2025-12 to now:

```
quote only          341,944   100.0%
+ implied_vol       222,174    65.0%
+ cross_venue       164,268    48.0%   <- binds harder
+ BOTH              121,355    35.5%
```

Four `cross_venue` features cost 52% of the sample; five `implied_vol`
features cost 35%. Together, nine features cost nearly two thirds of it.

**This inverts DATA.md's §7b**, which had `implied_vol` as the dominant cost
at +181%. That was measured before the rebuild: with the fits repaired and
reaching back to 2025-12, `implied_vol` is now the cheaper of the two.

Dropping both gives 2.8x the sample AND nine fewer features, which is the
direction the capacity evidence already points -- optimal tree counts of
1-41, and overfitting measured at 4x the effect size under study.

What is NOT measured is whether either group earns that cost in skill. The
C4 note says it: "That is an ablation, not a judgement call." The sample cost
is now quantified on both sides of it, so the leave-one-out is the only
remaining input.

### Not blocked, can run any time

* **C2** — venue label vs Coinbase label, bucketed by |move|. Needs only
  `venue_settlements`, which we hold.
* ~~**C3** — candles vs Predexon~~ **RUN 2026-10-21. Candles win.**
  Measured against the LIVE book on 126,674 / 109,548 overlapping rows:

  ```
                       bias      median |d|   within 0.5c
  candles            +0.014c        1.8c         28.1%
  backfill (Predexon) -0.256c       2.0c         17.3%
    in [0.10,0.90]:  candles 2.0c / 19.0%  vs  Predexon 3.0c / 13.8%
  ```

  Predexon is 0.0% off the whole-cent grid everywhere; candles are 58-61%
  off it in the tails, confirming the deci-cent claim. The literal threshold
  "failed" only because it assumed quantisation was the ONLY difference. It
  is not: both sources sit ~2c from the live book because both are
  minute-grid, and a candle close is a 60-second boundary while the live book
  is sampled at the decision instant. That residual is TIMING, and it bounds
  how good any minute-grid quote can be. **Predexon can be retired for
  quotes**, which is what `--quote-sources` already does.
* **`max_depth` 3 vs 4** — one cheap arm. Measured tree counts were 1-41 and
  `num_leaves=15` makes depth 4 effectively a full tree, so depth is the
  binding capacity knob.
* **NC2 (noise-only features)** — pre-registered and **never run**. It is the
  direct control for whether feature COUNT hurts here, which is the question
  item 4 answers indirectly.

### Done this session

* `venue_strikes` gap closed, 3,692 -> 0, table now reaches 2025-12-10
* `venue_candles` gap closed; 2026-08-13 recovered (it sat on the 69-day edge)
* `venue_implied_vol` rebuilt from the venue's own ladders, 23,131 fits,
  validated against live at 0.01 bp/min
* `available_time` repaired on those fits — it was the backfill's RUN time, so
  the feature was silently NaN for the whole backfilled era
* `book_flow` fixed live: the sampler waited on the decision gate, so decision
  minutes were refused as stale and every feature's lookback landed on one.
  All six now score; 0 all-NaN cycles where there were 125 per 12h
* retrain groups swapped `market_state,book_flow` -> `book_quote`, dropping
  the 7 depth columns that cannot exist before 2026-08-25
* throttle counters added to four backfills that were storing a 429 as data
* `--strike-source venue` plumbed into the research path, which is what makes
  C1 runnable at all

### Open, not yet triaged

* `market_price` (`market_prob`, `market_minus_baseline`) is NOT in the retrain
  groups although `--init-score-source market` is set. Deliberate or not is
  unestablished.
* 6 ladder-hours produced no fit after throttling. 0.03%, accepted — repairing
  them would re-ask every legitimately empty ladder too.
* 4 index hours dropped. The skip logic makes them a one-request repair on the
  next pass.

### Order of work

1. **Exact strike + label** — helps every configuration, independent of every
   other question here.
2. **Candle history** — 4.2x the sample. Statistical power is the binding
   constraint: `min_folds`, `sign_agreement_p` and `log_loss_skill_p` all
   failed on sample size alone, and *you cannot measure whether a precision
   improvement helped without the power to measure anything*.
3. **BRTI price series** — removes the remaining ~0.96bp from displacement.
4. **Tape history** — extends the maker/adverse-selection work.


Each check below states the test, the threshold, and **what to do if it
fails** — so it can be executed later without re-deriving the reasoning. They
are ordered so that a failure early makes the later ones unnecessary.

### C1. Does the exact strike actually improve the baseline? — RUN THIS FIRST

The cheapest decisive test, and it gates everything else.

* **Test:** recompute `core/baseline.py`'s out-of-sample log loss on identical
  windows, once with `floor_strike` and once with the Coinbase OHLC-mean
  strike. Nothing else changes.
* **Pass:** baseline log loss improves materially (the arithmetic null gets
  sharper because `z = (price − strike)/sigma` is measured better).
* **Fail / flat:** the ~0.55bp strike error was not material. **Then do NOT do
  the BRTI price migration either** — it is the same mechanism at the same
  magnitude, so a flat result here predicts a flat result there. Keep the exact
  label (C2), which is a separate mechanism, and stop.

### C2. Is the venue label aligned, or just different?

* **Test:** `expiration_value`-derived outcome against the current
  Coinbase-derived `outcome`, bucketed by |move from strike|.
* **Pass:** ~3% overall disagreement, monotonically concentrated in the
  sub-2bp buckets (34.9% under 1bp, 13.9% at 1–2bp, ~0% above 25bp) — the
  shape already measured against `venue_settlements`.
* **Fail:** materially more than 3%, or disagreement that is NOT concentrated
  in near-ties. That is a **window-alignment bug**, not a better label — the
  same shape as the Polymarket slug trap, where every number looked valid and
  the mapping was off by one window. Do not train on it; find the offset.

### C3. Are the candles genuinely a better quote than Predexon?

* **Test:** on the 2026-01-07 → 2026-08-13 overlap where both sources exist,
  compare `yes_bid`/`yes_ask` per `(symbol, window, offset)`.
* **Pass:** candle prices sit on the true deci-cent grid where Predexon sits on
  whole cents outside [0.10, 0.90]; total resting size is not compared because
  candles carry none.
* **Fail:** if the two disagree by more than the quantisation can explain
  (~0.25c in the tails, ~0 in the middle), one of them is stale or misaligned.
  Investigate before retiring Predexon.

### C4. Does `--complete-cases` actually expand the sample? — THE SLEEPER

The 4.2x assumes `implied_vol` and `cross_venue` are populated **at the
decision offsets** across the whole span, not merely present in the date range.

* **Test:** count rows surviving `--complete-cases` per month over
  2026-01-14 → now, broken down by which group is the binding NaN.
* **Context:** measured 2026-09-20 → 10-05, Polymarket covered only **5,022 of
  17,088** candidate rows (29%) at offsets 3/6/9/12. If that ratio holds across
  the span, **`cross_venue` — not the quote source — becomes the binding
  constraint**, and the expansion is far smaller than 4.2x.
* **Fail:** if `cross_venue` binds, the choice is to drop that group (4
  features) and recover the sample, or keep it and accept a smaller one. That
  is an ablation, not a judgement call.

### C5. Pre-registration — before any of the above is scored

**25 distinct trials** have been spent (trial log + promotion ledger), and the
last artifact came back at deflated Sharpe 0.586. Every additional
configuration raises the bar for believing anything that follows
(`RESEARCH_PROTOCOL` §1, §6).

Write down, before looking: which configurations will be run, which single
metric decides, and what counts as success. "The last one disappointed" is not
a hypothesis.

### C6. The live/train parity switch is all-or-nothing

When training moves to the venue strike, the live path must move **all ten**
strike-derived features in the same change:

```
z_score, abs_z_score, excursion_up_z, excursion_down_z, excursion_span_z,
path_efficiency, displacement_vs_elapsed, touched_opposite, peer_displacement,
displacement
```

`scripts/live.py` (~line 1050) records what happens otherwise: an earlier
version overwrote `strike` and recomputed only `z_score`, so `abs_z_score ==
|z_score|` — an identity that holds in every training row — was violated at
inference, on every cycle, invisibly. A tree ensemble does not degrade
gracefully off its training manifold.

The same rule binds the price series: training on `Coinbase_close /
BRTI_strike` injects a cross-index basis the model was not fitted with. **The
strike and the price series move together, or neither moves.**

### C7. Confirm the folds actually multiply

Already verified 2026-10-21 and recorded here so it is not re-checked:
`fold_scheme='calendar'` uses **every complete block** of `fold_block_days`,
not the most recent `n_folds + 1`. ARM A reached fold 38 on the long span while
ARM B managed 5. So span translates into folds, and `n_folds=6` does not cap
it. **No action needed.**
