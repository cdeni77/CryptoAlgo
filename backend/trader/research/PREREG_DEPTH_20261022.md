# Pre-registration — max_depth 3 vs 4, 2026-10-22

Written **before the run**. One hypothesis, not a survey.

## The question

`max_depth` is the binding capacity knob. `num_leaves=15` and a full depth-4
tree holds 16 leaves, so depth 4 is effectively unconstrained by `num_leaves`
while depth 3 caps the tree at 8 leaves. The arm is therefore **8 leaves
against 15**, everything else held.

## Why now

NC2 showed that adding eleven information-free columns did not hurt and may
have helped, with the tree count falling 80 to 64. That points at
regularisation strength rather than feature count, and depth is the most
direct regularisation knob that has never been tested here.

## Configuration

Identical to the ablation, so the two are comparable:

```
--groups cross_venue,geometry,vol_state,microstructure,cross_asset,offset,
         book_quote,implied_vol
--venue cfbenchmarks --strike-source venue --label-source venue
--start 2025-10-01 --complete-cases --entry-offsets 3 6 9 12
--init-score-source market
--quote-sources live_touch,live_ws,live,live_ws_replay,candles
--peer-quote-sources all
```

## The metric

`delta = skill(max_depth=4) - skill(max_depth=3)`, scored against the market
because `init_score_source='market'`, **paired fold by fold**. Both arms score
the same rows and the same folds.

## What counts as a result — fixed now

Depth 3 is preferred over the shipped depth 4 only if **all three** hold:

* `delta < 0` — the shallower tree scores better, and
* `|t| > 2.0` — one pre-registered hypothesis, so no Bonferroni, and
* `|delta|` exceeds the floor **measured for an arm of this kind**.

**CORRECTED before running.** This said `|delta| > 0.000152`, NC2's
feature-group figure. That is the wrong yardstick and it is too wide. A
feature-group delta compares arms holding DIFFERENT feature sets; a
hyperparameter arm moves one scalar and shares everything else, which is far
tighter -- a smoke run of this arm reported `se 0.000029`, roughly 3x more
precise than any feature-group delta. Reusing the feature floor would mark a
genuinely resolvable depth result UNRESOLVED.

The floor for this comparison comes from the **seed arm**: `model_seed` across
several values, where the true effect is zero by construction, so the spread
of those deltas IS the floor. That arm must run first and this one is blocked
on it. A result inside the measured band is recorded as **UNRESOLVED**, not as
a null and not as a win.

## Recorded before running

The noise band is a yardstick from a SINGLE draw (`se 0.000104`), not a
constant. A multi-seed calibration would replace it with a distribution and is
the better test; it is deferred, and this threshold is deliberately
conservative because of that.

**Expected outcome, stated in advance:** unknown, and deliberately not
guessed. The earlier draft predicted UNRESOLVED on the strength of the
feature-group band, which has since been established as the wrong yardstick
for this arm. The smoke run on a narrow sample gave `-0.000081` at `t -2.80`
with 9 of 10 folds favouring depth 3 -- consistent, and possibly resolvable
once the right floor is known.

**Also fixed before running:** every run on 2026-10-22 used
`ensemble_size=1` while the retrain ships 5. This arm runs at 5.

---

## The floor is measured — seed calibration, 2026-10-22

`model_seed` 17/18/19/20/21, everything else held, at the deployed
configuration (migrated basis, ensemble 5, seven groups). The true effect is
zero by construction.

```
seed      17        18        19        20        21
skill  +0.000693 +0.000692 +0.000685 +0.000700 +0.000701

paired deltas against seed 17
  18  -0.000000  se 0.000010  t -0.04   20/40
  19  -0.000007  se 0.000014  t -0.54   21/40
  20  +0.000008  se 0.000022  t +0.34   22/40
  21  +0.000008  se 0.000024  t +0.36   21/40

spread of the four deltas: sd 0.000007
```

Nothing is significant, which is what a zero effect should look like.

**The floor for a one-scalar arm is `0.000007`.** Five times tighter than the
feature-group floor (`0.000034`) and twenty times tighter than the
ensemble-1 figure (`0.000152`) that the first draft of this document wrongly
borrowed. The depth arm runs with `--noise-floor 0.000007`.

### What it says about where the noise lives

Each arm's own standard error is `0.000188`, **27x** the seed spread. So
model stochasticity is a negligible part of the measurement error at
`ensemble_size=5`: the five boosters have already averaged it away, and
raising the ensemble further buys nothing.

**The remaining error is the data and the folds.** The lever on precision is
more windows, not more ensembling and not a different fit. That retires the
"raise ensemble_size" branch of the diagnostic written before this ran.

---

## RESULT, 2026-10-22 — no change, `max_depth` stays 4

```
max_depth=4   skill +0.000693  se 0.000187  t +3.71  29/40  73 trees
max_depth=3   skill +0.000715  se 0.000179  t +4.00  31/40  87 trees

paired delta (3 against 4): +0.000022  se 0.000025  t +0.88  20/40 folds
```

Depth 3 wins on the point estimate and fails the pre-registered `|t| > 2.0`
at **0.88**. **No change.** The installed artifact keeps `max_depth 4`.

Depth 3 grew 87 trees against 73, which is what less capacity per tree
should do, and was positive on 31 of 40 folds against 29. It may well be
slightly better. That is not what the threshold asks, and the threshold was
fixed before any of this was visible.

### The smoke run misled me, and I said so too confidently

A narrow smoke run (one offset, from 2026-08-01) reported `-0.000081` at
`t -2.80` with 9 of 10 folds favouring depth 3, and this document called that
"consistent, and possibly resolvable". On the full sample the effect is
**+0.000022 — four times smaller and the opposite sign**. A wiring check is
not evidence.

### The noise-floor flag is the wrong statistic, and did not fire

`--noise-floor 0.000007` came from the SPREAD of the four seed deltas. The
run therefore called this delta resolved, because `0.000022 > 0.000007`,
while its own `t` said otherwise.

**An sd of four point estimates is not the yardstick for whether one delta is
real; that delta's own standard error is.** Each seed delta carried `se`
between `0.000010` and `0.000024`, and this depth delta carries `0.000025` --
so the depth effect is the size of the noise in an arm whose true effect is
zero by construction. The flag should compare against a delta's own error,
or be removed; as written it can only mislead.

### What both arms agree on

Both score `t +3.7` to `+4.0` against the market null, so the edge is robust
to this knob. With the seed calibration showing fit noise negligible at
`ensemble_size=5`, neither the model's randomness nor its depth is the
binding constraint. **The remaining lever is more windows.**
