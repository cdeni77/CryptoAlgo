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
* `|delta| > 0.000152` — **larger than the noise draw**.

The third condition is the one NC2 bought. An information-free feature group
scored `+0.000152` on this data, so an effect smaller than that cannot be
distinguished from an irrelevant one however its t reads. A result inside the
band is recorded as **UNRESOLVED**, not as a null and not as a win.

## Recorded before running

The noise band is a yardstick from a SINGLE draw (`se 0.000104`), not a
constant. A multi-seed calibration would replace it with a distribution and is
the better test; it is deferred, and this threshold is deliberately
conservative because of that.

**Expected outcome, stated in advance:** UNRESOLVED is the most likely result.
Every effect measured on this data so far, real or synthetic, has landed
inside this band.
