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


---

## AMENDMENT, 2026-10-22 07:00 — run 1 is VOID, metric corrected

Run 1 is recorded as a trial and discarded, not reinterpreted. Its result:
all eight groups between `-0.000144` and `+0.000309` against a standard
error of `0.000782`, and `all groups` at `+0.002237`. Every group "failed",
which is the tell.

**The metric could not answer the question.** `scripts/ablate` scored
`log_loss(baseline) - log_loss(model)` while the run used
`--init-score-source market`, so the model starts at the market logit and
the trees fit only a residual. That measures

```
[log_loss(baseline) - log_loss(market)]  +  [log_loss(market) - log_loss(model)]
   the market's edge over F(x/sigma)          what the FEATURES add
   IDENTICAL in all nine cells                the term under test
```

The constant is the whole of the reported number -- `+0.002237` is about the
size of `baseline - market` in the campaign ledger -- and its fold variance
sets the standard error. Eight near-zero deltas are one finding about the
init score reported eight times, not eight findings about features.

A second defect in the same line: `y = test['outcome']` is the Coinbase
proxy, so `--label-source venue` was ignored. The same bug was fixed in
`scripts/baseline.py` earlier the same night.

**Changing a metric after seeing results is what C5 exists to prevent**, so
this is written down rather than done quietly. The defence is that the
change is not a choice between metrics that could flatter a result: the
original cannot detect the quantity under test at all, and that is provable
from the decomposition above rather than from its numbers.

### Corrected, and fixed before run 2 is scored

* **Metric:** `delta = model_minus_market(all) - model_minus_market(all minus G)`.
  The null is `market_probability` whenever `init_score_source='market'`,
  which subtracts the constant and leaves the feature term.
* **Label:** the venue's settlement, per `--label-source venue`.
* **Threshold: unchanged.** `delta > 0` and `t > 2.7`, Bonferroni for eight
  tests. Not relaxed, which is the amendment that would actually be
  suspect.
* **Cells, configuration and sample: unchanged.**

### What a null result means NOW

If the groups still come back flat against the market null, that is a real
finding rather than an artefact: the features add nothing beyond the price.
It would be consistent with the standing result that the model has no
forecast edge over the market, and with `project-edge-is-vs-backfill-only`.
Run 1 cannot support that conclusion; run 2 can.

---

## NOTE, 2026-10-22 13:40 — run 2 restarted to pair the folds

Run 2 was stopped at 5 of 9 cells and restarted. **Nothing in the design
changed**: same cells, same configuration, same sample, same metric, same
`delta > 0` and `t > 2.7`.

What changed is that `scripts/ablate` now keeps the per-fold skill vectors and
differences them fold by fold. The printed table reports each cell's aggregate
and its standard error; the pre-registered quantity is the standard error of
`delta`, and a delta is paired — every cell scores the same folds on the same
rows. Differencing two aggregates discards that pairing and reports the error
of a difference of independent means, which is far too wide here because the
folds share most of their training windows and move together.

On synthetic vectors with a large shared fold effect, the two routes give the
same point estimate and standard errors of `0.000041` (paired) against
`0.000698` (unpaired) — a t of `+11.13` against `+0.66`. The unpaired route
could only ever return "nothing is significant", which is the same failure mode
as run 1 and would have been just as wrong.

The five cells scored before the restart are kept as a record. Their point
estimates (`delta` against `all groups` at `+0.000375`) were cross_venue
`+0.000239`, geometry `+0.000242`, vol_state `-0.000143`, microstructure
`+0.000009`, cross_asset `+0.000193`. **No threshold is read from them**; they
are superseded by the paired run.

---

## RESULT, 2026-10-22 — all eight groups DROP

```
all groups (vs market): +0.000334  se 0.000203  t +1.64  26/39 folds

group             delta         se       t  folds+    rho   verdict
cross_venue    +0.000166  0.000077   +2.14   20/39  0.925   DROP
geometry       +0.000174  0.000141   +1.23   24/39  0.727   DROP
cross_asset    +0.000134  0.000106   +1.26   25/39  0.867   DROP
implied_vol    +0.000039  0.000095   +0.41   24/39  0.893   DROP
book_quote     -0.000023  0.000082   -0.27   19/39  0.919   DROP
microstructure -0.000058  0.000106   -0.55   17/39  0.852   DROP
offset         -0.000111  0.000081   -1.37   16/39  0.918   DROP
vol_state      -0.000164  0.000112   -1.46   15/39  0.834   DROP
```

**The threshold was not touched.** `delta > 0` and `t > 2.7`, as fixed before
the run. The highest t is `cross_venue` at +2.14, which clears an uncorrected
two-sided 0.05 test but not Bonferroni for eight. Relaxing it after seeing
that number is precisely the amendment this document exists to prevent.

**Pairing was necessary and was not sufficient.** Measured fold correlation
averages 0.867, so the paired standard error is 2.8x tighter than
differencing the aggregates. Every unpaired t is under 0.7 — run 2 would have
reported "nothing is significant" for the wrong reason a second time. With the
pairing, the answer is the same but it is now a measurement rather than an
artefact of the method.

### What this does NOT say

**It does not say the features add nothing collectively.** Leave-one-out
measures a group's marginal value GIVEN the others. Groups that carry
overlapping information are each individually dispensable while the union is
not, and 33 of 48 features derive from the price and the strike, so overlap is
expected. The evidence about the whole set is the `all groups` cell:
`+0.000334` at `t +1.64` against the market, which is also not significant --
consistent with the standing "no forecast edge over the market" result and
with `project-edge-is-vs-backfill-only`, but not established by this design.

### A caution on `cross_venue`

It has the highest t and the WEAKEST fold agreement: **20 of 39**, a coin
flip. A positive weighted mean on even fold signs means a minority of folds
with large magnitude carry it. That is the profile of an unstable estimate,
not of the most promising group. Any future test of it should be a single
pre-registered hypothesis, not a survivor of this survey.

### Sample consequence

Dropping `cross_venue` and `implied_vol` takes the training sample from
121,355 to 341,944 rows -- 2.8x -- at a measured forecast cost of +0.000166
and +0.000039, neither significant. This is the cheapest available win and it
comes from a failure, as recorded in advance.

**NC2 is pending and gates the final configuration.** If information-free
features measurably hurt, the correct action is to shed feature COUNT rather
than these particular groups, and the group-level verdicts must be re-read in
that light.
