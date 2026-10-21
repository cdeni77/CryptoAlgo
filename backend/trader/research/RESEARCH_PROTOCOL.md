# RESEARCH_PROTOCOL.md — binding on every research session

> **Your job is not to find a profitable backtest. Your job is to determine whether
> a profitable strategy exists without allowing the research process itself to
> manufacture one.**

This file is in force across sessions. `DECISION_RULE.md`'s discipline about not
re-testing the same evidence applies to **every experiment**, not only to promotion.

## 0. The effective sample is far smaller than the row count

Never quote a row count as a sample size. In the July-October 2026 span:

| quantity | value |
|---|---|
| rows with a quote and a label | ~387,000 |
| symbol-windows | ~28,000 |
| **windows** (the correlation unit) | **~9,400** |
| windows with a TRUSTWORTHY market price (stream ladder) | **2,161** |

Four offsets share one settlement. BTC/ETH/SOL are ~0.7 correlated within a window.
Consecutive windows chain, because a window's strike is the previous window's
settlement. **All inference resamples WINDOWS, never rows.**

## 1. Research-level overfitting is the primary threat

- Every model, feature group, parameter set, training period, threshold and strategy
  variant evaluated is recorded in the campaign ledger — **including the failures.**
- Reporting only the successful experiments is a protocol violation.
- Historical data that has influenced a research decision **cannot later serve as
  independent confirmation.** Choosing a new split does not restore independence.
- A new experiment requires a distinct, economically or statistically defensible
  hypothesis. "The last one disappointed" is not a hypothesis.

## 2. Nested chronological validation

Outer chronological walk-forward; inner chronological selection. Inside each outer
training period and never using its test block: preprocessing, feature selection,
hyperparameters, calibration, thresholds, model choice.

Purge and embargo (1 day) for overlapping outcomes and feature lookbacks. Group by
window. **Never shuffle.**

Where no inner tuning is performed, say so — fixed hyperparameters chosen a priori
are *safer* than tuned ones at this sample size, and that is a result worth stating.

## 3. Control complexity; prefer the simpler candidate

Always compare against: regularized logistic regression, a small strongly-regularized
GBM, market-only calibration, and restricted feature sets. Full feature sets only
where the sample supports them.

For boosted trees, bound depth, leaves, min-samples-per-leaf, learning rate, rounds
and regularization. Early stopping only on training-internal validation.

**When two candidates differ by less than their statistical uncertainty, the simpler
one wins.** Do not expand hyperparameter ranges to chase marginal gains.

## 4. Feature stability is required, concentration must be quantified

Ablate feature groups on development data only. Measure incremental performance
**versus the market probability**, never versus 50% and never versus the baseline
alone. Report performance across symbols, months, decision offsets, market-probability
bands, volatility regimes, and fresh-versus-stale quotes.

Not every subset must be profitable — but concentration must be measured and stated.
**A strategy that works only in a segment discovered after looking at results is a
NEW hypothesis requiring independent confirmation**, not a finding.

## 5. Negative controls are mandatory

Every candidate that appears to beat the market must pass:

1. **Shuffled labels** within time blocks — skill must vanish.
2. **Noise-only features** — must score ~0.
3. **Deliberately delayed features** — skill must degrade in the expected direction.
4. **Future-data perturbation** — corrupt bars at or after the decision instant;
   features must not change.
5. **Missingness controls** — a feature's availability must not itself predict.

Negative controls are **not** proof of no leakage. Independent timestamp and
data-lineage tests are still required. **An unusually strong result is a reason to
audit the pipeline, not to celebrate.**

## 6. Uncertainty, and the distribution of all candidates

Report CIs on both the improvement versus the market and the net trading outcome,
using resampling that preserves dependence across symbols, offsets and nearby windows.
Account for model selection: deflated Sharpe, FDR control, or bootstrap selection-bias
diagnostics — **none of which is evidence of future profitability.**

Report **the best candidate and the distribution across all candidates.**

## 7. Regime sensitivity

Compare expanding against rolling training windows. Report by month. Test dependence
on one profitable week, one coin, one offset, one probability band. Vary fees, spread,
quote-age limits and fill assumptions. **Do not select the most favourable stress
assumptions.**

## 8. Protect the final evaluation

Freeze model, features, rules and thresholds before final evaluation. **If it fails,
that is a valid result** — do not tune on it and then reuse the period as untouched
evidence. Where no pristine historical holdout remains, say so plainly and use
forward paper-trading observations for confirmation.

## 9. Ranking criterion — generalization, not return

1. Point-in-time validity and data integrity
2. Consistent improvement versus the market probability
3. Positive executable expected value after realistic costs
4. Stability across chronological evaluation periods
5. Statistical uncertainty after accounting for selection
6. Simplicity and reproducibility

Historical return matters but never overrides a failure above. **+30% with unstable
evidence loses to +8% with credible evidence.**

Higher risk tolerance permits larger financial variance. **It never permits weaker
evidence.**

## 10. Stop condition

When the experiment budget is exhausted, **stop.** Deliver the best supported
candidate or state clearly that no robust edge was found. Do not keep modifying
features, architecture, datasets, evaluation boundaries or thresholds until something
positive appears.

**The campaign succeeds when its conclusions are trustworthy, even if every model
fails to beat Kalshi.**
