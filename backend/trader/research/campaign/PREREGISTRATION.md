# Pre-registration — 12 experiments

Written **before any experiment was run** and before any result was inspected.
Frozen 2026-10-17. Amendments are appended with a reason and a timestamp; the
original text is never edited.

## Independence of the evaluation period — stated honestly up front

Investigated: **no pristine historical test period remains in July-October 2026.**

- The deployed artifact `20261014T143301Z` was trained on data through 2026-10-14
  with `trial_count = 18` recorded in its own ledger.
- `AUDIT_REPORT.md`'s Edge Investigation measured offset-level skill over this span.
- Memory records a 64-cell selection sweep (2026-09-04) and multiple backtest-live
  gap investigations over September and October.
- The live start line is 2026-09-24; the live loop itself has been trading and
  recording across the whole window.

Choosing a new split cannot restore independence to data that has already shaped
decisions. Therefore:

| period | role | independence |
|---|---|---|
| 2026-08-25 -> 2026-09-08 (15d) | development: fitting, feature selection, tuning | none |
| 2026-09-09 -> 2026-09-16 (8d) | validation: candidate ranking, threshold choice | **weak** — the span has been inspected before, though not at this granularity by this campaign |
| forward paper trading from 2026-10-17 | **confirmation** | **the only genuinely pristine evidence available** |

**Every result in this campaign is EXPLORATORY.** The definitive test is locked
forward paper-trading data, not any historical split.

## Evaluation universe (fixed, from the diagnostics)

- **Benchmark + executable price**: stream ladder (`source in ('live','live_ws')`,
  venue kalshi), 2026-08-25 -> 2026-09-16. 6,488 symbol-windows, offsets 1..14.
- **Label**: the venue's own settlement (`venue_settlements`, kalshi), never the
  Coinbase proxy, because the market was priced against the venue's number.
- **Features** may use a longer history; the backfill quote may be used as a feature
  only when **shifted to offset m-1** (D5), never as a price or a benchmark.
- **Folds split on the window**, never the row. 1-day embargo. Symbols stay together.

## Fixed evaluation rules, identical for every candidate

1. Primary metric: **`model_minus_market`** = market log loss - model log loss, on
   out-of-fold predictions only. Positive means the model beat the Kalshi price.
2. Significance: paired bootstrap over **windows** (not rows), 10,000 resamples.
3. Multiplicity: 12 experiments; a candidate is "supported" only if its one-sided
   bootstrap p survives a Holm correction across all candidates actually run.
4. Money: one entry per (symbol, window) — walk offsets in order, take the first
   clearing the gate. Buy at the **live ask**, pay the venue fee schedule, hold to
   settlement. Never the mid, never the backfill.
5. Thresholds are chosen on **development** data only and applied unchanged to
   validation.
6. Compute budget: **20 minutes per experiment**. Exceeded = recorded as a failure
   with its partial state, not extended.

## Reference points (not experiments)

- **R0** — Kalshi mid, raw. The thing to beat.
- **R1** — the barrier baseline `F(x/sigma)` alone.
- **R2** — the deployed artifact `20261014T143301Z`, scored under these same rules.

## The 12 experiments

| # | hypothesis | why it could beat the price | falsified if |
|---|---|---|---|
| **E1** | The Kalshi mid carries a stable favourite-longshot miscalibration; a 1-D monotone recalibration of the mid beats the raw mid. | Binary venues price tails systematically wide; this needs no new information, only a correction. | OOF log loss >= raw mid. |
| **E2** | That miscalibration varies by decision offset. | Early windows are near coin-flip and late ones nearly settled; one curve cannot serve both. | No offset-conditional curve beats the pooled E1 curve OOF. |
| **E3** | The barrier baseline `F(x/sigma)` carries information the market does not. | The market may under-use the exact known displacement. | Baseline log loss >= market log loss. |
| **E4** | A regularized logistic model on the market logit plus geometry and vol_state beats the market. | Linear, low-variance, well suited to 2,163 windows. | `model_minus_market` CI includes 0. |
| **E5** | LightGBM on reliable bar-only features with the market logit as `init_score` beats the market. | The deployed architecture, evaluated honestly for the first time against a clean price. | as E4. |
| **E6** | A short recent training window (30d) beats E5's long window. | Regime drift; CLAUDE.md's retrain cadence assumes it. | E6 `model_minus_market` <= E5's. |
| **E7** | Adding `implied_vol` (the market's own forward sigma, BTC/ETH only) beats E5. | Free forward-looking input no feature group has used; SOL excluded for thinness. | E7 <= E5. |
| **E8** | Verified order-book features (`market_state`, `book_flow`) from the stream ladder beat E5. | The archive left imbalance open (t=+0.30 -> +1.84 over 67 days); the book is the one source the bar features cannot express. | E8 <= E5. |
| **E9** | CatBoost's ordered boosting beats LightGBM at this sample size. | Different bias/variance trade-off; LightGBM overfits small panels. | E9 <= E5. |
| **E10** | Targeting the market residual beats targeting settlement with the market as a feature. | Residual target is better conditioned and cannot echo the price. | residual target <= direct target. |
| **E11** | Polymarket's price, as a cross-venue feature, identifies Kalshi mispricing. | Different participants, different oracle; disagreement is information. | E11 <= E5. |
| **E12** | A cross-fitted ensemble of the candidates that individually beat the market beats each of them. | Error decorrelation. | **Not run unless >= 2 candidates individually clear their CI.** |

## Stopping and anti-loop rules

- An experiment that fails is recorded with its failure reason and **not retried**
  with a tweaked configuration. A new configuration requires a new numbered entry
  and a stated reason it is not a retry.
- If a dataset or leakage bug is found, the affected experiments are invalidated and
  **the same pre-registered specifications are rerun**. No new specifications are
  invented at that point.
- Tuning budget per candidate: one grid of at most 6 configurations, chosen on
  development data, before validation is touched.

---

# AMENDMENT 1 — 2026-10-17, after diagnostic D7

**The original text above is unchanged and remains the record.** This amendment
states what changed and why, per the amendment rule in the header.

## What changed

Diagnostic D7 overturned D3/D4/D5. The evaluation universe in the original text —
"stream ladder, 2026-08-25..09-16, 6,488 symbol-windows" — was derived from a
diagnostic that was itself wrong.

Measured: `record_ladder` samples the live book at **+32 seconds** past the minute it
stamps, while the Predexon backfill is stamped and available at the minute exactly.
A `live` row labelled offset m therefore describes the book at m+32s and **was not
available at the decision instant**. Against a genuinely contemporaneous price the
backfill is statistically indistinguishable (-0.000040, 95% CI [-0.00286, +0.00276]).

The original harness joined `live@m` as both benchmark and executable price, which
made every experiment compete against a market holding 32 extra seconds of
information in a 15-minute market.

## What this amendment does and does not do

**Does:**
- Replace the quote selection with an **as-of join on `available_time`**: the usable
  quote at offset m is the most recent Kalshi row available at or before
  `window_open + m + 3s`. One rule, every source, no per-source special case.
- Widen the evaluation universe accordingly:

  | | original | amended |
  |---|---|---|
  | span | 2026-08-25..09-16 (23d) | **2026-07-01..10-14 (105d)** |
  | rows | 84,513 | **388,466** |
  | windows | 2,161 | **9,757** |
  | symbol-windows | 6,482 | **28,841** |

- Reset the development / validation boundary to `DEV_END = 2026-09-16`
  (dev = Jul 1..Sep 15, validation = Sep 16..Oct 14).
- Put every candidate on **identical folds**, since the market price now exists
  across the whole era. The original's split into "full-history" and "stream-era"
  fold sets existed only because the price was thought to be confined to 23 days.

**Does NOT:**
- Add, remove or alter any of the twelve hypotheses. The same specifications are
  rerun verbatim, which is what RESEARCH_PROTOCOL.md section 1 requires after a
  dataset bug.
- Change the evaluation rules, the significance procedure, or the multiplicity
  correction.

## Invalidated by this amendment

E1, E1b, E2, E3 and batch 1 (E4, E5, E5c, E6, E6b, E9, E10) plus the negative
controls were all scored against the lookahead price. **All are invalidated and
rerun.** Their original numbers are kept in `LEDGER.md` under D7 as a record of the
error, not as results.

## Independence — unchanged and now worse, not better

Widening the span does **not** create a pristine holdout. September and October are
precisely where the deployed artifact was trained (through 2026-10-14, trial_count
18) and where the live loop has been running. **Every result in this campaign remains
EXPLORATORY.** The definitive test is still locked forward paper-trading data.
