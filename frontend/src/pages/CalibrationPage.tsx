/** Is the forecast honest about its own confidence?
 *
 * This page exists because the system only trades its confident predictions. A
 * model can hit the base rate exactly while being wrong at every level of
 * confidence, and a headline accuracy would not show it. What matters is the
 * 0.85–0.95 band, where the fee is cheapest and the trades actually happen.
 *
 * **Only the model's curve is here, and that is a MISSING measurement rather
 * than a null result.** `scripts/promote.py` mirrors the candidate's
 * reliability table with `write_calibration(version, 'model', rows)` and
 * nothing in the repository has ever written a `'baseline'` row — measured
 * 2026-10-20, `/model/calibration` returned 20 bins, all of them `model`. The
 * page used to render a grey-circle legend entry, a chart note reading
 * "circles the baseline", and an empty "Baseline" table falling through to
 * "Nothing recorded yet" anyway. Absence of points then reads as the two curves
 * AGREEING, which is the most flattering misreading available and the one
 * direction this dashboard must never fail in.
 *
 * The epistemological point it was making is still true and is kept: under the
 * residual architecture the model fits a correction to `F(x/sigma)`, so a
 * straight model curve beside a bent baseline means something quite different
 * from a straight one beside a straight one. With one curve on screen this page
 * cannot tell you which — `calibration_vs_baseline` on the model page is where
 * that comparison is actually measured.
 *
 * **And the table is ONE fold.** `promote.py` stores "the reliability table of
 * the fold that produced the candidate", ~2,222 windows, while
 * `windows_evaluated` counts all 28 folds at 22,741. Shown side by side without
 * saying so, a reader concludes the calibration was measured on the full
 * out-of-sample set; it was measured on about a tenth of it.
 */
import { usePolling } from '../hooks/usePolling';
import { fetchCalibration, fetchModel } from '../api/serving';
import { ReliabilityChart } from '../components/Charts';
import {
  Column,
  DataTable,
  Empty,
  Failed,
  Loading,
  Metric,
  Panel,
  SectionHead,
} from '../components/Primitives';
import { pct } from '../lib/format';
import type { CalibrationBin, DeflatedSharpe } from '../types';

/** `core/metrics.py` calls a bin adequately populated at 500 windows —
 *  `pooled_max_deviation` masks `counts >= 500` before taking its worst
 *  deviation, precisely because binned ECE is biased upward at small n. Most
 *  bands in a single fold's table are under it (the 0.00–0.02 band holds 72),
 *  so a band that LOOKS wild is usually just thin, and the page has to say
 *  which before the eye decides. */
const ADEQUATE_BIN_WINDOWS = 500;

export function CalibrationPage() {
  const calibration = usePolling(() => fetchCalibration(), 120_000);
  const model = usePolling(fetchModel, 120_000);

  if (calibration.loading) return <Loading what="the reliability table" />;
  if (calibration.error) return <Failed error={calibration.error} what="the reliability table" />;

  const bins = calibration.data?.bins ?? [];
  const modelBins = bins.filter((b) => b.source === 'model');
  const baselineBins = bins.filter((b) => b.source === 'baseline');
  const state = model.data;

  // The scope of the table, computed rather than asserted: these are the
  // windows the curve on this page was actually measured on, and the headline
  // `windows_evaluated` beside it is a different, much larger number.
  const tableWindows = modelBins.reduce((sum, b) => sum + b.count, 0);
  const evaluated = state?.windows_evaluated ?? null;
  const tableShare = evaluated && evaluated > 0 ? tableWindows / evaluated : null;
  const thinBands = modelBins.filter((b) => b.count < ADEQUATE_BIN_WINDOWS).length;

  const folds = state?.folds ?? null;
  const foldsPositive = state?.folds_positive ?? null;
  // The real p-value for THIS run's fold count. The page used to print a
  // hardcoded 10.9% in two places; that is the constant from the RETIRED
  // `folds_skill_positive >= 5` gate, which `core/metrics.py` replaced with
  // `sign_agreement_p` because a COUNT is not comparable across fold counts —
  // "5 of 10" is a 62% event under the null where "5 of 6" is a 10.9% one.
  // This run is 20 of 28 at p = 0.0178.
  const signP = state?.report_stats?.sign_agreement_p ?? null;
  const deflated = state?.deflated_sharpe ?? null;

  return (
    <div className="space-y-8">
      <SectionHead
        eyebrow={
          calibration.data?.version
            ? `model ${calibration.data.version}`
            : 'no model recorded'
        }
        title="Calibration"
        note="The diagonal is the subject. A point above it means the outcome happened more often than the forecast said; below, less. Squares are the model, and area is how many windows fell in the band. One fold's table, not the whole evaluation."
      />

      <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        <Panel>
          {bins.length === 0 ? (
            <Empty
              what={calibration.data?.reason ?? 'No reliability table stored.'}
              next="python -m scripts.evaluate"
            />
          ) : (
            <ReliabilityChart bins={bins} />
          )}
          {/* `flex-wrap` and a row gap: three legend chips plus a wrapped
              sentence do not fit one 375px line. */}
          <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-t border-rule pt-3 font-mono text-micro uppercase text-ink-3">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 bg-accent" aria-hidden /> model
            </span>
            {/* Only drawn when there is something to draw. A legend entry for a
                series the chart cannot plot is a claim that it was measured. */}
            {baselineBins.length > 0 && (
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-mid" aria-hidden /> baseline
              </span>
            )}
            <span className="flex items-center gap-2">
              <span className="h-px w-4 bg-rule-firm" aria-hidden /> perfect
            </span>
          </div>
          {baselineBins.length === 0 && (
            <p className="mt-3 text-tiny text-ink-2">
              <span className="font-mono text-micro uppercase text-warn">
                baseline not recorded
              </span>{' '}
              — the promotion mirror writes model rows only, so the baseline’s own
              curve has never been stored for any artifact. Nothing on this chart
              says the two curves agree; there is no second curve.
            </p>
          )}
        </Panel>

        <Panel>
          <SectionHead
            title="What the numbers say"
            note="Skill is measured against the baseline, never against a coin flip — the barrier arithmetic alone takes log loss roughly 26% below 50/50."
          />
          <div className="grid grid-cols-2 gap-x-4 gap-y-4">
            <Metric
              label="log loss skill"
              value={state?.log_loss_skill ?? null}
              digits={5}
              tone={(state?.log_loss_skill ?? 0) > 0 ? 'above' : 'below'}
              hint="baseline log loss minus model log loss. Positive means the model helped."
            />
            <Metric
              label="std error"
              value={state?.log_loss_skill_se ?? null}
              digits={5}
              tone="muted"
              hint="from fold dispersion, not from a breadth formula — four offsets share a label and the symbols are correlated"
            />
            <Metric
              label="folds positive"
              value={foldsPositive}
              unit={folds == null ? undefined : ` / ${folds}`}
              digits={0}
              // The denominator is part of the number. Without it the count is
              // uninterpretable, which is the whole reason the gate moved to a
              // p-value — see `sign agreement p` beside it.
              hint="how many folds scored positive, and out of how many. A count alone is not comparable across fold counts; read the p-value next to it."
            />
            <Metric
              label="sign agreement p"
              value={signP == null ? null : signP * 100}
              unit="%"
              digits={2}
              // The gate, not a round number: `GateSpec('sign_agreement_p',
              // 0.11, 'max')`, chosen because 0.11 reproduces the old
              // five-of-six bar exactly (p = 0.109) at any fold count.
              tone={signP == null ? 'muted' : signP <= 0.11 ? 'above' : 'below'}
              hint="the chance this many folds agree if each were a coin flip. The gate is 0.11, which reproduces the retired five-of-six count exactly and stays that strict at 28 folds."
            />
            <Metric
              label="calib. error"
              value={state?.calibration_error ?? null}
              digits={4}
              hint="worst fold's expected calibration error"
            />
            <Metric
              label="alpha"
              value={state?.residual_scale ?? null}
              digits={3}
              hint="how much of the model's claimed correction survives out of sample. Near zero means it found nothing."
            />
            <Metric
              label="control gain"
              value={state?.control_gain_share == null ? null : state.control_gain_share * 100}
              unit="%"
              digits={1}
              tone={(state?.control_gain_share ?? 0) > 0.3 ? 'below' : 'muted'}
              hint="share of model gain taken by hour-of-day. Time cannot forecast direction, so a large share indicts the measurement."
            />
            <Metric
              label="windows"
              value={evaluated}
              digits={0}
              // Scope stated on the metric itself. This figure sits directly
              // above a per-band window table measured on ONE fold, and the
              // adjacency alone made it read as the table's sample size.
              hint={
                'out-of-sample windows across every fold, not rows — the four decision offsets share one settlement. The reliability table below is ONE fold' +
                (tableWindows ? `, ${tableWindows.toLocaleString()} of them.` : '.')
              }
            />
            <Metric
              label="sharpe"
              value={state?.sharpe ?? null}
              digits={3}
              tone={(state?.sharpe ?? 0) > 5 ? 'below' : 'ink'}
              hint="annualised on trades actually placed. Above 5 this is a bug signature rather than an edge, and a gate says so."
            />
            <Metric
              label="deflated sharpe"
              value={deflated?.deflated_sharpe ?? null}
              digits={3}
              // Shown BESIDE the raw figure rather than instead of it. The raw
              // Sharpe is the best of `n_trials` attempts, and the only hint it
              // used to carry flagged values above 5 — which says nothing at
              // all about a 0.128 selected from 19 tries.
              tone={
                deflated?.deflated_sharpe == null
                  ? 'muted'
                  : deflated.deflated_sharpe >= 0.95
                    ? 'above'
                    : 'below'
              }
              hint={deflationHint(deflated)}
            />
          </div>
          <p className="mt-4 border-t border-rule pt-3 text-tiny text-ink-2">
            {signP == null || foldsPositive == null || folds == null ? (
              <>
                Read fold agreement with its p-value beside it: under no skill each
                fold is a coin flip, so a count of agreeing folds only means
                something against the number of folds run. This artifact records no{' '}
                <span className="font-mono">sign_agreement_p</span>, so the count
                above stands alone and should not be read as evidence.
              </>
            ) : (
              <>
                Read fold agreement with its p-value beside it: under no skill each
                fold is a coin flip, so {foldsPositive} of {folds} agreeing happens{' '}
                <span className="font-mono">{pct(signP, 1)}</span> of the time by
                chance. It is necessary, never sufficient.
              </>
            )}
          </p>
        </Panel>
      </div>

      <section className="grid gap-4 lg:grid-cols-2">
        <div>
          <SectionHead
            eyebrow={`${modelBins.length} bands · ${tableWindows.toLocaleString()} windows`}
            title="Model"
            note={
              `One fold — the one trained on the most history, and so the one deployed. These bands sum to ${tableWindows.toLocaleString()} windows` +
              (evaluated == null
                ? '.'
                : ` against the ${evaluated.toLocaleString()} evaluated across every fold${
                    tableShare == null ? '' : `, about ${(tableShare * 100).toFixed(0)}% of them`
                  }.`)
            }
          />
          <Panel flush>
            <DataTable columns={binColumns} rows={modelBins} keyOf={(b) => `m${b.bin_low}`} />
          </Panel>
          {thinBands > 0 && (
            <p className="mt-3 text-tiny text-ink-2">
              <span className="font-mono text-warn">†</span>{' '}
              {thinBands === modelBins.length
                ? `Every one of the ${modelBins.length} bands holds`
                : `${thinBands} of ${modelBins.length} bands hold`}{' '}
              fewer than {ADEQUATE_BIN_WINDOWS} windows, which is the floor{' '}
              <span className="font-mono">core/metrics.py</span> applies before it
              will call a bin’s deviation measurable. Their gaps are left uncoloured:
              at n = 72 a 10pp miss is what a thin bin looks like, not what a
              miscalibrated model looks like.
            </p>
          )}
        </div>
        <div>
          <SectionHead
            eyebrow={baselineBins.length > 0 ? `${baselineBins.length} bands` : 'not recorded'}
            title="Baseline"
            note={
              baselineBins.length > 0
                ? undefined
                : "The comparison this page was built around, and it is not on screen."
            }
          />
          {baselineBins.length > 0 ? (
            <Panel flush>
              <DataTable columns={binColumns} rows={baselineBins} keyOf={(b) => `b${b.bin_low}`} />
            </Panel>
          ) : (
            // Deliberately NOT `Empty`: that component means "waiting, and here
            // is the command that fills it". No command fills this one — no
            // writer exists — and offering `scripts.evaluate` would promise a
            // table that will never arrive.
            <Panel>
              <p className="text-tiny text-ink-2">
                <span className="font-mono">scripts/promote.py</span> mirrors the
                candidate’s reliability table as{' '}
                <span className="font-mono">{"source='model'"}</span>, and nothing in
                this repository writes a{' '}
                <span className="font-mono">{"'baseline'"}</span> row. This table has
                never held data and no run will fill it. That is an{' '}
                <em>absent</em> measurement, not an empty result — the difference
                matters, because an empty baseline table beside a straight model
                curve reads as the two agreeing.
              </p>
              <p className="mt-3 text-tiny text-ink-2">
                The comparison does exist numerically. The{' '}
                <span className="font-mono">calibration_vs_baseline</span> gate on
                the model page measures the model’s expected calibration error
                against the baseline’s on the same rows, which is the question this
                panel was meant to answer by eye.
              </p>
            </Panel>
          )}
        </div>
      </section>
    </div>
  );
}

/** The raw Sharpe's own denominator, spelled out in the tooltip.
 *
 *  Built from the payload rather than written in prose: `trial_count` grows with
 *  every promotion attempt, and a sentence naming "19 trials" would be stale by
 *  the next retrain — which is exactly how the 10.9% above survived a gate
 *  retirement. */
function deflationHint(d: DeflatedSharpe | null | undefined): string {
  const base =
    'probability the Sharpe survives the multiple-testing correction. The raw figure is the best of every trial run, so it is selected, not sampled.';
  if (d == null) return `${base} Not recorded for this artifact.`;
  const parts: string[] = [];
  if (d.n_trials != null) parts.push(`${d.n_trials} trials`);
  if (d.expected_max_sharpe != null) {
    parts.push(`best-of-trials expected under NO skill ${d.expected_max_sharpe.toFixed(2)}`);
  }
  if (d.sharpe != null) parts.push(`raw ${d.sharpe.toFixed(3)}`);
  if (parts.length === 0) return base;
  return `${base} Measured: ${parts.join(', ')}.`;
}

const binColumns: Column<CalibrationBin>[] = [
  {
    key: 'band',
    head: 'band',
    render: (b) => (
      <span className="font-mono">
        {b.bin_low.toFixed(2)}–{b.bin_high.toFixed(2)}
      </span>
    ),
  },
  { key: 'predicted', head: 'predicted', numeric: true, render: (b) => pct(b.predicted, 2) },
  { key: 'observed', head: 'observed', numeric: true, render: (b) => pct(b.observed, 2) },
  {
    key: 'gap',
    head: 'gap',
    numeric: true,
    render: (b) => {
      if (b.predicted == null || b.observed == null) return '—';
      const gap = b.observed - b.predicted;
      // **Colour only where the bin can carry the claim.** 0.02 is an
      // attention marker, not the gate — `calibration_max_deviation` sits at
      // 0.10 — and in a thin bin it fires on sampling noise: the 0.00–0.02
      // band observed 0.00% on 72 windows, which is the single most alarming
      // cell in the table and is explained entirely by n.
      const populated = b.count >= ADEQUATE_BIN_WINDOWS;
      return (
        <span className={populated && Math.abs(gap) > 0.02 ? 'text-below' : 'text-ink-3'}>
          {(gap * 100 >= 0 ? '+' : '') + (gap * 100).toFixed(2)}
          pp
        </span>
      );
    },
  },
  {
    key: 'n',
    head: 'windows',
    numeric: true,
    // The marker lives on the COUNT, which is the thing it is about, and not on
    // the band: at 375px the band cell is ~9ch and a trailing dagger pushed
    // `0.50–0.60 †` onto a third line in every row. Colour is not the only
    // signal — the dagger carries it for anyone who cannot separate the warn
    // ochre from the body ink.
    render: (b) =>
      b.count < ADEQUATE_BIN_WINDOWS ? (
        <span
          className="text-warn"
          title={`${b.count.toLocaleString()} windows — under the ${ADEQUATE_BIN_WINDOWS} this repo requires before a bin's deviation is read as a measurement`}
        >
          {b.count.toLocaleString()} †
        </span>
      ) : (
        b.count.toLocaleString()
      ),
  },
];
