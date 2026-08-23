/** Is the forecast honest about its own confidence?
 *
 * This page exists because the system only trades its confident predictions. A
 * model can hit the base rate exactly while being wrong at every level of
 * confidence, and a headline accuracy would not show it. What matters is the
 * 0.85–0.95 band, where the fee is cheapest and the trades actually happen.
 *
 * Both curves are shown, because the model's calibration is only interesting
 * relative to the baseline's: the residual architecture means the model is
 * fitting a correction, so if the baseline is already straight the model has
 * nothing to add, and if the baseline is bent the model's apparent skill is
 * partly the baseline's error.
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
import type { CalibrationBin } from '../types';

export function CalibrationPage() {
  const calibration = usePolling(() => fetchCalibration(), 120_000);
  const model = usePolling(fetchModel, 120_000);

  if (calibration.loading) return <Loading what="the reliability table" />;
  if (calibration.error) return <Failed error={calibration.error} what="the reliability table" />;

  const bins = calibration.data?.bins ?? [];
  const modelBins = bins.filter((b) => b.source === 'model');
  const baselineBins = bins.filter((b) => b.source === 'baseline');
  const state = model.data;

  return (
    <div className="space-y-8">
      <SectionHead
        eyebrow={
          calibration.data?.version
            ? `model ${calibration.data.version}`
            : 'no model recorded'
        }
        title="Calibration"
        note="The diagonal is the subject. A point above it means the outcome happened more often than the forecast said; below, less. Squares are the model, circles the baseline, and area is how many windows fell in the band."
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
          <div className="mt-4 flex gap-6 border-t border-rule pt-3 font-mono text-micro uppercase text-ink-3">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 bg-accent" aria-hidden /> model
            </span>
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-mid" aria-hidden /> baseline
            </span>
            <span className="flex items-center gap-2">
              <span className="h-px w-4 bg-rule-firm" aria-hidden /> perfect
            </span>
          </div>
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
              value={state?.folds_positive ?? null}
              digits={0}
              hint="five of six agreeing happens 10.9% of the time by chance"
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
              value={state?.windows_evaluated ?? null}
              digits={0}
              hint="out-of-sample windows, not rows — the four decision offsets share one settlement"
            />
            <Metric
              label="sharpe"
              value={state?.sharpe ?? null}
              digits={2}
              tone={(state?.sharpe ?? 0) > 5 ? 'below' : 'ink'}
              hint="annualised on trades actually placed. Above 5 this is a bug signature rather than an edge, and a gate says so."
            />
          </div>
          <p className="mt-4 border-t border-rule pt-3 text-tiny text-ink-2">
            Read fold agreement with its p-value beside it: under no skill each
            fold is a coin flip, so five of six agreeing happens{' '}
            <span className="font-mono">10.9%</span> of the time by chance. It is
            necessary, never sufficient.
          </p>
        </Panel>
      </div>

      <section className="grid gap-4 lg:grid-cols-2">
        <div>
          <SectionHead eyebrow={`${modelBins.length} bands`} title="Model" />
          <Panel flush>
            <DataTable columns={binColumns} rows={modelBins} keyOf={(b) => `m${b.bin_low}`} />
          </Panel>
        </div>
        <div>
          <SectionHead eyebrow={`${baselineBins.length} bands`} title="Baseline" />
          <Panel flush>
            <DataTable columns={binColumns} rows={baselineBins} keyOf={(b) => `b${b.bin_low}`} />
          </Panel>
        </div>
      </section>
    </div>
  );
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
    render: (b) =>
      b.predicted == null || b.observed == null ? (
        '—'
      ) : (
        <span
          className={
            Math.abs(b.observed - b.predicted) > 0.02 ? 'text-below' : 'text-ink-3'
          }
        >
          {((b.observed - b.predicted) * 100 >= 0 ? '+' : '') +
            ((b.observed - b.predicted) * 100).toFixed(2)}
          pp
        </span>
      ),
  },
  { key: 'n', head: 'windows', numeric: true, render: (b) => b.count.toLocaleString() },
];
