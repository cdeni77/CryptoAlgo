/** The gates, and every candidate that has ever been evaluated.
 *
 * Two things this page is careful about. The gates are shown in the order they
 * should be read — the forecast first, the money second — because a candidate
 * that fails a forecast gate should not have its Sharpe ratio discussed. And the
 * attempt history includes blocked candidates, because the trial count is what
 * any claim of skill has to be discounted by, and a project that hides its
 * failures cannot compute its own correction.
 */
import { usePolling } from '../hooks/usePolling';
import { fetchModel, fetchModelHistory } from '../api/serving';
import {
  Chip,
  Column,
  DataTable,
  Empty,
  Failed,
  Loading,
  Metric,
  Panel,
  SectionHead,
} from '../components/Primitives';
import { stamp } from '../lib/format';
import type { Gate, ModelAttempt } from '../types';

/** Which gates read the forecast rather than a simulated outcome. On the
 *  predecessor of this system every gate read an outcome, so a model 34x short
 *  of its cost hurdle failed all of them without any saying why. */
const FORECAST_GATES = new Set([
  'log_loss_skill',
  'folds_skill_positive',
  'calibration_error',
  'residual_scale',
  'control_gain_share',
  'windows_evaluated',
]);

export function ModelPage() {
  const model = usePolling(fetchModel, 60_000);
  const history = usePolling(() => fetchModelHistory(50), 120_000);

  if (model.loading) return <Loading what="the model" />;
  if (model.error) return <Failed error={model.error} what="the model" />;
  const state = model.data;

  if (!state?.present) {
    return (
      <div className="space-y-6">
        <SectionHead title="Model" note="Nothing has been promoted yet." />
        <Empty
          what={state?.reason ?? 'No promotion attempt recorded.'}
          next="python -m scripts.promote"
        />
      </div>
    );
  }

  const gates = state.gates ?? [];
  const forecast = gates.filter((g) => FORECAST_GATES.has(g.name));
  const money = gates.filter((g) => !FORECAST_GATES.has(g.name));
  const attempts = history.data?.attempts ?? [];

  return (
    <div className="space-y-8">
      <SectionHead
        eyebrow={`${state.version} · ${stamp(state.created_at)}`}
        title={state.installed ? 'Installed' : 'Blocked'}
        note={
          state.installed
            ? 'This is the artifact the live signal writer loads.'
            : 'Blocked candidates are kept. The trial count is the multiple-testing denominator.'
        }
        right={
          <div className="flex items-center gap-4">
            {state.forced && <Chip tone="warn">forced</Chip>}
            <Chip tone={state.installed ? 'pass' : 'fail'}>
              {state.installed ? 'live' : 'blocked'}
            </Chip>
          </div>
        }
      />

      {state.forced && state.force_reason && (
        <Panel className="border-warn/50 bg-surface">
          <div className="eyebrow text-warn">forced past the gates, reason recorded</div>
          <p className="mt-1 max-w-[80ch] text-tiny text-ink">{state.force_reason}</p>
        </Panel>
      )}

      <Panel>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Metric label="folds" value={state.folds ?? null} digits={0} />
          <Metric
            label="windows"
            value={state.windows_evaluated ?? null}
            digits={0}
            hint="out-of-sample windows, not rows — four offsets share a label"
          />
          <Metric
            label="skill"
            value={state.log_loss_skill ?? null}
            digits={5}
            tone={(state.log_loss_skill ?? 0) > 0 ? 'above' : 'below'}
          />
          <Metric label="alpha" value={state.residual_scale ?? null} digits={3} />
        </div>
      </Panel>

      <section className="grid gap-4 lg:grid-cols-2">
        <div>
          <SectionHead
            eyebrow={`${forecast.filter((g) => g.passed).length} of ${forecast.length} passed`}
            title="Forecast gates"
            note="Read these first. A candidate that fails here should not have its Sharpe discussed."
          />
          <Panel flush>
            <DataTable columns={gateColumns} rows={forecast} keyOf={(g) => g.name} />
          </Panel>
        </div>
        <div>
          <SectionHead
            eyebrow={`${money.filter((g) => g.passed).length} of ${money.length} passed`}
            title="Money gates"
            note="Including one that asks whether the result is possible rather than whether it is good."
          />
          <Panel flush>
            <DataTable columns={gateColumns} rows={money} keyOf={(g) => g.name} />
          </Panel>
        </div>
      </section>

      <section>
        <SectionHead
          eyebrow={`${attempts.length} attempts · ${attempts.filter((a) => a.installed).length} installed`}
          title="Every candidate"
          note="Blocked ones included, deliberately. Discount any claim of skill by this count."
        />
        <Panel flush>
          {history.loading ? (
            <Loading what="the ledger" />
          ) : (
            <DataTable
              columns={attemptColumns}
              rows={attempts}
              keyOf={(a) => a.version}
              empty={<Empty what="No attempts recorded." next="python -m scripts.promote" />}
            />
          )}
        </Panel>
      </section>
    </div>
  );
}

const gateColumns: Column<Gate>[] = [
  {
    key: 'state',
    head: '',
    width: '4.5rem',
    render: (g) => <Chip tone={g.passed ? 'pass' : 'fail'}>{g.passed ? 'pass' : 'fail'}</Chip>,
  },
  { key: 'name', head: 'gate', render: (g) => <span className="font-mono">{g.name}</span> },
  {
    key: 'value',
    head: 'value',
    numeric: true,
    render: (g) => (
      <span className={g.passed ? 'text-ink' : 'text-fail'}>{format(g.value)}</span>
    ),
  },
  {
    key: 'threshold',
    head: 'threshold',
    numeric: true,
    render: (g) => (
      <span className="text-ink-3">
        {g.direction === 'max' ? '≤' : '≥'} {format(g.threshold)}
      </span>
    ),
  },
];

const attemptColumns: Column<ModelAttempt>[] = [
  { key: 'version', head: 'version', render: (a) => <span className="font-mono">{a.version}</span> },
  { key: 'when', head: 'when', render: (a) => <span className="font-mono">{stamp(a.created_at)}</span> },
  {
    key: 'state',
    head: 'verdict',
    render: (a) => (
      <span className="flex items-center gap-3">
        <Chip tone={a.installed ? 'pass' : 'fail'}>{a.installed ? 'installed' : 'blocked'}</Chip>
        {a.forced && <Chip tone="warn">forced</Chip>}
      </span>
    ),
  },
  {
    key: 'skill',
    head: 'skill',
    numeric: true,
    render: (a) =>
      a.log_loss_skill == null ? '—' : (
        <span className={a.log_loss_skill > 0 ? 'text-above' : 'text-below'}>
          {a.log_loss_skill.toFixed(5)}
        </span>
      ),
  },
  { key: 'folds', head: 'agree', numeric: true, render: (a) => a.folds_positive ?? '—' },
  {
    key: 'windows',
    head: 'windows',
    numeric: true,
    render: (a) => (a.windows_evaluated ?? 0).toLocaleString(),
  },
  {
    key: 'failed',
    head: 'failed gates',
    render: (a) =>
      a.failed_gates.length === 0 ? (
        <span className="text-ink-3">none</span>
      ) : (
        <span className="font-mono text-micro text-fail">{a.failed_gates.join(', ')}</span>
      ),
  },
];

function format(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(5);
}
