/** The gates, and every candidate that has ever been evaluated.
 *
 * Two things this page is careful about. The gates are shown in the order they
 * should be read — the forecast first, the money second — because a candidate
 * that fails a forecast gate should not have its Sharpe ratio discussed. And the
 * attempt history includes blocked candidates, because the trial count is what
 * any claim of skill has to be discounted by, and a project that hides its
 * failures cannot compute its own correction.
 *
 * Both of those were stated here and neither was true on screen until
 * 2026-10-20. The split filed three FAILING benchmark gates under "Money" and
 * reported "Forecast gates — 4 of 4 passed", because two of the six names it
 * matched on (`folds_skill_positive`, `windows_evaluated`) are not gate names
 * at all; and the trial count printed the number of ledger ROWS the history
 * endpoint happened to return (17) rather than the multiple-testing denominator
 * the promoter computed (19). The page said the right things and showed the
 * opposite of them.
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

/** The first gate of the `--- the money ---` block in `core/metrics.py`'s
 *  `GATE_SPEC`, which is the order the API serialises them in.
 *
 *  **Splitting on a POSITION rather than on a membership list is the point.**
 *  The list this replaces named six gates, two of which had never existed, and
 *  it had no way to learn about the nine added since — a gate missing from the
 *  list silently became a money gate, which is how three failing benchmark
 *  comparisons ended up filed under Sharpe. An anchor inherits the spec's own
 *  split instead: a gate added above `trades` is a forecast gate here without
 *  this file being touched, and one added below it is a money gate. */
const FIRST_MONEY_GATE = 'trades';

/** Only read when the anchor above is absent — an old ledger row, or a spec
 *  that renamed it. A wrong split is still better than everything landing on
 *  one side, but it is a fallback and will drift; the anchor will not. */
const MONEY_GATES = new Set([
  'trades',
  'coverage',
  'realised_edge_pp',
  'realised_edge_lcb',
  'overconfidence_abs',
  'sharpe_implausible',
  'median_fold_drawdown',
  'halted',
]);

/** A Deflated Sharpe Ratio is a PROBABILITY, not a ratio: the chance the
 *  observed Sharpe beats what the best of `n_trials` tries would produce by
 *  luck alone. So "positive" is not the bar — near one is. 0.95 is the usual
 *  confidence level and is used only to colour the figure. */
const DSR_CONVINCING = 0.95;

function splitGates(gates: Gate[]): { forecast: Gate[]; money: Gate[] } {
  const at = gates.findIndex((g) => g.name === FIRST_MONEY_GATE);
  if (at >= 0) return { forecast: gates.slice(0, at), money: gates.slice(at) };
  return {
    forecast: gates.filter((g) => !MONEY_GATES.has(g.name)),
    money: gates.filter((g) => MONEY_GATES.has(g.name)),
  };
}

/** `passed of total`, with gates that could not be COMPUTED called out
 *  separately — they are neither a pass nor a loss, and counting them as
 *  failures would make "the live loop has not recorded enough quotes yet" read
 *  as "the model lost to the price". */
function tally(rows: Gate[]): string {
  const open = rows.filter((g) => g.measured === false).length;
  const passed = rows.filter((g) => g.passed).length;
  return `${passed} of ${rows.length} passed${open ? ` · ${open} not yet measurable` : ''}`;
}

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
  const { forecast, money } = splitGates(gates);
  const gateNamed = (name: string) => gates.find((g) => g.name === name);
  const attempts = history.data?.attempts ?? [];
  const stats = state.report_stats ?? {};
  const dsr = state.deflated_sharpe ?? null;

  // **The headline skill is a point estimate and must not be coloured as a
  // win on its own.** 0.00034 rendered green while `log_loss_skill_p` = 0.075
  // FAILED its own 0.05 gate on the same screen. The gate's verdict drives the
  // colour, so the two cannot disagree.
  const skillP = gateNamed('log_loss_skill_p');
  const skillSignificant = skillP?.passed ?? false;

  // **Three different numbers have been displayed under the label "alpha".**
  // `report_stats.mean_residual_scale` (0.687) is the one `core/metrics.py`
  // explicitly REJECTED as the gate aggregation — on a zero-signal null "the
  // mean was 0.667 and cleared the 0.25 gate while two thirds of the evidence
  // said there was nothing there". The `residual_scale` gate reads the median
  // over folds (0.940). And the artifact that actually trades applies 0.0151:
  // a 62x difference between the number on screen and the number in the
  // signal writer. The deployed one is the headline for that reason.
  const deployedAlpha = state.deployed_residual_scale ?? null;
  // Threshold from the payload, never hardcoded here: `deployed_residual_scale`
  // is gated at the walk-forward's own bar "because it is the same quantity
  // asked of the thing that actually trades".
  const alphaFloor =
    gateNamed('deployed_residual_scale')?.threshold ?? gateNamed('residual_scale')?.threshold ?? null;
  const alphaTone =
    deployedAlpha == null || alphaFloor == null
      ? 'ink'
      : deployedAlpha >= alphaFloor
        ? 'above'
        : 'below';

  return (
    <div className="space-y-8">
      <SectionHead
        // **One instant, stated once.** This used to print the version beside
        // `stamp(created_at)`: the version is a UTC stamp ending `Z` and the
        // stamp is Eastern, so the same moment appeared twice, four hours
        // apart, with neither zone labelled. They are the same second —
        // `20261018T064404Z` against `06:44:05Z` — so the ET reading is given
        // as an equality rather than as a second fact.
        eyebrow={`${state.version} (UTC) = ${stamp(state.created_at)} ET`}
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
          <Metric
            label="skill"
            value={state.log_loss_skill ?? null}
            digits={5}
            tone={skillSignificant && (state.log_loss_skill ?? 0) > 0 ? 'above' : 'ink'}
            hint="log-loss improvement on F(x/sigma), window-weighted over folds. A point estimate: read the p below it"
          />
          <Metric
            label="alpha deployed"
            value={deployedAlpha}
            digits={4}
            tone={alphaTone}
            hint="how much of its claimed correction the INSTALLED artifact keeps — not the walk-forward aggregate"
          />
          <Metric
            label="windows"
            value={state.windows_evaluated ?? null}
            digits={0}
            hint="out-of-sample windows, not rows — four offsets share a label"
          />
          <Metric label="folds" value={state.folds ?? null} digits={0} />
        </div>

        {/* The uncertainty belongs next to the estimate, not on another page.
            `se`/`t`/`p` are the same fold dispersion the gate uses, so a reader
            cannot take +0.00034 for a result without also seeing that it is
            1.5 standard errors from zero. */}
        <dl className="mt-4 flex flex-wrap gap-x-5 gap-y-1 border-t border-rule pt-3 font-mono text-micro text-ink-3">
          <Stat label="se" value={stats.log_loss_skill_se} digits={5} />
          <Stat label="t" value={stats.log_loss_skill_t} digits={2} />
          <Stat
            label="p"
            value={stats.log_loss_skill_p}
            digits={3}
            tone={skillP && !skillP.passed ? 'text-fail' : undefined}
            title={
              skillP && !skillP.passed
                ? `fails log_loss_skill_p ≤ ${format(skillP.threshold)} — the sign clears zero on half of all nulls`
                : undefined
            }
          />
          <Stat label="n_eff folds" value={stats.effective_folds} digits={1} />
          <Stat label="sign agreement p" value={stats.sign_agreement_p} digits={4} />
        </dl>

        {/* The same label on three different quantities is how a reader
            concludes the correction survives at 69% when it survives at 1.5%.
            The fold-level pair stays on the page — the gates read it — but as a
            named, secondary figure rather than as "alpha". */}
        <p className="mt-2 max-w-[80ch] text-micro text-ink-3">
          <span className="text-ink-2">walk-forward alpha, a different quantity:</span>{' '}
          median {format(gateNamed('residual_scale')?.value ?? NaN)} (what the{' '}
          <code className="font-mono">residual_scale</code> gate reads) · mean{' '}
          {format(stats.mean_residual_scale ?? NaN)} (reported, not gated — on a zero-signal null
          one runaway fold carried the mean over the bar). The artifact that trades applies{' '}
          {format(deployedAlpha ?? NaN)}.
        </p>
      </Panel>

      <Panel>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Metric
            label="trials"
            value={state.trial_count ?? null}
            digits={0}
            hint="every candidate ever evaluated, from the promotion ledger — the multiple-testing denominator"
          />
          <Metric label="sharpe raw" value={state.sharpe ?? null} digits={3} />
          <Metric
            label="expected best of n"
            value={dsr?.expected_max_sharpe ?? null}
            digits={3}
            hint="the Sharpe the best of this many tries returns by luck alone, given their spread"
          />
          <Metric
            label="sharpe deflated"
            value={dsr?.deflated_sharpe ?? null}
            digits={4}
            tone={
              dsr?.deflated_sharpe == null
                ? 'ink'
                : dsr.deflated_sharpe >= DSR_CONVINCING
                  ? 'above'
                  : 'below'
            }
            hint="probability the raw Sharpe beats the best-of-n-tries benchmark. A probability, so near 1 is the bar, not above 0"
          />
        </div>
        {/* Computed on every run, written to the ledger, and never once shown.
            A raw 0.128 against an expected best-of-19 of 2.375 is the whole
            reading: the live Sharpe is indistinguishable from what nineteen
            tries produce with no edge at all. */}
        <p className="mt-3 max-w-[80ch] border-t border-rule pt-3 text-micro text-ink-3">
          {dsr?.probabilistic_sharpe != null && (
            <>
              Probabilistic Sharpe {format(dsr.probabilistic_sharpe)} before the trial count,{' '}
              {format(dsr.deflated_sharpe ?? NaN)} after it.{' '}
            </>
          )}
          Discount any claim of skill on this page by {state.trial_count ?? dsr?.n_trials ?? '—'}{' '}
          trials.
        </p>
      </Panel>

      <section className="grid gap-4 lg:grid-cols-2">
        <div>
          <SectionHead
            eyebrow={tally(forecast)}
            title="Forecast gates"
            note="Read these first. A candidate that fails here should not have its Sharpe discussed."
          />
          <Panel flush>
            <DataTable columns={gateColumns} rows={forecast} keyOf={(g) => g.name} />
          </Panel>
        </div>
        <div>
          <SectionHead
            eyebrow={tally(money)}
            title="Money gates"
            note="Including one that asks whether the result is possible rather than whether it is good."
          />
          <Panel flush>
            <DataTable columns={gateColumns} rows={money} keyOf={(g) => g.name} />
          </Panel>
        </div>
        {/* Thresholds move. Without the hash of the spec they were evaluated
            under, a reader cannot tell whether the bars on screen are the ones
            in force — compare it with `backend/trader/core/gate_spec.lock.json`. */}
        {state.gate_spec_hash && (
          <p
            className="font-mono text-micro text-ink-3 lg:col-span-2"
            title={`gate spec sha256 ${state.gate_spec_hash} — compare with core/gate_spec.lock.json`}
          >
            gate spec {state.gate_spec_hash.slice(0, 12)}…
          </p>
        )}
      </section>

      <section>
        <SectionHead
          // The trial count is the promoter's own, not the number of rows this
          // endpoint returned. They differed by two, and the smaller one was
          // the one being used as a multiple-testing denominator.
          eyebrow={`${state.trial_count ?? attempts.length} trials · ${attempts.length} shown · ${attempts.filter((a) => a.installed).length} installed`}
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

/** One `label value` pair on the statistics line. Missing renders as an
 *  em-dash, never as zero — the house rule, and the reason `lib/format` exists. */
function Stat({
  label,
  value,
  digits,
  tone,
  title,
}: {
  label: string;
  value: number | null | undefined;
  digits: number;
  tone?: string;
  title?: string;
}) {
  return (
    <div className="flex gap-1.5" title={title}>
      <dt className="uppercase">{label}</dt>
      <dd className={tone ?? 'text-ink-2'}>
        {value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)}
      </dd>
    </div>
  );
}

const gateColumns: Column<Gate>[] = [
  {
    key: 'state',
    head: '',
    width: '4.5rem',
    // **"Cannot be measured" and "measured and lost" are different claims and
    // rendered identically.** The market comparisons read NaN until the live
    // loop has recorded enough quotes and arrived here as `passed: false` — a
    // red FAIL for a question nobody has asked yet. A neutral chip says the
    // gate is open; red stays for gates that were computed and lost.
    render: (g) =>
      g.measured === false ? (
        <Chip
          tone="neutral"
          title={g.note ?? 'not computable yet — this gate read NaN, it did not lose'}
        >
          open
        </Chip>
      ) : (
        <Chip tone={g.passed ? 'pass' : 'fail'} title={g.note}>
          {g.passed ? 'pass' : 'fail'}
        </Chip>
      ),
  },
  {
    key: 'name',
    head: 'gate',
    // The one sentence on what a gate means and why it has that threshold is
    // written in `core/metrics.py` and, until 2026-10-20, never sent. The
    // dotted underline is there so a reader knows there is something to hover;
    // a title nobody can see is a title nobody reads.
    render: (g) => (
      <span
        className={`font-mono ${g.note ? 'cursor-help underline decoration-dotted decoration-ink-3 underline-offset-2' : ''}`}
        title={g.note}
      >
        {g.name}
      </span>
    ),
  },
  {
    key: 'value',
    head: 'value',
    numeric: true,
    render: (g) =>
      g.measured === false ? (
        <span className="text-ink-3" title={g.note ?? 'not measured'}>
          —
        </span>
      ) : (
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
  {
    key: 'when',
    head: 'when',
    // Secondary on a phone: the version IS this instant in UTC, so the column
    // is a second reading of the same number and the first thing that should
    // give way at 375px.
    secondary: true,
    render: (a) => <span className="font-mono">{stamp(a.created_at)}</span>,
  },
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
  {
    key: 'folds',
    head: 'agree',
    numeric: true,
    // **A count of agreeing folds means nothing without its denominator.**
    // Anchored calendar blocks accumulate, so "5 of 10" is a 62% event under
    // the null where "5 of 6" is a 10.9% one — and this column printed the
    // numerator alone, which sorted a perfect 6/6 below a 25/29.
    render: (a) =>
      a.folds_positive == null ? '—' : (
        <span className="font-mono">
          {a.folds_positive}
          <span className="text-ink-3">/{a.folds ?? '?'}</span>
        </span>
      ),
  },
  {
    key: 'windows',
    head: 'windows',
    numeric: true,
    secondary: true,
    // `?? 0` rendered a missing measurement as the number zero — a claim that
    // the candidate evaluated nothing, which is a far stronger statement than
    // "the ledger row does not carry it".
    render: (a) => (a.windows_evaluated == null ? '—' : a.windows_evaluated.toLocaleString()),
  },
  {
    key: 'failed',
    head: 'failed gates',
    // `break-words` on the list: a run of failed gate names is one long
    // unbroken string of identifiers, which on a phone pushed the row to 781px
    // and made the whole table scroll sideways to read something that wraps
    // perfectly well.
    render: (a) =>
      a.failed_gates.length === 0 ? (
        <span className="text-ink-3">none</span>
      ) : (
        <span className="break-words font-mono text-micro text-fail">
          {a.failed_gates.join(', ')}
        </span>
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
