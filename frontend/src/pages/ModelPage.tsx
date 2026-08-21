import { useState } from 'react';
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { getFeatureImportance, getLiveModel, getPromotionHistory } from '../api/modelApi';
import { launchResearchJob } from '../api/researchApi';
import { hasApiToken } from '../api/client';
import GateTable from '../components/GateTable';
import { Empty, ErrorBlock, Freshness, Panel, Spinner } from '../components/StateBlock';
import { usePolling } from '../hooks/usePolling';
import { PathDistributionSummary, PromotionRecord } from '../types';

/**
 * What is trading, why it was allowed to, and what else was tried.
 *
 * This page did not exist. The dashboard's readiness display came from
 * `optimization_results/*_validation.json` — an artifact of a pipeline that has
 * been deleted — so it read "UNKNOWN" for every instrument while the real
 * decision, the promotion gates, was visible only in a container's stdout.
 *
 * The promote button launches `scripts.promote` through the authenticated launch
 * endpoint. It does not promote anything itself: the only thing allowed to install
 * a model is the thing that runs the gates.
 */

const CARD = 'glass-card rounded-xl p-5';
const LABEL = 'text-tx-muted text-[11px] font-medium tracking-widest uppercase';

function Metric({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: 'neutral' | 'good' | 'bad' | 'warn';
}) {
  const colour =
    tone === 'good'
      ? 'text-accent-emerald'
      : tone === 'bad'
        ? 'text-accent-rose'
        : tone === 'warn'
          ? 'text-accent-amber'
          : 'text-tx-primary';
  return (
    <div>
      <div className={LABEL}>{label}</div>
      <div className={`mt-1 font-mono text-lg tabular-nums ${colour}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] leading-snug text-tx-muted">{hint}</div>}
    </div>
  );
}

const num = (v: number | null | undefined, digits = 2, suffix = '') =>
  v === null || v === undefined ? '—' : `${v.toFixed(digits)}${suffix}`;
const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`;
const money = (v: number | null | undefined) =>
  v === null || v === undefined
    ? '—'
    : `${v < 0 ? '-' : '+'}$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;

function Distribution({
  label,
  dist,
  hint,
}: {
  label: string;
  dist: PathDistributionSummary | null;
  hint?: string;
}) {
  if (!dist || dist.median === null) {
    return <Metric label={label} value="—" hint="not measured" />;
  }
  return (
    <div>
      <div className={LABEL}>{label}</div>
      <div className="mt-1 font-mono text-lg tabular-nums text-tx-primary">
        {num(dist.median)}
      </div>
      {/* The interval, not just the point estimate — the width is the finding. */}
      <div className="mt-0.5 font-mono text-[11px] text-tx-muted tabular-nums">
        p05 {num(dist.p05)} · p95 {num(dist.p95)}
        {dist.n ? ` · n=${dist.n}` : ''}
      </div>
      {hint && <div className="mt-0.5 text-[11px] leading-snug text-tx-muted">{hint}</div>}
    </div>
  );
}

function Provenance({ record }: { record: PromotionRecord }) {
  const p = record.provenance;
  const rows: [string, string][] = [
    ['version', record.version],
    ['feature set', p.feature_set_hash ?? '—'],
    ['features', p.n_features === null ? '—' : String(p.n_features)],
    ['heads', p.heads.length ? p.heads.join(', ') : '—'],
    ['horizon', p.horizon_bars === null ? '—' : `${p.horizon_bars}h`],
    ['cost config', p.cost_config_version ?? '—'],
    ['trained', p.trained_at ? new Date(p.trained_at).toLocaleString('en-GB') : '—'],
    ['train window', p.train_start && p.train_end ? `${p.train_start.slice(0, 10)} → ${p.train_end.slice(0, 10)}` : '—'],
    ['train rows', p.train_rows === null ? '—' : p.train_rows.toLocaleString('en-US')],
    ['effective obs', p.effective_observations === null ? '—' : p.effective_observations.toFixed(0)],
    ['instruments', p.symbols.length ? p.symbols.join(' · ') : '—'],
  ];

  return (
    <div>
      <dl className="space-y-1.5">
        {rows.map(([key, value]) => (
          <div key={key} className="flex items-baseline justify-between gap-3 text-xs">
            <dt className="text-tx-muted">{key}</dt>
            <dd className="truncate font-mono text-tx-secondary" title={value}>
              {value}
            </dd>
          </div>
        ))}
      </dl>

      {p.uses_symbol_identity && (
        <div className="mt-3 rounded-lg border border-accent-amber/25 bg-accent-amber/5 px-3 py-2 text-[11px] leading-snug text-accent-amber">
          This model uses instrument identity as a feature. Identity alone scored
          an information coefficient of +0.54 on random walks, so a ranking built
          on it may be reproducing which instrument is which rather than when to
          trade it.
        </div>
      )}

      {p.effective_observations !== null && p.effective_observations < 200 && (
        <div className="mt-3 rounded-lg border border-accent-amber/25 bg-accent-amber/5 px-3 py-2 text-[11px] leading-snug text-accent-amber">
          {p.effective_observations.toFixed(0)} effective observations. Overlapping
          labels share information, so the row count overstates the sample — every
          statistic below is wider than it looks.
        </div>
      )}
    </div>
  );
}

export default function ModelPage() {
  const live = usePolling(getLiveModel, 30_000);
  const history = usePolling(() => getPromotionHistory(30), 60_000);
  const [head, setHead] = useState('price');
  const importance = usePolling(() => getFeatureImportance(head), 120_000, [head]);

  const [selected, setSelected] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<Error | null>(null);
  const [launchNote, setLaunchNote] = useState<string | null>(null);

  async function evaluateCandidate() {
    setLaunching(true);
    setLaunchError(null);
    setLaunchNote(null);
    try {
      const job = await launchResearchJob('promote', []);
      setLaunchNote(
        `Started as PID ${job.pid}. It trains, walk-forward backtests, bootstraps and gates a ` +
          `candidate, which takes minutes — the result appears here when it lands.`,
      );
      history.refresh();
    } catch (caught) {
      setLaunchError(caught instanceof Error ? caught : new Error(String(caught)));
    } finally {
      setLaunching(false);
    }
  }

  const record =
    (selected && history.data?.records.find((r) => r.version === selected)) ||
    live.data?.live ||
    null;

  return (
    <div className="max-w-[1600px] space-y-5 p-6">
      {/* ---- what is live ------------------------------------------------ */}
      <div className={CARD}>
        <div className="mb-4 flex items-center justify-between">
          <span className={LABEL}>Live model</span>
          <Freshness
            lastUpdated={live.lastUpdated}
            refreshing={live.refreshing}
            error={live.error}
          />
        </div>

        <Panel state={live} loadingLabel="Reading the promotion ledger">
          {(data) => (
            <>
              {!data.has_model && (
                <Empty
                  message="Nothing is promoted."
                  hint="Run the evaluation below, or `python -m scripts.promote` in the trader. A candidate installs only if every gate passes."
                />
              )}

              {data.unrecorded_artifact && (
                <div className="mb-4 rounded-lg border border-accent-rose/25 bg-accent-rose/5 px-3 py-2.5">
                  <div className="text-xs font-medium uppercase tracking-widest text-accent-rose">
                    Installed outside the gates
                  </div>
                  <div className="mt-1 text-[11px] leading-snug text-tx-secondary">
                    A model artifact exists with no promotion record beside it, so
                    nothing here can say what it was measured against. Re-evaluate
                    it before trusting anything it produces.
                  </div>
                </div>
              )}

              {data.live && (
                <>
                  <div className="mb-4 flex flex-wrap items-center gap-2">
                    <span className="rounded border border-accent-cyan/30 bg-accent-cyan/10 px-2 py-0.5 font-mono text-[11px] text-accent-cyan">
                      {data.live.version}
                    </span>
                    {data.live.forced && (
                      <span
                        className="rounded border border-accent-amber/30 bg-accent-amber/10 px-2 py-0.5 text-[11px] text-accent-amber"
                        title={data.live.force_reason ?? undefined}
                      >
                        forced past {data.live.failed_gates.length} gate
                        {data.live.failed_gates.length === 1 ? '' : 's'}
                      </span>
                    )}
                    {data.kill_switch.status === 'quarantined' && (
                      <span className="rounded border border-accent-rose/30 bg-accent-rose/10 px-2 py-0.5 text-[11px] text-accent-rose">
                        quarantined
                      </span>
                    )}
                    <span className="font-mono text-[11px] text-tx-muted">
                      {data.trials_to_date} candidate
                      {data.trials_to_date === 1 ? '' : 's'} evaluated
                    </span>
                  </div>

                  {data.live.forced && data.live.force_reason && (
                    <div className="mb-4 rounded-lg border border-accent-amber/25 bg-accent-amber/5 px-3 py-2 text-[11px] leading-snug text-accent-amber">
                      Override reason: {data.live.force_reason}
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-6">
                    <Metric
                      label="OOS Sharpe"
                      value={num(data.live.backtest.sharpe)}
                      tone={
                        data.live.backtest.sharpe === null
                          ? 'neutral'
                          : data.live.backtest.sharpe > 0
                            ? 'good'
                            : 'bad'
                      }
                    />
                    <Metric label="Trades" value={String(data.live.backtest.trades ?? '—')} />
                    <Metric
                      label="Net PnL"
                      value={money(data.live.backtest.net_pnl)}
                      tone={
                        data.live.backtest.net_pnl == null
                          ? 'neutral'
                          : data.live.backtest.net_pnl >= 0 ? 'good' : 'bad'
                      }
                      hint={`price ${money(data.live.backtest.price_pnl)} · funding ${money(
                        data.live.backtest.funding_pnl,
                      )} · fees ${money(data.live.backtest.fees != null ? -data.live.backtest.fees : null)}`}
                    />
                    <Metric
                      label="Max drawdown"
                      value={pct(data.live.backtest.max_drawdown)}
                      tone={
                        data.live.backtest.max_drawdown == null
                          ? 'neutral'
                          : data.live.backtest.max_drawdown > 0.25 ? 'bad' : 'neutral'
                      }
                    />
                    <Metric
                      label="Carry share"
                      value={pct(data.live.backtest.carry_contribution)}
                      hint="funding as a share of gross profit"
                    />
                    <Metric
                      label="Exit participation"
                      value={pct(data.live.backtest.max_exit_participation)}
                      tone={
                        (data.live.backtest.max_exit_participation ?? 0) > 0.2
                          ? 'bad'
                          : 'neutral'
                      }
                      hint="largest share of a bar an exit took"
                    />
                  </div>
                </>
              )}
            </>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* ---- gates ---------------------------------------------------- */}
        <div className={`${CARD} lg:col-span-2`}>
          <div className="mb-4 flex items-center justify-between">
            <span className={LABEL}>
              Promotion gates
              {selected && record && !record.is_live && (
                <span className="ml-2 font-mono normal-case tracking-normal text-accent-cyan">
                  {record.version}
                </span>
              )}
            </span>
            {selected && (
              <button
                onClick={() => setSelected(null)}
                className="text-[11px] text-tx-muted transition-colors hover:text-tx-secondary"
              >
                back to live
              </button>
            )}
          </div>

          {record ? (
            <GateTable gates={record.gates} />
          ) : live.loading ? (
            <Spinner label="Reading gate results" />
          ) : (
            <Empty
              message="No gate results yet."
              hint="Every candidate is measured against the same set; a gate with no measurement fails."
            />
          )}
        </div>

        {/* ---- provenance ----------------------------------------------- */}
        <div className={CARD}>
          <div className="mb-4 flex items-center justify-between">
            <span className={LABEL}>Provenance</span>
          </div>
          {record ? (
            <Provenance record={record} />
          ) : (
            <Empty message="No model to describe." />
          )}
        </div>
      </div>

      {/* ---- simulation ------------------------------------------------- */}
      {record && (
        <div className={CARD}>
          <div className="mb-1 flex items-center justify-between">
            <span className={LABEL}>Simulation</span>
          </div>
          <p className="mb-4 max-w-3xl text-[11px] leading-relaxed text-tx-muted">
            A single backtest number is one draw from a distribution nobody measured,
            chosen from however many configurations were tried, on the one price path
            history took. These remove those excuses one at a time. The synthetic
            panels are the exception worth stating: a generator contains only the
            structure calibrated into it, so they measure robustness and sizing,
            never edge.
          </p>
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-6">
            <Distribution
              label="Bootstrap Sharpe"
              dist={record.simulation.bootstrap_sharpe}
              hint="same trades, resampled order"
            />
            <Distribution
              label="Per-period Sharpe"
              dist={record.simulation.per_period_sharpe}
              hint="one per walk-forward stretch"
            />
            <Distribution
              label="Synthetic Sharpe"
              dist={record.simulation.synthetic_sharpe}
              hint="paths that did not happen"
            />
            <Metric
              label="P(positive)"
              value={pct(record.simulation.probability_positive)}
              tone={
                record.simulation.probability_positive == null
                  ? 'neutral'
                  : record.simulation.probability_positive >= 0.9 ? 'good' : 'warn'
              }
            />
            <Metric
              label="Risk of ruin"
              value={pct(record.simulation.risk_of_ruin)}
              tone={
                record.simulation.risk_of_ruin == null
                  ? 'neutral'
                  : record.simulation.risk_of_ruin > 0.05 ? 'bad' : 'good'
              }
              hint="paths hitting a 50% drawdown"
            />
            <Metric
              label="Parameter plateau"
              value={pct(record.simulation.parameter_plateau)}
              hint="neighbours that still work"
            />
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* ---- history --------------------------------------------------- */}
        <div className={CARD}>
          <div className="mb-4 flex items-center justify-between">
            <span className={LABEL}>Candidates</span>
            <Freshness lastUpdated={history.lastUpdated} refreshing={history.refreshing} />
          </div>
          <p className="mb-3 text-[11px] leading-relaxed text-tx-muted">
            Rejections are kept. The count of attempts is what the deflated Sharpe
            ratio discounts by, so a list of successes only would flatter whichever
            one survived.
          </p>

          <Panel
            state={history}
            emptyWhen={(d) => d.records.length === 0}
            emptyMessage="No candidates evaluated yet."
            loadingLabel="Reading the ledger"
          >
            {(data) => (
              <div className="max-h-80 space-y-1 overflow-y-auto">
                {data.records.map((r) => (
                  <button
                    key={r.version}
                    onClick={() => setSelected(r.version)}
                    className={`flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-2 text-left transition-colors ${
                      selected === r.version || (!selected && r.is_live)
                        ? 'border-accent-cyan/30 bg-accent-cyan/5'
                        : 'border-transparent hover:bg-[rgba(56,189,248,0.04)]'
                    }`}
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-mono text-[11px] text-tx-secondary">
                        {r.version}
                        {r.is_live && <span className="ml-2 text-accent-cyan">live</span>}
                      </span>
                      <span className="block truncate text-[10px] text-tx-muted">
                        {r.error
                          ? r.error
                          : r.failed_gates.length
                            ? `failed: ${r.failed_gates.join(', ')}`
                            : 'all gates passed'}
                      </span>
                    </span>
                    <span className="flex-shrink-0 text-right">
                      <span
                        className={`block font-mono text-xs tabular-nums ${
                          r.backtest.sharpe != null && r.backtest.sharpe >= 0
                            ? 'text-accent-emerald'
                            : 'text-accent-rose'
                        }`}
                      >
                        {num(r.backtest.sharpe)}
                      </span>
                      <span className="block text-[10px] text-tx-muted">
                        {r.promoted ? (r.forced ? 'forced' : 'promoted') : 'blocked'}
                      </span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </Panel>
        </div>

        {/* ---- evaluate a new candidate ---------------------------------- */}
        <div className={CARD}>
          <div className="mb-4">
            <span className={LABEL}>Evaluate a candidate</span>
          </div>
          <p className="mb-4 text-[11px] leading-relaxed text-tx-muted">
            Trains on the current feature panel, walk-forward backtests it,
            bootstraps the trades, stresses the costs, and installs it only if every
            gate passes. A blocked candidate leaves the live model alone and records
            why. This takes minutes, not seconds.
          </p>

          {!hasApiToken() && (
            <div className="mb-4 rounded-lg border border-accent-amber/25 bg-accent-amber/5 px-3 py-2 text-[11px] leading-snug text-accent-amber">
              No API token is configured in this build, so launching is disabled.
              Set <span className="font-mono">API_TOKEN</span> on the backend and{' '}
              <span className="font-mono">VITE_API_TOKEN</span> here to the same
              value.
            </div>
          )}

          <button
            onClick={evaluateCandidate}
            disabled={launching || !hasApiToken()}
            className="w-full rounded-lg border border-accent-cyan/30 bg-accent-cyan/15 py-2 text-sm font-medium text-accent-cyan transition-colors hover:bg-accent-cyan/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {launching ? 'Starting…' : 'Train and evaluate'}
          </button>

          {launchNote && (
            <div className="mt-3 rounded-lg border border-accent-cyan/25 bg-accent-cyan/5 px-3 py-2 text-[11px] leading-snug text-accent-cyan">
              {launchNote}
            </div>
          )}
          {launchError && (
            <div className="mt-3">
              <ErrorBlock error={launchError} compact />
            </div>
          )}

          {/* Kill switch — realised results on the model already trading, which
              is the one thing the gates cannot see. */}
          {live.data?.kill_switch && live.data.kill_switch.status !== 'not_evaluated' && (
            <div className="mt-5 border-t border-[rgba(56,189,248,0.08)] pt-4">
              <div className={`${LABEL} mb-2`}>Live monitor</div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <Metric
                  label="Status"
                  value={live.data.kill_switch.status}
                  tone={
                    live.data.kill_switch.status === 'quarantined' ? 'bad' : 'good'
                  }
                />
                <Metric
                  label="Win rate"
                  value={pct(live.data.kill_switch.win_rate)}
                  hint={`${live.data.kill_switch.trades ?? '—'} trades / ${
                    live.data.kill_switch.window_days ?? '—'
                  }d`}
                />
                <Metric
                  label="Drawdown"
                  value={pct(live.data.kill_switch.drawdown)}
                  tone={
                    live.data.kill_switch.drawdown == null
                      ? 'neutral'
                      : live.data.kill_switch.drawdown > 0.12 ? 'bad' : 'neutral'
                  }
                />
              </div>
              {live.data.kill_switch.reasons.length > 0 && (
                <ul className="mt-2 space-y-0.5">
                  {live.data.kill_switch.reasons.map((reason) => (
                    <li key={reason} className="font-mono text-[10px] text-accent-amber">
                      {reason}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ---- feature importance ----------------------------------------- */}
      <div className={CARD}>
        <div className="mb-4 flex items-center justify-between">
          <span className={LABEL}>Feature importance</span>
          <div className="flex gap-1">
            {['price', 'carry', 'dispersion'].map((h) => (
              <button
                key={h}
                onClick={() => setHead(h)}
                className={`rounded px-2 py-1 text-[11px] transition-colors ${
                  head === h
                    ? 'bg-accent-cyan/15 text-accent-cyan'
                    : 'text-tx-muted hover:text-tx-secondary'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        <Panel
          state={importance}
          emptyWhen={(d) => d.features.length === 0}
          emptyMessage="No importances available."
          emptyHint={importance.data?.unavailable_reason ?? undefined}
          loadingLabel="Loading the booster"
        >
          {(data) => (
            <ResponsiveContainer width="100%" height={Math.max(260, data.features.length * 18)}>
              <BarChart data={data.features} layout="vertical" margin={{ left: 0, right: 12 }}>
                <XAxis
                  type="number"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={190}
                  tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: '#111827',
                    border: '1px solid rgba(56,189,248,0.15)',
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                  formatter={(v: number) => [`${(v * 100).toFixed(2)}% of split gain`, 'importance']}
                />
                <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                  {data.features.map((f) => (
                    <Cell key={f.feature} fill="#38bdf8" opacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>
    </div>
  );
}
