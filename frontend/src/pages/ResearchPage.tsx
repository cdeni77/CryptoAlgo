import { useState } from 'react';
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import {
  getResearchFeatures,
  getResearchJobLogs,
  getResearchJobs,
  getResearchRuns,
  getResearchScripts,
  getResearchSummary,
  launchResearchJob,
} from '../api/researchApi';
import { hasApiToken } from '../api/client';
import { Empty, ErrorBlock, Freshness, Panel } from '../components/StateBlock';
import { usePolling } from '../hooks/usePolling';
import { Health, ResearchCoinHealth } from '../types';

/**
 * Per-instrument health, run history, and the job runner.
 *
 * Rewritten because what it displayed was not measured. The coin cards showed an
 * AUC — a classification metric, on a model that regresses net return, which the
 * API now leaves null — and a "readiness tier" read from an artifact of a deleted
 * pipeline, so it was "UNKNOWN" for everything. The api layer then *invented* a
 * tier from a boolean and a position scale from that invented tier, so the screen
 * displayed a readiness grade and a recommended size the backend never computed.
 *
 * What it shows now is the comparison the model can be held to: the edge
 * `decide()` claimed before each trade, against the edge the trades earned. The
 * gates themselves live on the Model page — this is the after, that is the before.
 */

const CARD = 'glass-card rounded-xl p-5';
const LABEL = 'text-tx-muted text-[11px] font-medium tracking-widest uppercase';

const HEALTH_STYLE: Record<Health, { text: string; bg: string; border: string }> = {
  healthy: { text: 'text-accent-emerald', bg: 'bg-accent-emerald/10', border: 'border-accent-emerald/30' },
  watch: { text: 'text-accent-amber', bg: 'bg-accent-amber/10', border: 'border-accent-amber/30' },
  at_risk: { text: 'text-accent-rose', bg: 'bg-accent-rose/10', border: 'border-accent-rose/30' },
  unknown: { text: 'text-tx-muted', bg: 'bg-surface-3', border: 'border-[rgba(56,189,248,0.08)]' },
};

function HealthBadge({ health }: { health: Health }) {
  const s = HEALTH_STYLE[health] ?? HEALTH_STYLE.unknown;
  return (
    <span className={`rounded border px-2 py-0.5 text-[10px] font-medium ${s.bg} ${s.text} ${s.border}`}>
      {health === 'at_risk' ? 'at risk' : health}
    </span>
  );
}

const bps = (v: number | null) => (v === null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}bp`);
const pct = (v: number | null, digits = 1) => (v === null ? '—' : `${(v * 100).toFixed(digits)}%`);

function CoinCard({
  coin,
  selected,
  onSelect,
}: {
  coin: ResearchCoinHealth;
  selected: boolean;
  onSelect: () => void;
}) {
  const delta = coin.calibration.delta_bps;
  return (
    <button
      onClick={onSelect}
      title={coin.health_reason ?? undefined}
      className={`glass-card glass-card-hover rounded-xl p-4 text-left transition-all ${
        selected ? 'border border-accent-cyan/30' : ''
      }`}
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-semibold text-tx-primary">{coin.coin}</span>
        <HealthBadge health={coin.health} />
      </div>
      <div className="space-y-1 font-mono text-xs">
        <div className="flex justify-between">
          <span className="text-tx-muted">expected</span>
          <span className="text-tx-secondary tabular-nums">{bps(coin.expected_net_bps)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-tx-muted">realised</span>
          <span className="text-tx-secondary tabular-nums">{bps(coin.realised_net_bps)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-tx-muted">gap</span>
          <span
            className={`tabular-nums ${
              delta === null
                ? 'text-tx-muted'
                : delta < -8
                  ? 'text-accent-rose'
                  : 'text-accent-emerald'
            }`}
          >
            {bps(delta)}
          </span>
        </div>
        <div className="flex justify-between border-t border-[rgba(56,189,248,0.06)] pt-1">
          <span className="text-tx-muted">signals</span>
          <span className="text-tx-secondary tabular-nums">
            {coin.signals_passed_gates}/{coin.signals_total}
          </span>
        </div>
      </div>
    </button>
  );
}

export default function ResearchPage() {
  const summary = usePolling(getResearchSummary, 20_000);
  const runs = usePolling(() => getResearchRuns(30), 30_000);
  const scripts = usePolling(getResearchScripts, 300_000);
  const jobs = usePolling(() => getResearchJobs(10), 10_000);

  const [coin, setCoin] = useState('ETH');
  const features = usePolling(() => getResearchFeatures(coin), 60_000, [coin]);

  const [script, setScript] = useState('');
  const [args, setArgs] = useState('');
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<Error | null>(null);
  const [logsPid, setLogsPid] = useState<number | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [logsError, setLogsError] = useState<Error | null>(null);

  const scriptOptions = scripts.data?.scripts ?? [];
  const chosenScript = script || scriptOptions[0]?.name || '';

  async function launch() {
    if (!chosenScript) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      await launchResearchJob(chosenScript, args.trim() ? args.trim().split(/\s+/) : []);
      setArgs('');
      jobs.refresh();
    } catch (caught) {
      setLaunchError(caught instanceof Error ? caught : new Error(String(caught)));
    } finally {
      setLaunching(false);
    }
  }

  async function viewLogs(pid: number) {
    setLogsPid(pid);
    setLogsError(null);
    try {
      setLogs((await getResearchJobLogs(pid, 200)).logs);
    } catch (caught) {
      setLogs([]);
      setLogsError(caught instanceof Error ? caught : new Error(String(caught)));
    }
  }

  const kpis = summary.data?.kpis;

  return (
    <div className="w-full space-y-5 p-6">
      {/* ---- universe calibration ---------------------------------------- */}
      <div className={CARD}>
        <div className="mb-1 flex items-center justify-between">
          <span className={LABEL}>Edge calibration</span>
          <Freshness
            lastUpdated={summary.lastUpdated}
            refreshing={summary.refreshing}
            error={summary.error}
          />
        </div>
        <p className="mb-4 max-w-3xl text-[11px] leading-relaxed text-tx-muted">
          The model states an expected net edge in basis points before every trade.
          That claim is checkable, and the gap between it and what the trades
          earned is the number worth watching: a model that overstates its edge
          over-sizes every position clearing the conviction floor, which loses
          money quietly rather than visibly.
        </p>

        <Panel state={summary} loadingLabel="Reading signals and trades">
          {(data) => (
            <div className="grid grid-cols-2 gap-5 sm:grid-cols-4 lg:grid-cols-7">
              <div>
                <div className={LABEL}>Expected</div>
                <div className="mt-1 font-mono text-lg text-tx-primary tabular-nums">
                  {bps(data.kpis.expected_net_bps)}
                </div>
              </div>
              <div>
                <div className={LABEL}>Realised</div>
                <div className="mt-1 font-mono text-lg text-tx-primary tabular-nums">
                  {bps(data.kpis.realised_net_bps)}
                </div>
              </div>
              <div>
                <div className={LABEL}>Gap</div>
                <div
                  className={`mt-1 font-mono text-lg tabular-nums ${
                    data.kpis.calibration_delta_bps === null
                      ? 'text-tx-muted'
                      : data.kpis.calibration_delta_bps < -8
                        ? 'text-accent-rose'
                        : 'text-accent-emerald'
                  }`}
                >
                  {bps(data.kpis.calibration_delta_bps)}
                </div>
              </div>
              <div>
                <div className={LABEL}>Gate pass</div>
                <div className="mt-1 font-mono text-lg text-tx-primary tabular-nums">
                  {pct(data.kpis.gate_pass_rate)}
                </div>
                <div className="mt-0.5 text-[11px] text-tx-muted">
                  {data.kpis.signals_passed_gates}/{data.kpis.signals_total} signals
                </div>
              </div>
              <div>
                <div className={LABEL}>Win rate</div>
                <div className="mt-1 font-mono text-lg text-tx-primary tabular-nums">
                  {pct(data.kpis.win_rate_realized)}
                </div>
                <div className="mt-0.5 text-[11px] text-tx-muted">
                  {data.kpis.trades_closed} closed
                </div>
              </div>
              <div>
                <div className={LABEL}>Carry share</div>
                <div className="mt-1 font-mono text-lg text-tx-primary tabular-nums">
                  {pct(data.kpis.expected_carry_share)}
                </div>
                <div className="mt-0.5 text-[11px] text-tx-muted">of expected edge</div>
              </div>
              <div>
                <div className={LABEL}>Model</div>
                <div className="mt-1 truncate font-mono text-xs text-tx-secondary">
                  {data.kpis.model_version ?? 'none promoted'}
                </div>
                <div className="mt-0.5 text-[11px] text-tx-muted">
                  {data.kpis.model_age_hours === null
                    ? '—'
                    : `${data.kpis.model_age_hours.toFixed(0)}h old`}
                  {data.kpis.model_forced && (
                    <span className="ml-1.5 text-accent-amber">forced</span>
                  )}
                </div>
              </div>
            </div>
          )}
        </Panel>

        {kpis && kpis.kill_switch_status === 'quarantined' && (
          <div className="mt-4 rounded-lg border border-accent-rose/25 bg-accent-rose/5 px-3 py-2 text-[11px] leading-snug text-accent-rose">
            The live model is quarantined: realised paper results have breached the
            monitor’s thresholds, and the orchestrator has stopped writing signals.
          </div>
        )}
      </div>

      {/* ---- per instrument ---------------------------------------------- */}
      <div>
        <div className="mb-3 flex items-center justify-between">
          <span className={LABEL}>Per instrument</span>
        </div>
        <Panel
          state={summary}
          emptyWhen={(d) => d.coins.length === 0}
          emptyMessage="No instruments have signals yet."
          emptyHint="Run the pipeline, then scripts.signals."
        >
          {(data) => (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                {data.coins.map((c) => (
                  <CoinCard
                    key={c.coin}
                    coin={c}
                    selected={coin === c.coin}
                    onSelect={() => setCoin(c.coin)}
                  />
                ))}
              </div>
              {/* The reason a grade was reached, for the selected instrument.
                  A badge alone cannot say "not enough observations yet", which
                  is the correct answer most of the time early on. */}
              {(() => {
                const chosen = data.coins.find((c) => c.coin === coin);
                if (!chosen?.health_reason) return null;
                return (
                  <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-tx-muted">
                    <span>
                      <span className="text-tx-secondary">{chosen.coin}</span>:{' '}
                      {chosen.health_reason}
                    </span>
                    {chosen.top_gate_reason && (
                      <span>
                        most common block:{' '}
                        <span className="font-mono text-tx-secondary">
                          {chosen.top_gate_reason}
                        </span>
                      </span>
                    )}
                    {chosen.mean_cost_bps !== null && (
                      <span>
                        round trip:{' '}
                        <span className="font-mono text-tx-secondary tabular-nums">
                          {chosen.mean_cost_bps.toFixed(1)}bp
                        </span>
                      </span>
                    )}
                  </div>
                );
              })()}
            </>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* ---- signal distribution --------------------------------------- */}
        <div className={CARD}>
          <div className="mb-4 flex items-center justify-between">
            <span className={LABEL}>Signal distribution — {coin}</span>
          </div>
          <Panel
            state={features}
            emptyWhen={(d) => d.signal_distribution.every((s) => s.value === 0)}
            emptyMessage={`No signals recorded for ${coin} yet.`}
          >
            {(data) => (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={data.signal_distribution} margin={{ left: 0, right: 8 }}>
                  <XAxis
                    dataKey="label"
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: '#64748b', fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#111827',
                      border: '1px solid rgba(56,189,248,0.15)',
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                  />
                  <Bar dataKey="value" fill="#38bdf8" opacity={0.75} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Panel>
          <p className="mt-3 text-[11px] leading-snug text-tx-muted">
            Feature importances are per-model, not per-instrument — the model is a
            pooled panel with one feature set across the universe. They live on the
            Model page.
          </p>
        </div>

        {/* ---- job runner ------------------------------------------------- */}
        <div className={`${CARD} space-y-4`}>
          <span className={LABEL}>Run a script</span>

          {!hasApiToken() && (
            <div className="rounded-lg border border-accent-amber/25 bg-accent-amber/5 px-3 py-2 text-[11px] leading-snug text-accent-amber">
              Launching is disabled: no API token is configured in this build. Set{' '}
              <span className="font-mono">API_TOKEN</span> on the backend and{' '}
              <span className="font-mono">VITE_API_TOKEN</span> here to the same
              value.
            </div>
          )}

          <div className="space-y-3">
            <div>
              <label htmlFor="script" className="mb-1.5 block text-xs text-tx-muted">
                Script
              </label>
              <select
                id="script"
                value={chosenScript}
                onChange={(e) => setScript(e.target.value)}
                disabled={!scriptOptions.length}
                className="w-full rounded-lg border border-[rgba(56,189,248,0.12)] bg-surface-2 px-3 py-2 text-sm text-tx-primary focus:border-accent-cyan/40 focus:outline-none disabled:opacity-40"
              >
                {scriptOptions.length === 0 && <option>loading…</option>}
                {scriptOptions.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="args" className="mb-1.5 block text-xs text-tx-muted">
                Arguments
              </label>
              <input
                id="args"
                type="text"
                value={args}
                onChange={(e) => setArgs(e.target.value)}
                placeholder="--venue coinbase --periods 6"
                className="w-full rounded-lg border border-[rgba(56,189,248,0.12)] bg-surface-2 px-3 py-2 font-mono text-sm text-tx-primary placeholder-tx-muted focus:border-accent-cyan/40 focus:outline-none"
              />
              {/* The backend validates these and rejects rather than sanitises,
                  so saying what is allowed here avoids a confusing 400. */}
              <p className="mt-1 text-[10px] leading-snug text-tx-muted">
                Long flags only. Filesystem paths are refused — the scripts resolve
                their own store and config locations.
              </p>
            </div>
            <button
              onClick={launch}
              disabled={launching || !chosenScript || !hasApiToken()}
              className="w-full rounded-lg border border-accent-cyan/30 bg-accent-cyan/15 py-2 text-sm font-medium text-accent-cyan transition-colors hover:bg-accent-cyan/20 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {launching ? 'Starting…' : 'Run'}
            </button>
            {launchError && <ErrorBlock error={launchError} compact />}
          </div>

          <div className="space-y-1 border-t border-[rgba(56,189,248,0.08)] pt-3">
            <Panel
              state={jobs}
              emptyWhen={(d) => d.length === 0}
              emptyMessage="No jobs started in this session."
            >
              {(data) => (
                <>
                  {data.slice(0, 6).map((j) => (
                    <div
                      key={j.pid}
                      className="flex items-center justify-between gap-2 border-b border-[rgba(56,189,248,0.06)] py-1.5 text-xs last:border-0"
                    >
                      <span className="truncate font-mono text-tx-secondary">{j.job}</span>
                      <span className="flex-shrink-0 font-mono text-[10px] text-tx-muted">
                        {new Date(j.launched_at).toLocaleTimeString('en-GB', { hour12: false })}
                      </span>
                      <button
                        onClick={() => viewLogs(j.pid)}
                        className="flex-shrink-0 text-[10px] text-accent-cyan hover:underline"
                      >
                        logs
                      </button>
                    </div>
                  ))}
                </>
              )}
            </Panel>
          </div>
        </div>
      </div>

      {/* ---- run history -------------------------------------------------- */}
      <div className={CARD}>
        <div className="mb-1 flex items-center justify-between">
          <span className={LABEL}>Retrain history</span>
          <Freshness lastUpdated={runs.lastUpdated} refreshing={runs.refreshing} error={runs.error} />
        </div>
        <p className="mb-4 text-[11px] leading-relaxed text-tx-muted">
          Real attempts, from the orchestrator’s run table joined to the promotion
          ledger. This table used to invent three runs per signal — a train, an
          optimize and a validate, with fabricated durations and a status hardcoded
          to success — none of which had happened.
        </p>

        <Panel
          state={runs}
          emptyWhen={(d) => d.length === 0}
          emptyMessage="No retrain attempts recorded."
          emptyHint="The orchestrator records one per cadence; scripts.promote records one per evaluation."
        >
          {(data) => (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[rgba(56,189,248,0.08)]">
                    {['Version', 'Type', 'Status', 'Sharpe', 'Trades', 'Blocked by', 'Duration', 'Started'].map(
                      (h) => (
                        <th
                          key={h}
                          className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-tx-muted"
                        >
                          {h}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {data.map((r) => (
                    <tr
                      key={r.id}
                      className="border-b border-[rgba(56,189,248,0.04)] hover:bg-[rgba(56,189,248,0.03)]"
                    >
                      <td className="px-3 py-2 font-mono text-[11px] text-tx-secondary">
                        {r.artifacts_version ?? '—'}
                      </td>
                      <td className="px-3 py-2 font-mono text-tx-secondary">{r.run_type}</td>
                      <td className="px-3 py-2">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] ${
                            r.promoted
                              ? 'bg-accent-emerald/10 text-accent-emerald'
                              : r.status === 'blocked'
                                ? 'bg-accent-amber/10 text-accent-amber'
                                : 'bg-accent-rose/10 text-accent-rose'
                          }`}
                        >
                          {r.forced ? 'forced' : r.promoted ? 'promoted' : r.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-tx-secondary tabular-nums">
                        {r.sharpe === null ? '—' : r.sharpe.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 font-mono text-tx-secondary tabular-nums">
                        {r.trades ?? '—'}
                      </td>
                      <td
                        className="max-w-[220px] truncate px-3 py-2 font-mono text-[10px] text-tx-muted"
                        title={r.failed_gates.join(', ') || r.error || ''}
                      >
                        {r.failed_gates.join(', ') || r.error || '—'}
                      </td>
                      <td className="px-3 py-2 font-mono text-tx-muted tabular-nums">
                        {r.duration_seconds === null ? '—' : `${Math.round(r.duration_seconds)}s`}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 font-mono text-tx-muted">
                        {new Date(r.started_at).toLocaleString('en-GB', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                          hour12: false,
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      {/* ---- logs --------------------------------------------------------- */}
      {logsPid !== null && (
        <div className={CARD}>
          <div className="mb-3 flex items-center justify-between">
            <span className={LABEL}>Logs — PID {logsPid}</span>
            <div className="flex gap-3">
              <button
                onClick={() => viewLogs(logsPid)}
                className="text-xs text-accent-cyan hover:underline"
              >
                Refresh
              </button>
              <button
                onClick={() => {
                  setLogsPid(null);
                  setLogs([]);
                  setLogsError(null);
                }}
                className="text-xs text-tx-muted transition-colors hover:text-tx-secondary"
              >
                Close
              </button>
            </div>
          </div>
          {logsError ? (
            <ErrorBlock error={logsError} onRetry={() => viewLogs(logsPid)} />
          ) : logs.length ? (
            <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-lg bg-surface-2 p-4 font-mono text-[11px] text-tx-secondary">
              {logs.join('\n')}
            </pre>
          ) : (
            <Empty message="No output yet." hint="A job that has just started may not have written anything." />
          )}
        </div>
      )}
    </div>
  );
}
