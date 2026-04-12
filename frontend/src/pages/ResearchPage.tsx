import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { getResearchSummary, getResearchRuns, getResearchFeatures, getResearchScripts, launchResearchJob, getResearchJobs, getResearchJobLogs } from '../api/researchApi';
import { ResearchSummary, ResearchRun, ResearchFeatures, ResearchScriptInfo, ResearchJobLaunchResponse, ReadinessTier } from '../types';

const TIER_STYLE: Record<ReadinessTier, { bg: string; text: string; border: string; label: string }> = {
  FULL:    { bg: 'bg-accent-emerald/10', text: 'text-accent-emerald', border: 'border-accent-emerald/30', label: 'Full' },
  PILOT:   { bg: 'bg-accent-amber/10',   text: 'text-accent-amber',   border: 'border-accent-amber/30',   label: 'Pilot' },
  SHADOW:  { bg: 'bg-accent-cyan/10',    text: 'text-accent-cyan',    border: 'border-accent-cyan/30',    label: 'Shadow' },
  REJECT:  { bg: 'bg-accent-rose/10',    text: 'text-accent-rose',    border: 'border-accent-rose/30',    label: 'Reject' },
  UNKNOWN: { bg: 'bg-surface-3',         text: 'text-tx-muted',       border: 'border-[rgba(56,189,248,0.08)]', label: '—' },
};

function TierBadge({ tier }: { tier?: ReadinessTier }) {
  const t = tier ?? 'UNKNOWN';
  const s = TIER_STYLE[t];
  return (
    <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${s.bg} ${s.text} ${s.border}`}>
      {s.label}
    </span>
  );
}

export default function ResearchPage() {
  const [summary,  setSummary]  = useState<ResearchSummary | null>(null);
  const [runs,     setRuns]     = useState<ResearchRun[]>([]);
  const [features, setFeatures] = useState<ResearchFeatures | null>(null);
  const [scripts,  setScripts]  = useState<ResearchScriptInfo[]>([]);
  const [jobs,     setJobs]     = useState<ResearchJobLaunchResponse[]>([]);
  const [selCoin,  setSelCoin]  = useState('ETH');
  const [selScript,setSelScript]= useState('');
  const [args,     setArgs]     = useState('');
  const [launching,setLaunching]= useState(false);
  const [logs,     setLogs]     = useState<string[]>([]);
  const [logsJob,  setLogsJob]  = useState<number | null>(null);

  useEffect(() => {
    const load = async () => {
      try { setSummary(await getResearchSummary()); } catch { /* empty */ }
      try { setRuns(await getResearchRuns(30)); } catch { /* empty */ }
      try { const r = await getResearchScripts(); setScripts(r.scripts); if (!selScript && r.scripts.length) setSelScript(r.scripts[0].name); } catch { /* empty */ }
      try { setJobs(await getResearchJobs(10)); } catch { /* empty */ }
    };
    load();
    const id = setInterval(load, 20000);
    return () => clearInterval(id);
  }, [selScript]);

  useEffect(() => {
    if (!selCoin) return;
    getResearchFeatures(selCoin).then(setFeatures).catch(() => {});
  }, [selCoin]);

  async function handleLaunch() {
    if (!selScript) return;
    setLaunching(true);
    try {
      const argList = args.trim() ? args.trim().split(/\s+/) : [];
      await launchResearchJob(selScript, argList);
      setJobs(await getResearchJobs(10));
    } catch { /* empty */ } finally { setLaunching(false); }
  }

  async function handleViewLogs(pid: number) {
    setLogsJob(pid);
    try { const r = await getResearchJobLogs(pid, 100); setLogs(r.logs); } catch { /* empty */ }
  }

  const coins = summary?.coins ?? [];
  const topFeatures = (features?.feature_importance ?? []).slice(0, 15);

  return (
    <div className="p-6 space-y-5 max-w-[1600px]">
      {/* Coin health grid */}
      <div>
        <div className="text-tx-muted text-[11px] font-medium tracking-widest uppercase mb-3">Model Readiness</div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {coins.map(c => (
            <button
              key={c.coin}
              onClick={() => setSelCoin(c.coin)}
              className={`glass-card glass-card-hover rounded-xl p-4 text-left transition-all ${selCoin === c.coin ? 'border border-accent-cyan/30' : ''}`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-tx-primary text-sm">{c.coin}</span>
                <TierBadge tier={c.readiness_tier} />
              </div>
              <div className="space-y-1 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-tx-muted">AUC</span>
                  <span className="text-tx-secondary">{c.holdout_auc?.toFixed(3) ?? '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-tx-muted">Win %</span>
                  <span className="text-tx-secondary">{c.win_rate_realized > 0 ? `${(c.win_rate_realized*100).toFixed(1)}%` : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-tx-muted">Health</span>
                  <span className={c.health === 'healthy' ? 'text-accent-emerald' : c.health === 'watch' ? 'text-accent-amber' : 'text-accent-rose'}>
                    {c.health}
                  </span>
                </div>
              </div>
            </button>
          ))}
          {!coins.length && (
            <div className="col-span-5 text-center text-tx-muted py-8 text-sm">Loading research data…</div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Feature importance */}
        <div className="glass-card rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Feature Importance — {selCoin}</span>
          </div>
          {topFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 0, right: 8 }}>
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="feature" width={140} tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid rgba(56,189,248,0.15)', borderRadius: 8, fontSize: 11 }}
                  formatter={(v: number) => [v.toFixed(4), 'importance']}
                />
                <Bar dataKey="importance" fill="#38bdf8" opacity={0.75} radius={[0,3,3,0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-40 text-tx-muted text-sm">No feature data for {selCoin}</div>
          )}
        </div>

        {/* Job launcher */}
        <div className="glass-card rounded-xl p-5 space-y-4">
          <div className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Launch Research Job</div>
          <div className="space-y-3">
            <div>
              <label className="block text-tx-muted text-xs mb-1.5">Script</label>
              <select
                value={selScript}
                onChange={e => setSelScript(e.target.value)}
                className="w-full bg-surface-2 border border-[rgba(56,189,248,0.12)] rounded-lg px-3 py-2 text-tx-primary text-sm focus:outline-none focus:border-accent-cyan/40"
              >
                {scripts.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-tx-muted text-xs mb-1.5">Arguments (space-separated)</label>
              <input
                type="text"
                value={args}
                onChange={e => setArgs(e.target.value)}
                placeholder="--coin ETH --trials 100"
                className="w-full bg-surface-2 border border-[rgba(56,189,248,0.12)] rounded-lg px-3 py-2 text-tx-primary text-sm font-mono placeholder-tx-muted focus:outline-none focus:border-accent-cyan/40"
              />
            </div>
            <button
              onClick={handleLaunch}
              disabled={launching || !selScript}
              className="w-full py-2 rounded-lg bg-accent-cyan/15 border border-accent-cyan/30 text-accent-cyan text-sm font-medium hover:bg-accent-cyan/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {launching ? 'Launching…' : 'Launch'}
            </button>
          </div>
          {/* Recent jobs */}
          <div className="mt-2 space-y-1">
            {jobs.slice(0,5).map(j => (
              <div key={j.pid} className="flex items-center justify-between text-xs border-b border-[rgba(56,189,248,0.06)] py-1.5">
                <span className="font-mono text-tx-secondary">{j.job}</span>
                <span className="text-tx-muted font-mono">PID {j.pid}</span>
                <button onClick={() => handleViewLogs(j.pid)} className="text-accent-cyan hover:underline text-[10px]">logs</button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent runs */}
      <div className="glass-card rounded-xl p-5">
        <div className="text-tx-secondary text-xs font-medium tracking-widest uppercase mb-4">Recent Runs</div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[rgba(56,189,248,0.08)]">
                {['Coin','Type','Status','AUC','Tier','Duration','Finished'].map(h => (
                  <th key={h} className="text-left px-3 py-2 text-tx-muted font-medium tracking-wider uppercase text-[10px]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.length === 0 && (
                <tr><td colSpan={7} className="px-3 py-6 text-center text-tx-muted">No runs recorded</td></tr>
              )}
              {runs.map(r => (
                <tr key={r.id} className="border-b border-[rgba(56,189,248,0.04)] hover:bg-[rgba(56,189,248,0.03)]">
                  <td className="px-3 py-2 font-medium text-tx-primary">{r.coin}</td>
                  <td className="px-3 py-2 text-tx-secondary font-mono">{r.run_type}</td>
                  <td className="px-3 py-2">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${r.status === 'success' ? 'text-accent-emerald bg-accent-emerald/10' : 'text-accent-rose bg-accent-rose/10'}`}>
                      {r.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-mono text-tx-secondary">{r.holdout_auc?.toFixed(3) ?? '—'}</td>
                  <td className="px-3 py-2"><TierBadge tier={r.readiness_tier} /></td>
                  <td className="px-3 py-2 font-mono text-tx-muted">{r.duration_seconds ? `${Math.round(r.duration_seconds)}s` : '—'}</td>
                  <td className="px-3 py-2 font-mono text-tx-muted whitespace-nowrap">
                    {r.finished_at ? new Date(r.finished_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Logs drawer */}
      {logsJob && logs.length > 0 && (
        <div className="glass-card rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Logs — PID {logsJob}</span>
            <button onClick={() => { setLogsJob(null); setLogs([]); }} className="text-tx-muted hover:text-tx-secondary text-xs">✕ Close</button>
          </div>
          <pre className="text-[11px] font-mono text-tx-secondary bg-surface-2 rounded-lg p-4 overflow-x-auto max-h-72 overflow-y-auto whitespace-pre-wrap">
            {logs.join('\n')}
          </pre>
        </div>
      )}
    </div>
  );
}
