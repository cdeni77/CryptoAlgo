import { useCallback, useEffect, useMemo, useState } from 'react';
import { getPaperEquity, getPaperFills, getPaperPositions } from '../api/paperApi';
import { getResearchCoin, getResearchFeatures, getResearchRuns, getResearchSummary } from '../api/researchApi';
import PaperEquityTable from '../components/PaperEquityTable';
import PaperFillsTable from '../components/PaperFillsTable';
import PaperPerformancePanel from '../components/PaperPerformancePanel';
import PaperPositionsTable from '../components/PaperPositionsTable';
import { PaperEquityPoint, PaperFill, PaperPosition, ResearchCoinHealth, ResearchFeatures, ResearchRun, ResearchSummary } from '../types';

type PaperTab = 'positions' | 'equity' | 'performance' | 'fills';

const formatAgo = (d: Date | null) => {
  if (!d) return '—';
  const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
};

export default function StrategyLabPage() {
  const [summary, setSummary] = useState<ResearchSummary | null>(null);
  const [runs, setRuns] = useState<ResearchRun[]>([]);
  const [selectedCoin, setSelectedCoin] = useState('BTC');
  const [features, setFeatures] = useState<ResearchFeatures | null>(null);
  const [coinDetail, setCoinDetail] = useState<ResearchCoinHealth | null>(null);
  const [paperPositions, setPaperPositions] = useState<PaperPosition[]>([]);
  const [paperEquity, setPaperEquity] = useState<PaperEquityPoint[]>([]);
  const [paperFills, setPaperFills] = useState<PaperFill[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingPaper, setLoadingPaper] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [paperTab, setPaperTab] = useState<PaperTab>('positions');
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [paperUpdatedAt, setPaperUpdatedAt] = useState<Date | null>(null);

  const loadResearch = useCallback(async () => {
    setLoading(true);
    try {
      const [summaryRes, runsRes] = await Promise.all([getResearchSummary(), getResearchRuns(30)]);
      setSummary(summaryRes);
      setRuns(runsRes);
      const preferredCoin = summaryRes.coins[0]?.coin ?? selectedCoin;
      setSelectedCoin(preferredCoin);
      const [featureRes, coinRes] = await Promise.all([getResearchFeatures(preferredCoin), getResearchCoin(preferredCoin)]);
      setFeatures(featureRes);
      setCoinDetail(coinRes.coin);
      setUpdatedAt(new Date());
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [selectedCoin]);

  const loadCoinSpecific = useCallback(async (coin: string) => {
    try {
      const [featureRes, coinRes] = await Promise.all([getResearchFeatures(coin), getResearchCoin(coin)]);
      setFeatures(featureRes);
      setCoinDetail(coinRes.coin);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const loadPaper = useCallback(async () => {
    setLoadingPaper(true);
    try {
      const [positions, equity, fills] = await Promise.all([
        getPaperPositions(),
        getPaperEquity(250),
        getPaperFills(250),
      ]);
      setPaperPositions(positions);
      setPaperEquity(equity);
      setPaperFills(fills);
      setPaperUpdatedAt(new Date());
    } finally {
      setLoadingPaper(false);
    }
  }, []);

  useEffect(() => {
    loadResearch();
    loadPaper();
    const researchIv = setInterval(loadResearch, 45000);
    const paperIv = setInterval(loadPaper, 20000);
    return () => {
      clearInterval(researchIv);
      clearInterval(paperIv);
    };
  }, [loadPaper, loadResearch]);

  const staleData = useMemo(() => {
    if (!updatedAt || !paperUpdatedAt) return false;
    return Date.now() - updatedAt.getTime() > 120000 || Date.now() - paperUpdatedAt.getTime() > 40000;
  }, [updatedAt, paperUpdatedAt]);

  const kpiCards = useMemo(() => {
    const k = summary?.kpis;
    if (!k) return [];
    return [
      { label: 'Holdout AUC', value: k.holdout_auc !== null ? k.holdout_auc.toFixed(3) : '—' },
      { label: 'PR-AUC', value: k.pr_auc !== null ? k.pr_auc.toFixed(3) : '—' },
      { label: 'Precision@Threshold', value: k.precision_at_threshold !== null ? `${(k.precision_at_threshold * 100).toFixed(1)}%` : '—' },
      { label: 'Win Rate (Realized)', value: `${k.win_rate_realized.toFixed(1)}%` },
      { label: 'Acted Signal Rate', value: `${k.acted_signal_rate.toFixed(1)}%` },
      { label: 'Drift Delta', value: `${k.drift_delta.toFixed(1)}%` },
    ];
  }, [summary]);

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[var(--border-subtle)] bg-[var(--bg-primary)]/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-5 py-4 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">Strategy Lab</h1>
            <p className="text-xs text-[var(--text-muted)]">Model lifecycle telemetry + paper trading workflow.</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono-trade text-[var(--text-muted)]">Research: {formatAgo(updatedAt)} · Paper: {formatAgo(paperUpdatedAt)}</span>
            <button onClick={() => { loadResearch(); loadPaper(); }} className="px-3 py-2 rounded-lg text-xs border border-[var(--border-accent)] text-[var(--accent-cyan)]">Refresh All</button>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-5 py-6 space-y-6">
        {staleData && (
          <div className="glass-card rounded-xl p-3 text-sm text-[var(--accent-amber)]">Some data may be stale (last update: {updatedAt?.toLocaleTimeString()}).</div>
        )}
        {error && <div className="glass-card rounded-xl p-3 text-sm text-[var(--accent-rose)]">Could not load strategy data. Retrying automatically. {error}</div>}

        <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {loading && kpiCards.length === 0 ? (
            <div className="glass-card rounded-xl p-5 text-sm text-[var(--text-muted)] col-span-full">Loading strategy telemetry…</div>
          ) : kpiCards.map((card) => (
            <div key={card.label} className="glass-card rounded-xl p-3">
              <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{card.label}</p>
              <p className="text-xl font-semibold mt-1 text-[var(--text-primary)]">{card.value}</p>
            </div>
          ))}
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="glass-card rounded-xl p-4 lg:col-span-8 overflow-x-auto">
            <h2 className="text-sm font-semibold mb-3">Coin Strategy Scoreboard</h2>
            {!summary || summary.coins.length === 0 ? (
              <p className="text-sm text-[var(--text-muted)]">No research runs found yet. Run training/optimization to populate this view.</p>
            ) : (
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-[var(--border-subtle)] text-[10px] uppercase text-[var(--text-muted)]">
                    {['Coin', 'Health', 'AUC', 'Win Rate', 'Acted', 'Drift', 'Freshness'].map((h) => <th key={h} className="text-left py-2">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {summary.coins.map((c) => (
                    <tr key={c.coin} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-elevated)]/40 cursor-pointer" onClick={() => { setSelectedCoin(c.coin); loadCoinSpecific(c.coin); }}>
                      <td className="py-2 pr-4 font-semibold">{c.coin}</td>
                      <td className="py-2 pr-4">
                        <span className={`inline-flex px-2 py-0.5 rounded text-[10px] uppercase ${c.health === 'healthy' ? 'bg-emerald-500/15 text-[var(--accent-emerald)]' : c.health === 'watch' ? 'bg-amber-500/15 text-[var(--accent-amber)]' : 'bg-rose-500/15 text-[var(--accent-rose)]'}`}>{c.health.replace('_', ' ')}</span>
                      </td>
                      <td className="py-2 pr-4">{c.holdout_auc !== null ? c.holdout_auc.toFixed(3) : '—'}</td>
                      <td className="py-2 pr-4">{c.win_rate_realized.toFixed(1)}%</td>
                      <td className="py-2 pr-4">{c.acted_signal_rate.toFixed(1)}%</td>
                      <td className="py-2 pr-4">{c.drift_delta.toFixed(1)}%</td>
                      <td className="py-2">{c.optimization_freshness_hours !== null ? `${(c.optimization_freshness_hours / 24).toFixed(1)}d` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="glass-card rounded-xl p-4 lg:col-span-4">
            <h2 className="text-sm font-semibold mb-3">Explainability Lite — {selectedCoin}</h2>
            {!features ? (
              <p className="text-sm text-[var(--text-muted)]">Loading strategy telemetry…</p>
            ) : (
              <div className="space-y-4">
                <div>
                  <p className="text-xs uppercase text-[var(--text-muted)] mb-2">Feature Importance</p>
                  <div className="space-y-2">
                    {features.feature_importance.map((item) => (
                      <div key={item.feature}>
                        <div className="flex justify-between text-xs"><span>{item.feature}</span><span>{(item.importance * 100).toFixed(1)}%</span></div>
                        <div className="h-1.5 rounded bg-[var(--bg-secondary)]"><div className="h-1.5 rounded bg-[var(--accent-cyan)]" style={{ width: `${item.importance * 100}%` }} /></div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-xs uppercase text-[var(--text-muted)] mb-2">Signal Distribution</p>
                  <div className="grid grid-cols-2 gap-2">
                    {features.signal_distribution.map((s) => (
                      <div key={s.label} className="bg-[var(--bg-secondary)]/70 rounded p-2">
                        <p className="text-[11px] text-[var(--text-muted)]">{s.label}</p>
                        <p className="text-sm font-semibold">{s.value}</p>
                      </div>
                    ))}
                  </div>
                </div>
                {coinDetail && <p className="text-xs text-[var(--text-muted)]">Robustness gate: <span className={coinDetail.robustness_gate ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}>{coinDetail.robustness_gate ? 'pass' : 'fail'}</span></p>}
              </div>
            )}
          </div>
        </section>

        <section className="glass-card rounded-xl p-4">
          <h2 className="text-sm font-semibold mb-3">Experiment Timeline</h2>
          {runs.length === 0 ? (
            <p className="text-sm text-[var(--text-muted)]">No research runs found yet. Run training/optimization to populate this view.</p>
          ) : (
            <div className="space-y-2">
              {runs.map((run) => (
                <div key={run.id} className="flex flex-wrap items-center justify-between text-xs border-b border-[var(--border-subtle)] pb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold w-12">{run.coin}</span>
                    <span className="uppercase text-[var(--accent-cyan)]">{run.run_type}</span>
                    <span className="text-[var(--text-muted)]">{new Date(run.finished_at).toLocaleString()}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span>AUC {run.holdout_auc !== null ? run.holdout_auc.toFixed(3) : '—'}</span>
                    <span>Duration {(run.duration_seconds / 60).toFixed(0)}m</span>
                    <span className={run.robustness_gate ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}>{run.robustness_gate ? 'Gate Pass' : 'Gate Fail'}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section>
          <div className="flex flex-wrap gap-2 mb-4 overflow-x-auto">
            {[
              ['positions', `Positions (${paperPositions.length})`],
              ['equity', `Equity (${paperEquity.length})`],
              ['performance', 'Performance'],
              ['fills', `Fills (${paperFills.length})`],
            ].map(([key, label]) => (
              <button key={key} onClick={() => setPaperTab(key as PaperTab)} className={`px-4 py-2 rounded-lg text-sm ${paperTab === key ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]' : 'text-[var(--text-muted)]'}`}>
                {label}
              </button>
            ))}
          </div>

          {paperTab === 'positions' && <PaperPositionsTable positions={paperPositions} loading={loadingPaper} />}
          {paperTab === 'equity' && <PaperEquityTable points={paperEquity} loading={loadingPaper} />}
          {paperTab === 'performance' && <PaperPerformancePanel equity={paperEquity} fills={paperFills} loading={loadingPaper} />}
          {paperTab === 'fills' && <PaperFillsTable fills={paperFills} loading={loadingPaper} />}
        </section>
      </main>
    </>
  );
}
