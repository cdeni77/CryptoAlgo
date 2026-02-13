import { PaperEquityPoint, PaperFill } from '../types';

export default function PaperPerformancePanel({ equity, fills, loading }: { equity: PaperEquityPoint[]; fills: PaperFill[]; loading: boolean }) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Loading performance...</div>;
  const latest = equity[0];
  const first = equity[equity.length - 1];
  const returnPct = latest && first ? ((latest.equity - first.equity) / first.equity) * 100 : 0;
  const turnover = fills.reduce((acc, f) => acc + f.notional, 0);
  return (
    <div className="glass-card rounded-xl p-5 grid grid-cols-1 sm:grid-cols-3 gap-4">
      <div>
        <p className="text-xs text-[var(--text-muted)]">Paper Return</p>
        <p className={`text-xl font-semibold ${returnPct >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>{returnPct.toFixed(2)}%</p>
      </div>
      <div>
        <p className="text-xs text-[var(--text-muted)]">Latest Equity</p>
        <p className="text-xl font-semibold">${latest?.equity?.toFixed(2) ?? '—'}</p>
      </div>
      <div>
        <p className="text-xs text-[var(--text-muted)]">Filled Notional</p>
        <p className="text-xl font-semibold">${turnover.toFixed(2)}</p>
      </div>
    </div>
  );
}
