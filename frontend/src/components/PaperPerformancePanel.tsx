import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { PaperEquityPoint, PaperFill } from '../types';

const fmt = (v: number) => `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

export default function PaperPerformancePanel({
  equity,
  fills,
  loading,
}: {
  equity: PaperEquityPoint[];
  fills: PaperFill[];
  loading: boolean;
}) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Loading performance...</div>;

  // equity is newest-first from the API
  const latest = equity[0];
  const first = equity[equity.length - 1];
  const returnPct = latest && first && first.equity > 0 ? ((latest.equity - first.equity) / first.equity) * 100 : 0;

  const totalFees = fills.reduce((acc, f) => acc + f.fee, 0);
  const totalNotional = fills.reduce((acc, f) => acc + f.notional, 0);
  const tradeCount = fills.length;
  const avgFee = tradeCount > 0 ? totalFees / tradeCount : 0;

  const maxDrawdown = useMemo(() => {
    if (equity.length < 2) return 0;
    // equity is newest-first; reverse for chronological order
    const chronological = [...equity].reverse();
    let peak = chronological[0].equity;
    let maxDd = 0;
    for (const p of chronological) {
      if (p.equity > peak) peak = p.equity;
      const dd = peak > 0 ? ((peak - p.equity) / peak) * 100 : 0;
      if (dd > maxDd) maxDd = dd;
    }
    return maxDd;
  }, [equity]);

  // chart series in chronological order
  const chartSeries = useMemo(() => [...equity].reverse(), [equity]);

  const chartBounds = useMemo(() => {
    if (!chartSeries.length) return { min: 0, max: 0, pad: 1 };
    const values = chartSeries.map((p) => p.equity);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(1, max - min);
    return { min, max, pad: span * 0.1 };
  }, [chartSeries]);

  const initialEquity = first?.equity ?? 0;

  const stats = [
    { label: 'Paper Return', value: pct(returnPct), color: returnPct >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]' },
    { label: 'Latest Equity', value: latest ? fmt(latest.equity) : '—', color: 'text-[var(--text-primary)]' },
    { label: 'Realized PNL', value: latest ? `${latest.realized_pnl >= 0 ? '+' : ''}${fmt(latest.realized_pnl)}` : '—', color: latest && latest.realized_pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]' },
    { label: 'Unrealized PNL', value: latest ? `${latest.unrealized_pnl >= 0 ? '+' : ''}${fmt(latest.unrealized_pnl)}` : '—', color: latest && latest.unrealized_pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]' },
    { label: 'Filled Notional', value: fmt(totalNotional), color: 'text-[var(--text-primary)]' },
    { label: 'Total Fees', value: fmt(totalFees), color: 'text-[var(--accent-amber)]' },
    { label: 'Trades', value: String(tradeCount), color: 'text-[var(--text-primary)]' },
    { label: 'Avg Fee / Trade', value: tradeCount > 0 ? fmt(avgFee) : '—', color: 'text-[var(--text-muted)]' },
    { label: 'Max Drawdown', value: `-${maxDrawdown.toFixed(2)}%`, color: maxDrawdown > 5 ? 'text-[var(--accent-rose)]' : 'text-[var(--text-muted)]' },
  ];

  return (
    <div className="glass-card rounded-xl p-5 space-y-5">
      <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-9 gap-3">
        {stats.map(({ label, value, color }) => (
          <div key={label} className="bg-[var(--bg-secondary)]/60 rounded-lg p-3">
            <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade mb-1">{label}</p>
            <p className={`text-base font-semibold ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      <div className="h-52">
        <p className="text-xs uppercase text-[var(--text-muted)] mb-2">Equity Curve</p>
        {chartSeries.length >= 2 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartSeries} margin={{ top: 4, right: 10, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                tickFormatter={(v: string) => new Date(v).toLocaleDateString([], { month: 'short', day: 'numeric' })}
              />
              <YAxis
                domain={[chartBounds.min - chartBounds.pad, chartBounds.max + chartBounds.pad]}
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
              />
              <Tooltip
                formatter={(value: number) => fmt(value)}
                labelFormatter={(label: string) => new Date(label).toLocaleString()}
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(148,163,184,0.35)', borderRadius: 8 }}
              />
              {initialEquity > 0 && (
                <ReferenceLine y={initialEquity} stroke="rgba(148,163,184,0.4)" strokeDasharray="4 4" />
              )}
              <Line type="monotone" dataKey="equity" stroke="#34d399" strokeWidth={2} dot={false} name="Equity" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-[var(--text-muted)]">Not enough data points to render chart yet.</p>
        )}
      </div>
    </div>
  );
}
