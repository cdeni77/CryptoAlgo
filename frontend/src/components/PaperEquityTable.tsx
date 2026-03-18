import { PaperEquityPoint } from '../types';

export default function PaperEquityTable({ points, loading }: { points: PaperEquityPoint[]; loading: boolean }) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Loading equity curve...</div>;
  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto max-h-[560px]">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10 bg-[var(--bg-card)]/95">
            <tr className="border-b border-[var(--border-subtle)]">
              {['Time', 'Equity', 'Cash', 'Realized PNL', 'Unrealized PNL', 'Open Pos'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-[10px] uppercase text-[var(--text-muted)]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {points.length === 0 ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-[var(--text-muted)]">No equity snapshots yet. The paper engine writes a point each time a signal is acted on.</td></tr>
            ) : points.map((p) => (
              <tr key={p.id} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-elevated)]/30">
                <td className="px-4 py-2 text-xs text-[var(--text-muted)]">{new Date(p.timestamp).toLocaleString()}</td>
                <td className="px-4 py-2 font-semibold">${p.equity.toFixed(2)}</td>
                <td className="px-4 py-2">${p.cash_balance.toFixed(2)}</td>
                <td className={`px-4 py-2 ${p.realized_pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>
                  {p.realized_pnl >= 0 ? '+' : ''}{p.realized_pnl.toFixed(2)}
                </td>
                <td className={`px-4 py-2 ${p.unrealized_pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>
                  {p.unrealized_pnl >= 0 ? '+' : ''}{p.unrealized_pnl.toFixed(2)}
                </td>
                <td className="px-4 py-2">{p.open_positions}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
