import { PaperEquityPoint } from '../types';

export default function PaperEquityTable({ points, loading }: { points: PaperEquityPoint[]; loading: boolean }) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Loading equity curve...</div>;
  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto max-h-[560px]">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10 bg-[var(--bg-card)]/95">
            <tr className="border-b border-[var(--border-subtle)]">
              {['Time', 'Equity', 'Cash', 'Realized', 'Unrealized', 'Open Pos'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-[10px] uppercase text-[var(--text-muted)]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.id} className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2">{new Date(p.timestamp).toLocaleString()}</td>
                <td className="px-4 py-2">${p.equity.toFixed(2)}</td>
                <td className="px-4 py-2">${p.cash_balance.toFixed(2)}</td>
                <td className="px-4 py-2">${p.realized_pnl.toFixed(2)}</td>
                <td className="px-4 py-2">${p.unrealized_pnl.toFixed(2)}</td>
                <td className="px-4 py-2">{p.open_positions}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
