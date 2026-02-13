import { PaperPosition } from '../types';

export default function PaperPositionsTable({ positions, loading }: { positions: PaperPosition[]; loading: boolean }) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Loading paper positions...</div>;
  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto max-h-[560px]">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10 bg-[var(--bg-card)]/95">
            <tr className="border-b border-[var(--border-subtle)]">
              {['Coin', 'Side', 'Contracts', 'Entry', 'Mark', 'Notional', 'Unrealized', 'Fees'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-[10px] uppercase text-[var(--text-muted)]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-[var(--text-muted)]">No open paper positions.</td></tr>
            ) : positions.map((p) => (
              <tr key={p.id} className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2">{p.coin}</td>
                <td className="px-4 py-2">{p.side}</td>
                <td className="px-4 py-2">{p.contracts}</td>
                <td className="px-4 py-2">${p.entry_price.toFixed(2)}</td>
                <td className="px-4 py-2">${p.mark_price.toFixed(2)}</td>
                <td className="px-4 py-2">${p.notional.toFixed(2)}</td>
                <td className={`px-4 py-2 ${p.unrealized_pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>{p.unrealized_pnl.toFixed(2)}</td>
                <td className="px-4 py-2">${p.fees_paid.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
