import { PaperFill } from '../types';

const sideClass = (side: string) =>
  side === 'long' ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]';

export default function PaperFillsTable({ fills, loading }: { fills: PaperFill[]; loading: boolean }) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Fetching paper trading state...</div>;

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto max-h-[560px]">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10 bg-[var(--bg-card)]/95">
            <tr className="border-b border-[var(--border-subtle)]">
              {['Time', 'Coin', 'Side', 'Contracts', 'Fill Price', 'Notional', 'Fee', 'Slippage (bps)'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-[10px] uppercase text-[var(--text-muted)]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {fills.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-[var(--text-muted)]">No paper fills yet.</td></tr>
            ) : fills.map((f) => (
              <tr key={f.id} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-elevated)]/30">
                <td className="px-4 py-2 text-xs text-[var(--text-muted)]">{new Date(f.created_at).toLocaleString()}</td>
                <td className="px-4 py-2 font-semibold">{f.coin}</td>
                <td className={`px-4 py-2 font-semibold uppercase text-xs ${sideClass(f.side)}`}>{f.side}</td>
                <td className="px-4 py-2">{f.contracts}</td>
                <td className="px-4 py-2">${f.fill_price.toFixed(2)}</td>
                <td className="px-4 py-2">${f.notional.toFixed(2)}</td>
                <td className="px-4 py-2">${f.fee.toFixed(4)}</td>
                <td className="px-4 py-2">{f.slippage_bps.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
