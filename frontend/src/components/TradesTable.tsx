import { Trade } from '../types';

interface TradesTableProps {
  trades: Trade[];
  loading?: boolean;
}

export default function TradesTable({ trades, loading = false }: TradesTableProps) {
  if (loading) {
    return (
      <div className="glass-card rounded-xl p-10 flex flex-col items-center justify-center h-80">
        <div className="w-10 h-10 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-sm text-[var(--text-muted)]">Loading trades...</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="border-b border-[var(--border-subtle)]">
              {['ID', 'Coin', 'Opened', 'Side', 'Contracts', 'Entry', 'Exit', 'PNL', 'Status'].map(h => (
                <th key={h} className="px-4 py-3 text-left text-[10px] font-mono-trade font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-4 py-12 text-center text-[var(--text-muted)] text-sm">
                  No trades recorded yet
                </td>
              </tr>
            ) : (
              trades.map((trade) => (
                <tr key={trade.id} className="hover:bg-[var(--bg-elevated)]/50 transition-colors">
                  <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-muted)]">
                    #{trade.id}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-sm font-semibold text-[var(--text-primary)]">{trade.coin}</span>
                  </td>
                  <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">
                    {new Date(trade.datetime_open).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                    {' '}
                    {new Date(trade.datetime_open).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`
                      inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono-trade font-bold uppercase
                      ${trade.side === 'long'
                        ? 'bg-emerald-500/10 text-[var(--accent-emerald)] ring-1 ring-emerald-500/20'
                        : 'bg-rose-500/10 text-[var(--accent-rose)] ring-1 ring-rose-500/20'
                      }
                    `}>
                      {trade.side}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono-trade text-sm text-[var(--text-primary)]">
                    {trade.contracts}
                  </td>
                  <td className="px-4 py-3 font-mono-trade text-sm text-[var(--text-secondary)]">
                    ${Number(trade.entry_price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-4 py-3 font-mono-trade text-sm text-[var(--text-secondary)]">
                    {trade.exit_price !== null
                      ? `$${Number(trade.exit_price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                      : '—'}
                  </td>
                  <td className="px-4 py-3">
                    {trade.net_pnl !== null ? (
                      <span className={`font-mono-trade text-sm font-semibold ${trade.net_pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>
                        {trade.net_pnl >= 0 ? '+' : ''}${trade.net_pnl.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-[var(--text-muted)] text-sm">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`
                      inline-flex items-center gap-1.5 text-xs font-medium
                      ${trade.status === 'open' ? 'text-[var(--accent-cyan)]' : 'text-[var(--text-muted)]'}
                    `}>
                      {trade.status === 'open' && (
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-cyan)] animate-pulse" />
                      )}
                      {trade.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}