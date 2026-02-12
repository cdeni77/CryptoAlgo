import { Signal } from '../types';

interface SignalsTableProps {
  signals: Signal[];
  loading: boolean;
}

function GateBadge({ pass: p, label }: { pass: boolean | null; label: string }) {
  if (p === null || p === undefined) return null;
  return (
    <span
      className={`
        inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-mono-trade font-bold uppercase mr-1 mb-1
        ${p
          ? 'bg-emerald-500/10 text-[var(--accent-emerald)] ring-1 ring-emerald-500/20'
          : 'bg-rose-500/10 text-[var(--accent-rose)] ring-1 ring-rose-500/20'
        }
      `}
    >
      {label}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const color =
    pct >= 80 ? 'bg-[var(--accent-emerald)]' :
    pct >= 60 ? 'bg-[var(--accent-cyan)]' :
    pct >= 40 ? 'bg-[var(--accent-amber)]' :
    'bg-[var(--accent-rose)]';
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 h-1.5 rounded-full bg-[var(--bg-secondary)] overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono-trade text-xs text-[var(--text-secondary)] w-10 text-right">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export default function SignalsTable({ signals, loading }: SignalsTableProps) {
  if (loading) {
    return (
      <div className="glass-card rounded-xl p-10 flex flex-col items-center justify-center h-64">
        <div className="w-10 h-10 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-sm text-[var(--text-muted)]">Loading signals...</p>
      </div>
    );
  }

  if (signals.length === 0) {
    return (
      <div className="glass-card rounded-xl p-10 text-center">
        <p className="text-[var(--text-muted)] text-sm">
          No signals yet — the trader writes hourly ML predictions here once running.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="border-b border-[var(--border-subtle)]">
              {['Time', 'Coin', 'Direction', 'Confidence', 'AUC', 'Price', 'Gates', 'Contracts', 'Acted'].map(h => (
                <th key={h} className="px-4 py-3 text-left text-[10px] font-mono-trade font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {signals.map((s) => (
              <tr key={s.id} className="hover:bg-[var(--bg-elevated)]/50 transition-colors">
                {/* Time */}
                <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">
                  {new Date(s.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                  {' '}
                  {new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </td>
                {/* Coin */}
                <td className="px-4 py-3 text-sm font-semibold text-[var(--text-primary)]">
                  {s.coin}
                </td>
                {/* Direction */}
                <td className="px-4 py-3">
                  <span className={`
                    inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono-trade font-bold uppercase
                    ${s.direction === 'long'
                      ? 'bg-emerald-500/10 text-[var(--accent-emerald)] ring-1 ring-emerald-500/20'
                      : s.direction === 'short'
                        ? 'bg-rose-500/10 text-[var(--accent-rose)] ring-1 ring-rose-500/20'
                        : 'bg-gray-500/10 text-[var(--text-muted)] ring-1 ring-gray-500/20'
                    }
                  `}>
                    {s.direction}
                  </span>
                </td>
                {/* Confidence */}
                <td className="px-4 py-3">
                  <ConfidenceBar value={s.confidence} />
                </td>
                {/* AUC */}
                <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">
                  {s.model_auc !== null ? s.model_auc.toFixed(3) : '—'}
                </td>
                {/* Price */}
                <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">
                  {s.price_at_signal !== null
                    ? `$${s.price_at_signal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                    : '—'}
                </td>
                {/* Gates */}
                <td className="px-4 py-3">
                  <div className="flex flex-wrap">
                    <GateBadge pass={s.momentum_pass} label="MOM" />
                    <GateBadge pass={s.trend_pass} label="TRD" />
                    <GateBadge pass={s.regime_pass} label="REG" />
                    <GateBadge pass={s.ml_pass} label="ML" />
                  </div>
                </td>
                {/* Contracts */}
                <td className="px-4 py-3 font-mono-trade text-sm text-[var(--text-primary)]">
                  {s.contracts_suggested ?? '—'}
                </td>
                {/* Acted */}
                <td className="px-4 py-3">
                  {s.acted_on ? (
                    <span className="inline-flex items-center gap-1 text-xs text-[var(--accent-emerald)]">
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-emerald)]" />
                      #{s.trade_id}
                    </span>
                  ) : (
                    <span className="text-xs text-[var(--text-muted)]">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}