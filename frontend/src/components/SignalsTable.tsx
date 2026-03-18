import { useState } from 'react';
import { Signal } from '../types';

interface SignalsTableProps {
  signals: Signal[];
  loading: boolean;
}

function GateBadge({ pass: p, label }: { pass: boolean | null; label: string }) {
  if (p === null || p === undefined) return null;
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-mono-trade font-bold uppercase mr-1 mb-1 ${p ? 'bg-emerald-500/10 text-[var(--accent-emerald)] ring-1 ring-emerald-500/20' : 'bg-rose-500/10 text-[var(--accent-rose)] ring-1 ring-rose-500/20'}`}>
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
      <span className="font-mono-trade text-xs text-[var(--text-secondary)] w-10 text-right">{pct.toFixed(0)}%</span>
    </div>
  );
}

export default function SignalsTable({ signals, loading }: SignalsTableProps) {
  const [showAll, setShowAll] = useState(false);

  if (loading) {
    return (
      <div className="glass-card rounded-xl p-10 flex flex-col items-center justify-center h-64">
        <div className="w-10 h-10 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-sm text-[var(--text-muted)]">Loading signals...</p>
      </div>
    );
  }

  const filtered = showAll ? signals : signals.filter((s) => s.passed_gates);
  const rejectedCount = signals.filter((s) => !s.passed_gates).length;

  if (signals.length === 0) {
    return (
      <div className="glass-card rounded-xl p-10 text-center">
        <p className="text-[var(--text-muted)] text-sm">No signals yet — the trader writes hourly ML predictions here once running.</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-subtle)]">
        <span className="text-xs text-[var(--text-muted)]">
          {showAll ? `${signals.length} total` : `${filtered.length} actionable`}
          {rejectedCount > 0 && !showAll && <span className="ml-1 text-[var(--accent-amber)]">· {rejectedCount} rejected hidden</span>}
        </span>
        <button
          onClick={() => setShowAll((v) => !v)}
          className={`px-3 py-1 rounded text-xs border transition-colors ${showAll ? 'border-[var(--border-accent)] text-[var(--accent-cyan)]' : 'border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'}`}
        >
          {showAll ? 'Actionable only' : 'Show all'}
        </button>
      </div>
      <div className="overflow-x-auto max-h-[560px]">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10 bg-[var(--bg-card)]/95 backdrop-blur">
            <tr className="border-b border-[var(--border-subtle)]">
              {['Time', 'Coin', 'Direction', 'Confidence', 'AUC', 'Price', 'Gates', 'Contracts', 'Status'].map(h => (
                <th key={h} className="px-4 py-3 text-left text-[10px] font-mono-trade font-semibold text-[var(--text-muted)] uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {filtered.map((s) => (
              <tr key={s.id} className={`hover:bg-[var(--bg-elevated)]/50 transition-colors ${!s.passed_gates ? 'opacity-55' : ''}`}>
                <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">
                  {new Date(s.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                  {' '}
                  {new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </td>
                <td className="px-4 py-3 text-sm font-semibold text-[var(--text-primary)]">{s.coin}</td>
                <td className="px-4 py-3">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono-trade font-bold uppercase ${s.direction === 'long' ? 'bg-emerald-500/10 text-[var(--accent-emerald)] ring-1 ring-emerald-500/20' : s.direction === 'short' ? 'bg-rose-500/10 text-[var(--accent-rose)] ring-1 ring-rose-500/20' : 'bg-gray-500/10 text-[var(--text-muted)] ring-1 ring-gray-500/20'}`}>
                    {s.direction}
                  </span>
                </td>
                <td className="px-4 py-3"><ConfidenceBar value={s.confidence} /></td>
                <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">{s.model_auc !== null ? s.model_auc.toFixed(3) : '—'}</td>
                <td className="px-4 py-3 font-mono-trade text-xs text-[var(--text-secondary)]">
                  {s.price_at_signal !== null ? `$${s.price_at_signal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap">
                    <GateBadge pass={s.momentum_pass} label="MOM" />
                    <GateBadge pass={s.trend_pass} label="TRD" />
                    <GateBadge pass={s.regime_pass} label="REG" />
                    <GateBadge pass={s.ml_pass} label="ML" />
                  </div>
                </td>
                <td className="px-4 py-3 font-mono-trade text-sm text-[var(--text-primary)]">{s.contracts_suggested ?? '—'}</td>
                <td className="px-4 py-3">
                  {s.passed_gates ? (
                    s.acted_on ? (
                      <span className="inline-flex items-center gap-1 text-xs text-[var(--accent-emerald)]">
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-emerald)]" /> {s.trade_id != null ? `Filled #${s.trade_id}` : 'Filled'}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-xs text-[var(--accent-cyan)]">
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-cyan)]" /> Actionable
                      </span>
                    )
                  ) : (
                    <span className="text-xs text-[var(--text-muted)] font-mono-trade" title={s.gate_failure_reason ?? ''}>
                      {s.gate_failure_reason ?? 'rejected'}
                    </span>
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
