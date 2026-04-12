import { Signal } from '../types';

function Gate({ pass }: { pass: boolean | null }) {
  if (pass === null) return <span className="text-tx-muted">—</span>;
  return <span className={pass ? 'text-accent-emerald' : 'text-accent-rose'}>{pass ? '✓' : '✗'}</span>;
}

interface Props { signals: Signal[]; limit?: number }

export default function SignalsTable({ signals, limit = 20 }: Props) {
  const rows = signals.slice(0, limit);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[rgba(56,189,248,0.08)]">
            {['Time', 'Coin', 'Dir', 'Conf', 'Mom', 'Trend', 'ML', 'AUC', 'Price', 'Acted'].map(h => (
              <th key={h} className="text-left px-2 py-2 text-tx-muted font-medium tracking-wider uppercase text-[10px]">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr><td colSpan={10} className="px-2 py-6 text-center text-tx-muted">No signals yet</td></tr>
          )}
          {rows.map(s => (
            <tr key={s.id} className="border-b border-[rgba(56,189,248,0.04)] hover:bg-[rgba(56,189,248,0.03)] transition-colors">
              <td className="px-2 py-1.5 font-mono text-tx-muted whitespace-nowrap">
                {new Date(s.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}
              </td>
              <td className="px-2 py-1.5 font-medium text-tx-secondary">{s.coin}</td>
              <td className="px-2 py-1.5">
                <span className={`font-mono font-semibold ${
                  s.direction === 'long' ? 'text-accent-emerald' : s.direction === 'short' ? 'text-accent-rose' : 'text-tx-muted'
                }`}>
                  {s.direction === 'long' ? '▲' : s.direction === 'short' ? '▼' : '—'}
                </span>
              </td>
              <td className="px-2 py-1.5 font-mono text-tx-secondary">{(s.confidence * 100).toFixed(0)}%</td>
              <td className="px-2 py-1.5"><Gate pass={s.momentum_pass} /></td>
              <td className="px-2 py-1.5"><Gate pass={s.trend_pass} /></td>
              <td className="px-2 py-1.5"><Gate pass={s.ml_pass} /></td>
              <td className="px-2 py-1.5 font-mono text-tx-muted">{s.model_auc?.toFixed(3) ?? '—'}</td>
              <td className="px-2 py-1.5 font-mono text-tx-secondary">
                {s.price_at_signal != null ? `$${s.price_at_signal.toLocaleString('en-US', { maximumFractionDigits: 2 })}` : '—'}
              </td>
              <td className="px-2 py-1.5">
                {s.acted_on
                  ? <span className="text-accent-cyan text-[10px]">✓</span>
                  : <span className="text-tx-muted text-[10px]">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
