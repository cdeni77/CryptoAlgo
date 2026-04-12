import { PaperFill } from '../types';

function fmt(v: number) {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

interface Props { fills: PaperFill[]; limit?: number }

export default function PaperFillsTable({ fills, limit = 20 }: Props) {
  const rows = fills.slice(0, limit);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[rgba(56,189,248,0.08)]">
            {['Time', 'Coin', 'Side', 'Qty', 'Price', 'Notional', 'Fee', 'Slip'].map(h => (
              <th key={h} className="text-left px-2 py-2 text-tx-muted font-medium tracking-wider uppercase text-[10px]">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr><td colSpan={8} className="px-2 py-6 text-center text-tx-muted">No fills yet</td></tr>
          )}
          {rows.map(f => (
            <tr key={f.id} className="border-b border-[rgba(56,189,248,0.04)] hover:bg-[rgba(56,189,248,0.03)]">
              <td className="px-2 py-1.5 font-mono text-tx-muted whitespace-nowrap">
                {new Date(f.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}
              </td>
              <td className="px-2 py-1.5 font-medium text-tx-secondary">{f.coin}</td>
              <td className="px-2 py-1.5">
                <span className={`font-mono text-[11px] font-semibold ${f.side === 'long' ? 'text-accent-emerald' : 'text-accent-rose'}`}>
                  {f.side.toUpperCase()}
                </span>
              </td>
              <td className="px-2 py-1.5 font-mono text-tx-secondary">{f.contracts}</td>
              <td className="px-2 py-1.5 font-mono text-tx-secondary">${fmt(f.fill_price)}</td>
              <td className="px-2 py-1.5 font-mono text-tx-secondary">${fmt(f.notional)}</td>
              <td className="px-2 py-1.5 font-mono text-accent-rose">${fmt(f.fee)}</td>
              <td className="px-2 py-1.5 font-mono text-tx-muted">{f.slippage_bps?.toFixed(1)}bps</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
