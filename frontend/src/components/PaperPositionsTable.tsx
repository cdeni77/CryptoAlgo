import { PaperPosition, PriceData } from '../types';

// Must match trading_costs.py CONTRACT_SPECS
const UNITS: Record<string, number> = {
  BTC: 0.01, ETH: 0.10, SOL: 5, XRP: 500, DOGE: 5000,
  AVAX: 10, ADA: 1000, LINK: 50, LTC: 5,
};

function fmt(v: number) {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

interface Props {
  positions: PaperPosition[];
  prices?: PriceData | null;
}

export default function PaperPositionsTable({ positions, prices }: Props) {
  const open = positions.filter(p => p.is_open);

  if (!open.length) {
    return <div className="px-4 py-8 text-center text-tx-muted text-sm">No open positions</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[rgba(56,189,248,0.08)]">
            {['Coin', 'Side', 'Qty', 'Entry', 'Mark', 'Unrealized', 'Fees', 'Opened'].map(h => (
              <th key={h} className="text-left px-3 py-2 text-tx-muted font-medium tracking-wider uppercase text-[10px]">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {open.map(p => {
            const livePrice = prices?.[p.coin as keyof typeof prices]?.price ?? null;
            const units = UNITS[p.coin.toUpperCase()] ?? 1;
            const sign = p.side === 'long' ? 1 : -1;
            const liveUnrealized = livePrice != null
              ? p.contracts * units * (livePrice - p.entry_price) * sign
              : p.unrealized_pnl;
            const markDisplay = livePrice ?? p.mark_price;
            const isLive = livePrice != null;

            return (
              <tr key={p.id} className="border-b border-[rgba(56,189,248,0.04)] hover:bg-[rgba(56,189,248,0.03)]">
                <td className="px-3 py-2.5 font-medium text-tx-primary">{p.coin}</td>
                <td className="px-3 py-2.5">
                  <span className={`font-mono text-[11px] font-semibold px-1.5 py-0.5 rounded ${
                    p.side === 'long' ? 'text-accent-emerald bg-accent-emerald/10' : 'text-accent-rose bg-accent-rose/10'
                  }`}>
                    {p.side.toUpperCase()}
                  </span>
                </td>
                <td className="px-3 py-2.5 font-mono text-tx-secondary">{p.contracts}</td>
                <td className="px-3 py-2.5 font-mono text-tx-secondary">${fmt(p.entry_price)}</td>
                <td className="px-3 py-2.5 font-mono text-tx-secondary">
                  ${fmt(markDisplay)}
                  {isLive && <span className="ml-1 text-[9px] text-accent-cyan opacity-60">live</span>}
                </td>
                <td className={`px-3 py-2.5 font-mono font-semibold ${liveUnrealized >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}`}>
                  {liveUnrealized >= 0 ? '+' : ''}${fmt(liveUnrealized)}
                </td>
                <td className="px-3 py-2.5 font-mono text-tx-muted">${fmt(p.fees_paid)}</td>
                <td className="px-3 py-2.5 font-mono text-tx-muted whitespace-nowrap">
                  {new Date(p.opened_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
