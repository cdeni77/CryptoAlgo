import { PaperPosition, PriceData, CDESpecs, CoinSymbol } from '../types';

const sideClass = (side: string) =>
  side === 'long' ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]';

function calcLive(
  p: PaperPosition,
  prices: PriceData | null,
  cdeSpecs: CDESpecs | null,
): { pnl: number; markPrice: number; isLive: boolean } {
  const coin = p.coin as CoinSymbol;
  const livePrice = prices?.[coin]?.price ?? null;
  const spec = cdeSpecs?.[coin] ?? null;
  if (livePrice !== null && spec !== null) {
    const sign = p.side === 'long' ? 1 : -1;
    const pnl = p.contracts * spec.units_per_contract * (livePrice - p.entry_price) * sign;
    return { pnl, markPrice: livePrice, isLive: true };
  }
  return { pnl: p.unrealized_pnl, markPrice: p.mark_price, isLive: false };
}

interface Props {
  positions: PaperPosition[];
  loading: boolean;
  prices: PriceData | null;
  cdeSpecs: CDESpecs | null;
}

export default function PaperPositionsTable({ positions, loading, prices, cdeSpecs }: Props) {
  if (loading) return <div className="glass-card rounded-xl p-6 text-sm text-[var(--text-muted)]">Loading paper positions...</div>;
  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="overflow-x-auto max-h-[560px]">
        <table className="min-w-full">
          <thead className="sticky top-0 z-10 bg-[var(--bg-card)]/95">
            <tr className="border-b border-[var(--border-subtle)]">
              {['Coin', 'Side', 'Contracts', 'Entry', 'Mark', 'Notional', 'Unrealized PNL', 'Fees', 'Opened'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-[10px] uppercase text-[var(--text-muted)]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 ? (
              <tr><td colSpan={9} className="px-4 py-8 text-center text-[var(--text-muted)]">No open paper positions.</td></tr>
            ) : positions.map((p) => {
              const { pnl, markPrice, isLive } = calcLive(p, prices, cdeSpecs);
              return (
                <tr key={p.id} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-elevated)]/30">
                  <td className="px-4 py-2 font-semibold">{p.coin}</td>
                  <td className={`px-4 py-2 font-semibold uppercase text-xs ${sideClass(p.side)}`}>{p.side}</td>
                  <td className="px-4 py-2">{p.contracts}</td>
                  <td className="px-4 py-2">${p.entry_price.toFixed(4)}</td>
                  <td className="px-4 py-2">
                    ${markPrice.toFixed(4)}
                    {isLive && <span className="ml-1 text-[9px] text-[var(--accent-emerald)] opacity-60">live</span>}
                  </td>
                  <td className="px-4 py-2">${p.notional.toFixed(2)}</td>
                  <td className={`px-4 py-2 font-semibold ${pnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>
                    {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                  </td>
                  <td className="px-4 py-2">${p.fees_paid.toFixed(2)}</td>
                  <td className="px-4 py-2 text-xs text-[var(--text-muted)]">{new Date(p.opened_at).toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
