interface Props {
  coin: string;
  price: number | null;
  change24h: number | null;
}

const COIN_COLORS: Record<string, string> = {
  BTC: '#fbbf24', ETH: '#a78bfa', SOL: '#34d399',
  XRP: '#38bdf8', DOGE: '#f59e0b', AVAX: '#fb7185',
  ADA: '#60a5fa', LINK: '#3b82f6', LTC: '#94a3b8',
};

export default function PriceCard({ coin, price, change24h }: Props) {
  const up = (change24h ?? 0) >= 0;
  const color = COIN_COLORS[coin] ?? '#94a3b8';

  return (
    <div className="glass-card glass-card-hover rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold"
            style={{ background: `${color}22`, color }}
          >
            {coin[0]}
          </div>
          <span className="text-tx-secondary text-xs font-medium tracking-wide">{coin}</span>
        </div>
        {change24h !== null && (
          <span className={`text-[11px] font-mono font-medium px-1.5 py-0.5 rounded ${
            up ? 'text-accent-emerald bg-accent-emerald/10' : 'text-accent-rose bg-accent-rose/10'
          }`}>
            {up ? '+' : ''}{change24h.toFixed(2)}%
          </span>
        )}
      </div>
      <div className="font-mono text-tx-primary font-semibold text-base">
        {price !== null
          ? `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: price > 100 ? 2 : 4 })}`
          : <span className="text-tx-muted text-sm">—</span>}
      </div>
    </div>
  );
}
