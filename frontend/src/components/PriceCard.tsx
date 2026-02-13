import { useEffect, useRef, useState } from 'react';
import { CDESpec, DataSource } from '../types';
import DataSourceToggle from './DataSourceToggle';

interface PriceCardProps {
  coin: string;
  icon: string;
  price: number | null;
  change24h: number | null;
  loading: boolean;
  error?: string;
  cdeSpec?: CDESpec;
  dataSource: DataSource;
  onDataSourceChange: (source: DataSource) => void;
  selected?: boolean;
  onClick?: () => void;
}

export default function PriceCard({
  coin, icon, price, change24h, loading, error,
  cdeSpec, dataSource, onDataSourceChange,
  selected = false, onClick,
}: PriceCardProps) {
  const prevPriceRef = useRef<number | null>(null);
  const [direction, setDirection] = useState<'up' | 'down' | null>(null);

  useEffect(() => {
    if (loading || error || price === null) {
      prevPriceRef.current = null;
      setDirection(null);
      return;
    }
    const currentPrice = Number(price);
    if (isNaN(currentPrice)) { prevPriceRef.current = null; setDirection(null); return; }

    const prev = prevPriceRef.current;
    if (prev !== null) {
      const diff = currentPrice - prev;
      if (Math.abs(diff) > 0.01) {
        setDirection(diff > 0 ? 'up' : 'down');
        const timer = setTimeout(() => setDirection(null), 1200);
        return () => clearTimeout(timer);
      } else {
        setDirection(null);
      }
    }
    prevPriceRef.current = currentPrice;
  }, [price, loading, error, coin]);

  const priceTextClass =
    direction === 'up' ? 'animate-price-up' :
    direction === 'down' ? 'animate-price-down' : '';

  const formattedPrice = price !== null
    ? `$${Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    : '—';

  const changeClass = change24h === null ? 'text-[var(--text-muted)]'
    : change24h >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]';
  const changeText = change24h === null ? '—'
    : `${change24h >= 0 ? '+' : ''}${change24h.toFixed(2)}%`;

  // CDE computed price (contract value based on current spot)
  const cdeContractValue = (price !== null && cdeSpec)
    ? price * cdeSpec.units_per_contract
    : null;

  return (
    <div
      onClick={onClick}
      className={`
        glass-card glass-card-hover rounded-xl p-5 cursor-pointer relative overflow-hidden
        ${selected ? 'border-[var(--accent-cyan)] shadow-lg shadow-cyan-500/5' : ''}
      `}
    >
      {/* Selection indicator */}
      {selected && (
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[var(--accent-cyan)] to-transparent" />
      )}

      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[var(--bg-elevated)] flex items-center justify-center text-xl font-bold text-[var(--accent-cyan)] border border-[var(--border-subtle)]">
            {icon}
          </div>
          <div>
            <h3 className="text-base font-bold text-[var(--text-primary)]">{coin}</h3>
            {dataSource === 'cde' && cdeSpec && (
              <span className="text-[10px] font-mono-trade text-[var(--accent-cyan)]">{cdeSpec.code}</span>
            )}
          </div>
        </div>
        <DataSourceToggle source={dataSource} onChange={onDataSourceChange} compact />
      </div>

      {/* Price */}
      {loading ? (
        <div className="space-y-2">
          <div className="h-8 bg-[var(--bg-elevated)] rounded animate-pulse w-3/4" />
          <div className="h-5 bg-[var(--bg-elevated)] rounded animate-pulse w-1/3" />
        </div>
      ) : error ? (
        <p className="text-sm text-[var(--accent-rose)]">Error loading</p>
      ) : (
        <>
          <div className={`text-2xl font-bold font-mono-trade mb-1 ${priceTextClass}`}>
            {formattedPrice}
          </div>
          <div className={`text-sm font-medium font-mono-trade ${changeClass}`}>
            {changeText}
            <span className="text-[var(--text-muted)] ml-2 text-xs font-normal">24h</span>
          </div>

          {/* CDE contract details */}
          {dataSource === 'cde' && cdeSpec && (
            <div className="mt-3 pt-3 border-t border-[var(--border-subtle)] space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Contract</span>
                <span className="font-mono-trade text-[var(--accent-cyan)]">{cdeSpec.symbol}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Units/contract</span>
                <span className="font-mono-trade text-[var(--text-secondary)]">{cdeSpec.units_per_contract}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Contract value</span>
                <span className="font-mono-trade text-[var(--text-primary)]">
                  ~${cdeContractValue !== null ? cdeContractValue.toFixed(2) : cdeSpec.approx_contract_value.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[var(--text-muted)]">Fee/side</span>
                <span className="font-mono-trade text-[var(--text-secondary)]">{(cdeSpec.fee_pct * 100).toFixed(3)}%</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
