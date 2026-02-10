import { useEffect, useRef, useState } from 'react';

interface PriceCardProps {
  coin: string;
  icon: string;
  price: number | null;
  change24h: number | null;
  loading: boolean;
  error?: string;
}

export default function PriceCard({ coin, icon, price, change24h, loading, error }: PriceCardProps) {
  const prevPriceRef = useRef<number | null>(null);
  const [direction, setDirection] = useState<'up' | 'down' | null>(null);

  useEffect(() => {
    if (loading || error || price === null) {
      prevPriceRef.current = null;
      setDirection(null);
      return;
    }

    // Coerce to number early
    const currentPrice = Number(price);
    if (isNaN(currentPrice)) {
      console.log(`[${coin}] Invalid price: ${price}`); // Temp log for bad data
      prevPriceRef.current = null;
      setDirection(null);
      return;
    }

    const prev = prevPriceRef.current;

    if (prev !== null) {
      const diff = currentPrice - prev;
      const threshold = 0.01;
      if (Math.abs(diff) > threshold) {
        setDirection(diff > 0 ? 'up' : 'down');

        const timer = setTimeout(() => {
          setDirection(null);
        }, 1200); 

        return () => {
          clearTimeout(timer);
        };
      } else {
        setDirection(null); 
      }
    }

    prevPriceRef.current = currentPrice;

  }, [price, loading, error, coin]); 

  const priceTextClass = 
    direction === 'up' ? 'text-green-400 animate-price-up' :
    direction === 'down' ? 'text-red-400 animate-price-down' :
    'text-white';

  const formattedPrice =
    price !== null
      ? `$${Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
      : '—';

  const changeClass =
    change24h === null
      ? 'text-gray-500'
      : change24h >= 0
        ? 'text-green-400'
        : 'text-red-400';

  const changeText =
    change24h === null
      ? '—'
      : `${change24h >= 0 ? '+' : ''}${change24h.toFixed(2)}%`;

  return (
    <div className="glass rounded-2xl p-6 transition-all duration-300 hover:scale-[1.02] hover:shadow-indigo-900/40">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-600 to-blue-600 flex items-center justify-center text-white text-2xl font-bold shadow-md">
            {icon}
          </div>
          <h3 className="text-xl font-bold text-indigo-300">{coin}</h3>
        </div>
        <span className="text-xs px-3 py-1 rounded-full bg-indigo-900/60 text-indigo-200">
          Live
        </span>
      </div>

      {loading ? (
        <div className="space-y-3">
          <div className="h-10 bg-gray-700/50 rounded animate-pulse"></div>
          <div className="h-6 w-24 bg-gray-700/50 rounded animate-pulse mx-auto"></div>
        </div>
      ) : error ? (
        <div className="text-red-400 text-sm text-center">Error loading</div>
      ) : (
        <div className="text-center">
          <div
            className={`text-4xl md:text-5xl font-mono font-extrabold tracking-tight transition-colors duration-300 ${priceTextClass}`}
          >
            {formattedPrice}
          </div>

          {/* 24h change display */}
          <div className={`mt-3 text-lg font-semibold flex items-center justify-center gap-1.5 ${changeClass}`}>
            {change24h !== null && change24h >= 0 ? '▲' : change24h !== null ? '▼' : ''}
            {changeText}
            <span className="text-gray-400 text-base ml-1">24h</span>
          </div>
        </div>
      )}
    </div>
  );
}