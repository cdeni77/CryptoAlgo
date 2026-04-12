import { useEffect, useState } from 'react';
import { getCoinHistory, getCurrentPrices, getCDEPrices } from '../api/coinsApi';
import { getPaperFills } from '../api/paperApi';
import { getRecentSignals as getSignals } from '../api/signalsApi';
import { getWallet } from '../api/walletApi';
import { CoinSymbol, HistoryEntry, PaperFill, Signal, PriceData, WalletData, WalletAsset } from '../types';
import PriceChart from '../components/PriceChart';
import SignalsTable from '../components/SignalsTable';
import PaperFillsTable from '../components/PaperFillsTable';

function fmt(v: number) {
  return `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtAmt(amount: number, asset: string) {
  // Show enough decimal places based on coin type
  const decimals = ['BTC','ETH'].includes(asset) ? 5 : ['SOL','AVAX','LINK','LTC'].includes(asset) ? 3 : 2;
  return amount.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: decimals });
}

function AssetRow({ a }: { a: WalletAsset }) {
  if (a.value_usd < 0.01) return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[rgba(56,189,248,0.06)] last:border-0">
      <div className="flex items-center gap-2">
        <span className="text-tx-secondary text-xs font-mono font-medium w-12">{a.asset}</span>
        <span className="text-tx-muted text-[10px] font-mono">({fmtAmt(a.amount, a.asset)})</span>
      </div>
      <span className="text-tx-primary text-xs font-mono font-semibold">{fmt(a.value_usd)}</span>
    </div>
  );
}

const COINS: CoinSymbol[] = ['ETH','BTC','AVAX','SOL','XRP','DOGE','ADA','LINK','LTC'];
const RANGES = ['1h','1d','1w','1m','1y'] as const;
type Range = typeof RANGES[number];
type ChartMode = 'candle' | 'line';


export default function TradingPage() {
  const [coin,      setCoin]      = useState<CoinSymbol>('ETH');
  const [range,     setRange]     = useState<Range>('1d');
  const [chartMode, setChartMode] = useState<ChartMode>('candle');
  const [history,   setHistory]   = useState<HistoryEntry[]>([]);
  const [fills,     setFills]     = useState<PaperFill[]>([]);
  const [signals,   setSignals]   = useState<Signal[]>([]);
  const [spotPrices,  setSpotPrices]  = useState<PriceData | null>(null);
  const [cdePrices,   setCdePrices]   = useState<PriceData | null>(null);
  const [priceSource, setPriceSource] = useState<'spot' | 'cde'>('spot');
  const [loading,   setLoading]   = useState(false);
  const [wallet,    setWallet]    = useState<WalletData | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getCoinHistory(coin, range).then(d => { if (!cancelled) { setHistory(d); setLoading(false); } }).catch(() => setLoading(false));
    return () => { cancelled = true; };
  }, [coin, range]);

  useEffect(() => {
    const load = async () => {
      try { setFills(await getPaperFills(50)); } catch { /* empty */ }
      try { setSignals(await getSignals(50)); } catch { /* empty */ }
      try { setSpotPrices(await getCurrentPrices()); } catch { /* empty */ }
      try { setCdePrices(await getCDEPrices()); } catch { /* empty */ }
    };
    const loadWallet = async () => {
      try { setWallet(await getWallet()); } catch { /* empty */ }
    };
    load();
    loadWallet();
    const id = setInterval(load, 8000);
    const walletId = setInterval(loadWallet, 60000);
    return () => { clearInterval(id); clearInterval(walletId); };
  }, []);

  const prices = priceSource === 'cde' ? cdePrices : spotPrices;
  const coinPrice = prices?.[coin]?.price;
  const coinChange = prices?.[coin]?.change24h;
  const coinSignals = signals.filter(s => s.coin === coin);
  const coinFills   = fills.filter(f => f.coin === coin);

  // Range % change: use exchange's rolling 24h for 1d, compute from chart history otherwise
  const rangeChange = range === '1d'
    ? (coinChange ?? null)
    : history.length >= 2
      ? ((history[history.length - 1].close - history[0].open) / history[0].open) * 100
      : null;
  const priceColor = coinChange == null ? 'text-tx-primary' : coinChange >= 0 ? 'text-accent-emerald' : 'text-accent-rose';

  return (
    <div className="p-6 space-y-5 max-w-[1600px]">
      {/* Coin selector */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex gap-0.5 p-0.5 rounded bg-[rgba(56,189,248,0.05)] border border-[rgba(56,189,248,0.08)] flex-shrink-0">
          {(['spot', 'cde'] as const).map(s => (
            <button
              key={s}
              onClick={() => setPriceSource(s)}
              className={`px-2.5 py-1 rounded text-[10px] font-mono transition-all ${
                s === priceSource
                  ? 'bg-accent-cyan/15 text-accent-cyan'
                  : 'text-tx-muted hover:text-tx-secondary'
              }`}
            >
              {s.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="flex gap-1 flex-wrap">
          {COINS.map(c => (
            <button
              key={c}
              onClick={() => setCoin(c)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                c === coin
                  ? 'border-[rgba(56,189,248,0.4)] bg-[rgba(56,189,248,0.08)] text-accent-cyan'
                  : 'border-[rgba(56,189,248,0.08)] text-tx-muted hover:text-tx-secondary hover:border-[rgba(56,189,248,0.15)]'
              }`}
            >
              {c}
            </button>
          ))}
        </div>

        {/* Price display */}
        {coinPrice && (
          <div className="ml-auto flex items-baseline gap-2">
            <span className={`font-mono text-2xl font-semibold ${priceColor}`}>
              ${coinPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: coinPrice > 100 ? 2 : 4 })}
            </span>
            {coinChange != null && (
              <span className={`font-mono text-sm ${coinChange >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}`}>
                {coinChange >= 0 ? '+' : ''}{coinChange.toFixed(2)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">{coin} / USD</span>
            {rangeChange != null && (
              <span className={`font-mono text-xs font-semibold ${rangeChange >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}`}>
                {rangeChange >= 0 ? '+' : ''}{rangeChange.toFixed(2)}%
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Candle / Line toggle */}
            <div className="flex gap-0.5 p-0.5 rounded bg-[rgba(56,189,248,0.05)] border border-[rgba(56,189,248,0.08)]">
              {(['candle', 'line'] as ChartMode[]).map(m => (
                <button
                  key={m}
                  onClick={() => setChartMode(m)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono transition-all ${
                    m === chartMode
                      ? 'bg-accent-cyan/15 text-accent-cyan'
                      : 'text-tx-muted hover:text-tx-secondary'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
            {/* Range buttons */}
            <div className="flex gap-1">
              {RANGES.map(r => (
                <button
                  key={r}
                  onClick={() => setRange(r)}
                  className={`px-2.5 py-1 rounded text-xs font-mono transition-all ${
                    r === range
                      ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30'
                      : 'text-tx-muted hover:text-tx-secondary border border-transparent'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="h-72 relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-[rgba(8,12,20,0.6)] z-10 rounded-lg">
              <span className="text-tx-muted text-sm">Loading…</span>
            </div>
          )}
          <PriceChart data={history} fills={coinFills} coin={coin} mode={chartMode} />
        </div>
      </div>

      {/* Real portfolio — external holdings only (no paper) */}
      {wallet && (() => {
        const spotVal   = wallet.coinbase?.spot?.value_usd ?? 0;
        const ledgerVal = wallet.ledger?.value_usd ?? 0;
        const total = spotVal + ledgerVal;
        if (total <= 0) return null;
        const spotAssets  = (wallet.coinbase?.spot?.assets  ?? []).filter(a => a.value_usd >= 0.01);
        const ledgerAssets = (wallet.ledger?.assets ?? []).filter(a => a.value_usd >= 0.01);
        return (
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Real Portfolio</span>
              <span className="font-mono text-tx-primary text-sm font-semibold">{fmt(total)} external</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {spotVal > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-tx-muted text-[10px] uppercase tracking-widest">Coinbase Spot</span>
                    <span className="font-mono text-tx-primary text-sm font-semibold">{fmt(spotVal)}</span>
                  </div>
                  {spotAssets.map(a => <AssetRow key={a.asset} a={a} />)}
                </div>
              )}
              {ledgerVal > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-tx-muted text-[10px] uppercase tracking-widest">Ledger</span>
                    <span className="font-mono text-tx-primary text-sm font-semibold">{fmt(ledgerVal)}</span>
                  </div>
                  {ledgerAssets.map(a => <AssetRow key={a.asset} a={a} />)}
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* Signals + fills */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-card rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Signals — {coin}</span>
            <span className="text-tx-muted text-xs">{coinSignals.length} total</span>
          </div>
          <SignalsTable signals={coinSignals} limit={20} />
        </div>
        <div className="glass-card rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Fills — {coin}</span>
            <span className="text-tx-muted text-xs">{coinFills.length} total</span>
          </div>
          <PaperFillsTable fills={coinFills} limit={20} />
        </div>
      </div>
    </div>
  );
}
