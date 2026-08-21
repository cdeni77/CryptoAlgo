import { useMemo, useState } from 'react';

import { getCDEPrices, getCoinHistory, getCurrentPrices } from '../api/coinsApi';
import { getPaperFills } from '../api/paperApi';
import { getRecentSignals } from '../api/signalsApi';
import { getWallet } from '../api/walletApi';
import PaperFillsTable from '../components/PaperFillsTable';
import PriceChart from '../components/PriceChart';
import SignalsTable from '../components/SignalsTable';
import { Empty, ErrorBlock, Freshness, Panel, Spinner } from '../components/StateBlock';
import { usePolling } from '../hooks/usePolling';
import { CoinSymbol, WalletAsset } from '../types';

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
  const [coin, setCoin] = useState<CoinSymbol>('ETH');
  const [range, setRange] = useState<Range>('1d');
  const [chartMode, setChartMode] = useState<ChartMode>('candle');
  const [priceSource, setPriceSource] = useState<'spot' | 'cde'>('spot');

  // History reloads when the instrument or range changes; the rest polls at the
  // rate its data moves, and everything stops while the tab is hidden. The
  // wallet in particular calls Coinbase, and it was being refetched every minute
  // forever regardless of whether anyone was looking.
  const history = usePolling(() => getCoinHistory(coin, range), 60_000, [coin, range]);
  const fills = usePolling(() => getPaperFills(50), 20_000);
  const signals = usePolling(() => getRecentSignals(50), 20_000);
  const spot = usePolling(getCurrentPrices, 5_000);
  const cde = usePolling(getCDEPrices, 5_000);
  const wallet = usePolling(getWallet, 120_000);

  const priceState = priceSource === 'cde' ? cde : spot;
  const prices = priceState.data;
  const coinPrice = prices?.[coin]?.price;
  const coinChange = prices?.[coin]?.change24h;

  const coinSignals = useMemo(
    () => (signals.data ?? []).filter((s) => s.coin === coin),
    [signals.data, coin],
  );
  const coinFills = useMemo(
    () => (fills.data ?? []).filter((f) => f.coin === coin),
    [fills.data, coin],
  );

  // The exchange's rolling 24h for the 1d range; computed from the chart's own
  // window otherwise, since no endpoint reports a 1w or 1y change.
  const bars = history.data ?? [];
  const rangeChange =
    range === '1d'
      ? (coinChange ?? null)
      : bars.length >= 2
        ? ((bars[bars.length - 1].close - bars[0].open) / bars[0].open) * 100
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

        {/* Price display. A missing price used to render as nothing at all, so a
            dead price feed and a market with no quote looked the same. */}
        {!coinPrice && priceState.error && (
          <div className="ml-auto max-w-md">
            <ErrorBlock error={priceState.error} onRetry={priceState.refresh} compact />
          </div>
        )}
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
        <div className="relative h-72">
          {history.refreshing && bars.length > 0 && (
            <div className="absolute right-2 top-2 z-10">
              <Freshness lastUpdated={history.lastUpdated} refreshing />
            </div>
          )}
          {history.error && bars.length === 0 ? (
            <ErrorBlock error={history.error} onRetry={history.refresh} />
          ) : history.loading && bars.length === 0 ? (
            <Spinner label={`Loading ${coin} history`} />
          ) : bars.length === 0 ? (
            <Empty message={`No ${range} history for ${coin}.`} />
          ) : (
            <PriceChart data={bars} fills={coinFills} coin={coin} mode={chartMode} />
          )}
        </div>
      </div>

      {/* Real portfolio — external holdings only, never paper. */}
      {wallet.error && !wallet.data && (
        <ErrorBlock error={wallet.error} onRetry={wallet.refresh} compact />
      )}
      {wallet.data && (() => {
        const held = wallet.data;
        const spotVal   = held.coinbase?.spot?.value_usd ?? 0;
        const ledgerVal = held.ledger?.value_usd ?? 0;
        const total = spotVal + ledgerVal;
        if (total <= 0) return null;
        const spotAssets  = (held.coinbase?.spot?.assets  ?? []).filter(a => a.value_usd >= 0.01);
        const ledgerAssets = (held.ledger?.assets ?? []).filter(a => a.value_usd >= 0.01);
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
            <div className="flex items-center gap-3">
              <span className="text-tx-muted text-xs">{coinSignals.length} total</span>
              <Freshness
                lastUpdated={signals.lastUpdated}
                refreshing={signals.refreshing}
                error={signals.error}
              />
            </div>
          </div>
          <Panel
            state={signals}
            emptyWhen={() => coinSignals.length === 0}
            emptyMessage={`No signals for ${coin}.`}
            loadingLabel="Loading signals"
          >
            {() => <SignalsTable signals={coinSignals} limit={20} />}
          </Panel>
        </div>
        <div className="glass-card rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Fills — {coin}</span>
            <div className="flex items-center gap-3">
              <span className="text-tx-muted text-xs">{coinFills.length} total</span>
              <Freshness
                lastUpdated={fills.lastUpdated}
                refreshing={fills.refreshing}
                error={fills.error}
              />
            </div>
          </div>
          <Panel
            state={fills}
            emptyWhen={() => coinFills.length === 0}
            emptyMessage={`No fills for ${coin}.`}
            loadingLabel="Loading fills"
          >
            {() => <PaperFillsTable fills={coinFills} limit={20} />}
          </Panel>
        </div>
      </div>
    </div>
  );
}
