import { useEffect, useState } from 'react';
import PriceCard from './components/PriceCard';
import PriceChart from './components/PriceChart';
import TradesTable from './components/TradesTable';
import SignalsTable from './components/SignalsTable';
import WalletInfo from './components/WalletInfo';
import { getCurrentPrices, getCoinHistory, getCDESpecs } from './api/coinsApi';
import { getAllTrades } from './api/tradesApi';
import { getRecentSignals } from './api/signalsApi';
import { PriceData, HistoryEntry, Trade, Signal, CoinSymbol, DataSource, CDESpecs } from './types';

const COINS: CoinSymbol[] = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE'];

const coinIcons: Record<CoinSymbol, string> = {
  BTC: '₿',
  ETH: 'Ξ',
  SOL: '◎',
  XRP: '✕',
  DOGE: 'Ð',
};

type BottomTab = 'trades' | 'signals';

function App() {
  const [prices, setPrices] = useState<PriceData | null>(null);
  const [cdeSpecs, setCdeSpecs] = useState<CDESpecs | null>(null);
  const [selectedCoin, setSelectedCoin] = useState<CoinSymbol>('BTC');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loadingPrices, setLoadingPrices] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingTrades, setLoadingTrades] = useState(true);
  const [loadingSignals, setLoadingSignals] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '1d' | '1w' | '1m' | '1y'>('1d');
  const [loadingWallet, setLoadingWallet] = useState(true);
  const [bottomTab, setBottomTab] = useState<BottomTab>('signals');

  // Per-card data source
  const [cardSources, setCardSources] = useState<Record<CoinSymbol, DataSource>>(
    Object.fromEntries(COINS.map(c => [c, 'spot'])) as Record<CoinSymbol, DataSource>
  );
  const [chartSource, setChartSource] = useState<DataSource>('spot');

  // Fetch CDE specs once
  useEffect(() => {
    getCDESpecs().then(setCdeSpecs).catch(console.error);
  }, []);

  // Fetch prices on interval
  useEffect(() => {
    const load = async () => {
      try {
        const data = await getCurrentPrices();
        setPrices(data);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoadingPrices(false);
      }
    };
    load();
    const iv = setInterval(load, 3000);
    return () => clearInterval(iv);
  }, []);

  // Fetch history when coin or range changes
  useEffect(() => {
    const load = async () => {
      setLoadingHistory(true);
      try {
        const data = await getCoinHistory(selectedCoin, timeRange);
        setHistory(data);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoadingHistory(false);
      }
    };
    load();
  }, [selectedCoin, timeRange]);

  // Fetch trades
  useEffect(() => {
    const load = async () => {
      setLoadingTrades(true);
      try {
        setTrades(await getAllTrades(0, 100));
      } catch (err) {
        console.error('Trades fetch error:', err);
      } finally {
        setLoadingTrades(false);
      }
    };
    load();
    const iv = setInterval(load, 10000);
    return () => clearInterval(iv);
  }, []);

  // Fetch signals
  useEffect(() => {
    const load = async () => {
      setLoadingSignals(true);
      try {
        setSignals(await getRecentSignals(100));
      } catch (err) {
        console.error('Signals fetch error:', err);
      } finally {
        setLoadingSignals(false);
      }
    };
    load();
    const iv = setInterval(load, 15000);
    return () => clearInterval(iv);
  }, []);

  // Wallet
  useEffect(() => {
    const t = setTimeout(() => setLoadingWallet(false), 1000);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="min-h-screen bg-grid font-sans antialiased">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-[var(--border-subtle)] bg-[var(--bg-primary)]/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-5 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">
              Trading Terminal
            </h1>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono-trade text-[var(--text-muted)]">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[var(--accent-emerald)] animate-pulse" />
              <span className="text-[var(--accent-emerald)]">LIVE</span>
            </span>
            <span className="hidden sm:inline">
              {new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-5 py-6 space-y-6">
        {/* Price Cards */}
        <section>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
            {COINS.map((coin) => (
              <PriceCard
                key={coin}
                coin={coin}
                icon={coinIcons[coin]}
                price={prices?.[coin]?.price ?? null}
                change24h={prices?.[coin]?.change24h ?? null}
                loading={loadingPrices}
                error={error ?? undefined}
                cdeSpec={cdeSpecs?.[coin]}
                dataSource={cardSources[coin]}
                onDataSourceChange={(src) =>
                  setCardSources(prev => ({ ...prev, [coin]: src }))
                }
                selected={selectedCoin === coin}
                onClick={() => setSelectedCoin(coin)}
              />
            ))}
          </div>
        </section>

        {/* Chart */}
        <section>
          <PriceChart
            data={history}
            symbol={selectedCoin}
            loading={loadingHistory}
            timeRange={timeRange}
            setTimeRange={setTimeRange}
            dataSource={chartSource}
            onDataSourceChange={setChartSource}
            cdeSpec={cdeSpecs?.[selectedCoin]}
          />
        </section>

        {/* Wallet */}
        <section>
          <WalletInfo loading={loadingWallet} />
        </section>

        {/* Tabs: Signals + Trades */}
        <section>
          <div className="flex items-center gap-1 mb-4">
            <button
              onClick={() => setBottomTab('signals')}
              className={`
                px-4 py-2 rounded-lg text-sm font-medium font-mono-trade transition-all
                ${bottomTab === 'signals'
                  ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }
              `}
            >
              Signals
              {signals.length > 0 && (
                <span className="ml-2 text-[10px] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] px-1.5 py-0.5 rounded-full">
                  {signals.length}
                </span>
              )}
            </button>
            <button
              onClick={() => setBottomTab('trades')}
              className={`
                px-4 py-2 rounded-lg text-sm font-medium font-mono-trade transition-all
                ${bottomTab === 'trades'
                  ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }
              `}
            >
              Trade History
              {trades.length > 0 && (
                <span className="ml-2 text-[10px] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] px-1.5 py-0.5 rounded-full">
                  {trades.length}
                </span>
              )}
            </button>
          </div>

          {bottomTab === 'signals' ? (
            <SignalsTable signals={signals} loading={loadingSignals} />
          ) : (
            <TradesTable trades={trades} loading={loadingTrades} />
          )}
        </section>
      </main>

      <footer className="border-t border-[var(--border-subtle)] mt-8">
        <div className="max-w-[1400px] mx-auto px-5 py-5 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] font-mono-trade text-[var(--text-muted)]">
          <span>© {new Date().getFullYear()} Trading Terminal</span>
          <span>Spot data via Coinbase · CDE contracts via Coinbase Financial Markets</span>
        </div>
      </footer>
    </div>
  );
}

export default App;