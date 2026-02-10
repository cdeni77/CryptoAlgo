import { useEffect, useState } from 'react';
import PriceCard from './components/PriceCard';
import PriceChart from './components/PriceChart';
import TradesTable from './components/TradesTable';
import WalletInfo from './components/WalletInfo';  // ← New: Import WalletInfo
import { getCurrentPrices, getCoinHistory } from './api/coinsApi';
import { getAllTrades } from './api/tradesApi';
import { PriceData, HistoryEntry, Trade } from './types';

const coinIcons: Record<'BTC' | 'ETH' | 'SOL', string> = {
  BTC: '₿',
  ETH: 'Ξ',
  SOL: '◎'
};

function App() {
  const [prices, setPrices] = useState<PriceData | null>(null);
  const [selectedCoin, setSelectedCoin] = useState<'BTC' | 'ETH' | 'SOL'>('BTC');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loadingPrices, setLoadingPrices] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingTrades, setLoadingTrades] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ← New: time range state for the chart
  const [timeRange, setTimeRange] = useState<'1h' | '1d' | '1w' | '1m' | '1y'>('1d');

  // ← New: Wallet loading state (to sync with overall app if needed, but WalletInfo handles its own)
  const [loadingWallet, setLoadingWallet] = useState(true);

  useEffect(() => {
    const loadPrices = async () => {
      try {
        const data = await getCurrentPrices();
        setPrices(data);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoadingPrices(false);
      }
    };

    loadPrices();
    const interval = setInterval(loadPrices, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const loadHistory = async () => {
      setLoadingHistory(true);
      try {
        // ← Now passes timeRange instead of fixed days
        const data = await getCoinHistory(selectedCoin, timeRange);
        setHistory(data);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoadingHistory(false);
      }
    };

    loadHistory();
  }, [selectedCoin, timeRange]); // ← Added timeRange dependency

  useEffect(() => {
    const loadTrades = async () => {
      setLoadingTrades(true);
      try {
        const data = await getAllTrades(0, 50);
        setTrades(data);
      } catch (err) {
        console.error('Trades fetch error:', err);
      } finally {
        setLoadingTrades(false);
      }
    };

    loadTrades();
  }, []);

  // ← New: Optional effect to sync wallet loading (WalletInfo refreshes internally, but this sets initial loading)
  useEffect(() => {
    // WalletInfo handles its own fetching, so we just wait a bit or sync if needed
    const timer = setTimeout(() => setLoadingWallet(false), 1000); // Placeholder; adjust if you want to await getWallet here
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-indigo-950/40 to-gray-950 text-gray-100 font-sans antialiased">
      <header className="bg-gradient-to-r from-indigo-950/80 to-gray-900/80 backdrop-blur-lg border-b border-indigo-700/50 sticky top-0 z-50 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <h1 className="text-3xl md:text-4xl font-extrabold bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent tracking-tight">
            Crypto Trading Dashboard
          </h1>
          <div className="flex items-center gap-6 text-sm text-indigo-300">
            <span className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-green-400 animate-pulse"></span>
              Live
            </span>
            <span>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Live Prices */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-10 text-center bg-gradient-to-r from-blue-300 to-indigo-400 bg-clip-text text-transparent">
            Live Market Prices
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {['BTC', 'ETH', 'SOL'].map((coin) => (
              <PriceCard
                key={coin}
                coin={coin}
                icon={coinIcons[coin as keyof typeof coinIcons]}
                price={prices?.[coin as keyof PriceData]?.price ?? null}
                change24h={prices?.[coin as keyof PriceData]?.change24h ?? null}
                loading={loadingPrices}
                error={error ?? undefined}
              />
            ))}
          </div>
        </section>

        {/* Chart */}
        <section className="mb-16">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-10 gap-6">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-300 to-indigo-400 bg-clip-text text-transparent">
              {selectedCoin} Price History
            </h2>
            <div className="flex flex-wrap gap-4">
              {(['BTC', 'ETH', 'SOL'] as const).map((coin) => (
                <button
                  key={coin}
                  onClick={() => setSelectedCoin(coin)}
                  className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 shadow-lg ${
                    selectedCoin === coin
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-blue-700/40 scale-105'
                      : 'bg-gray-800/70 text-gray-300 hover:bg-gray-700/80 hover:shadow-indigo-600/30 border border-indigo-700/50'
                  }`}
                >
                  {coin}
                </button>
              ))}
            </div>
          </div>

          {/* Updated: pass timeRange and setter */}
          <PriceChart 
            data={history} 
            symbol={selectedCoin} 
            loading={loadingHistory}
            timeRange={timeRange}
            setTimeRange={setTimeRange}
          />
        </section>

        {/* ← New: Wallet Overview Section (above Trades) */}
        <section className="mb-16">
          <WalletInfo loading={loadingWallet} />
        </section>

        {/* Trades Table */}
        <section>
          <h2 className="text-3xl font-bold mb-10 bg-gradient-to-r from-blue-300 to-indigo-400 bg-clip-text text-transparent">
            Trade History
          </h2>

          {loadingTrades ? (
            <div className="glass rounded-2xl p-16 flex flex-col items-center justify-center h-96">
              <div className="w-20 h-20 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-8"></div>
              <p className="text-xl text-gray-300">Loading your trade history...</p>
            </div>
          ) : trades.length === 0 ? (
            <div className="glass rounded-2xl p-16 text-center">
              <svg className="w-24 h-24 mx-auto mb-8 text-indigo-600/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <h3 className="text-2xl font-bold text-gray-300 mb-4">No trades yet</h3>
              <p className="text-lg text-gray-500 max-w-md mx-auto">
                Your trading bot will automatically log and display trades here once it starts executing.
              </p>
            </div>
          ) : (
            <TradesTable trades={trades} />
          )}
        </section>
      </main>

      <footer className="border-t border-indigo-900/50 bg-indigo-950/50 backdrop-blur-md mt-20">
        <div className="max-w-7xl mx-auto px-6 py-10 text-center text-gray-400 text-sm">
          © {new Date().getFullYear()} Crypto Trading Dashboard • Real-time data from Coinbase
        </div>
      </footer>
    </div>
  );
}

export default App;