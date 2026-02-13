import { useEffect, useMemo, useState } from 'react';
import { getCDESpecs, getCoinHistory, getCurrentPrices } from './api/coinsApi';
import { getPaperEquity, getPaperFills, getPaperPositions } from './api/paperApi';
import { getRecentSignals } from './api/signalsApi';
import { getAllTrades } from './api/tradesApi';
import PaperEquityTable from './components/PaperEquityTable';
import PaperPerformancePanel from './components/PaperPerformancePanel';
import PaperPositionsTable from './components/PaperPositionsTable';
import PriceCard from './components/PriceCard';
import PriceChart from './components/PriceChart';
import SignalsTable from './components/SignalsTable';
import TradesTable from './components/TradesTable';
import WalletInfo from './components/WalletInfo';
import OpsPanel from './components/OpsPanel';
import { getCurrentPrices, getCoinHistory, getCDESpecs } from './api/coinsApi';
import { getAllTrades } from './api/tradesApi';
import { getRecentSignals } from './api/signalsApi';
import { PriceData, HistoryEntry, Trade, Signal, CoinSymbol, DataSource, CDESpecs } from './types';
import { CDESpecs, CoinSymbol, DataSource, HistoryEntry, PaperEquityPoint, PaperFill, PaperPosition, PriceData, Signal, Trade } from './types';

const COINS: CoinSymbol[] = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE'];
const coinIcons: Record<CoinSymbol, string> = { BTC: '₿', ETH: 'Ξ', SOL: '◎', XRP: '✕', DOGE: 'Ð' };
type BottomTab = 'signals' | 'trades' | 'paperPositions' | 'paperEquity' | 'paperPerformance';

const formatAgo = (d: Date | null) => {
  if (!d) return '—';
  const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
};

function App() {
  const [prices, setPrices] = useState<PriceData | null>(null);
  const [cdeSpecs, setCdeSpecs] = useState<CDESpecs | null>(null);
  const [selectedCoin, setSelectedCoin] = useState<CoinSymbol>('BTC');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [paperPositions, setPaperPositions] = useState<PaperPosition[]>([]);
  const [paperEquity, setPaperEquity] = useState<PaperEquityPoint[]>([]);
  const [paperFills, setPaperFills] = useState<PaperFill[]>([]);
  const [loadingPrices, setLoadingPrices] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingTrades, setLoadingTrades] = useState(true);
  const [loadingSignals, setLoadingSignals] = useState(true);
  const [loadingPaper, setLoadingPaper] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '1d' | '1w' | '1m' | '1y'>('1d');
  const [loadingWallet, setLoadingWallet] = useState(true);
  const [bottomTab, setBottomTab] = useState<BottomTab>('signals');
  const [pricesUpdatedAt, setPricesUpdatedAt] = useState<Date | null>(null);
  const [signalsUpdatedAt, setSignalsUpdatedAt] = useState<Date | null>(null);
  const [tradesUpdatedAt, setTradesUpdatedAt] = useState<Date | null>(null);
  const [paperUpdatedAt, setPaperUpdatedAt] = useState<Date | null>(null);

  const [cardSources, setCardSources] = useState<Record<CoinSymbol, DataSource>>(Object.fromEntries(COINS.map((c) => [c, 'spot'])) as Record<CoinSymbol, DataSource>);
  const [chartSource, setChartSource] = useState<DataSource>('spot');

  useEffect(() => { getCDESpecs().then(setCdeSpecs).catch(console.error); }, []);

  useEffect(() => {
    const load = async () => {
      try { setPrices(await getCurrentPrices()); setPricesUpdatedAt(new Date()); }
      catch (err) { setError((err as Error).message); }
      finally { setLoadingPrices(false); }
    };
    load();
    const iv = setInterval(load, 3000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoadingHistory(true);
      try { setHistory(await getCoinHistory(selectedCoin, timeRange)); }
      catch (err) { setError((err as Error).message); }
      finally { setLoadingHistory(false); }
    };
    load();
  }, [selectedCoin, timeRange]);

  useEffect(() => {
    const load = async () => {
      setLoadingTrades(true);
      try { setTrades(await getAllTrades(0, 100)); setTradesUpdatedAt(new Date()); }
      finally { setLoadingTrades(false); }
    };
    load();
    const iv = setInterval(load, 10000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoadingSignals(true);
      try { setSignals(await getRecentSignals(100)); setSignalsUpdatedAt(new Date()); }
      finally { setLoadingSignals(false); }
    };
    load();
    const iv = setInterval(load, 15000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoadingPaper(true);
      try {
        const [positions, equity, fills] = await Promise.all([getPaperPositions(), getPaperEquity(250), getPaperFills(250)]);
        setPaperPositions(positions);
        setPaperEquity(equity);
        setPaperFills(fills);
        setPaperUpdatedAt(new Date());
      } finally {
        setLoadingPaper(false);
      }
    };
    load();
    const iv = setInterval(load, 8000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => { const t = setTimeout(() => setLoadingWallet(false), 1000); return () => clearTimeout(t); }, []);

    const validConf = signals.filter(s => Number.isFinite(s.confidence));
    const avgConfidence = validConf.length > 0
      ? (validConf.reduce((acc, s) => acc + s.confidence, 0) / validConf.length) * 100
      : 0;

    return { openTrades, closedTrades: closedTrades.length, winRate, actedRate, avgConfidence };
  }, [trades, signals]);

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
            <span className="hidden md:inline">Prices: {formatAgo(pricesUpdatedAt)}</span>
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

        {/* Quick stats */}
        <section className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          {[
            { label: 'Open Trades', value: kpis.openTrades.toString(), sub: `Closed ${kpis.closedTrades}` },
            { label: 'Closed Win Rate', value: `${kpis.winRate.toFixed(1)}%`, sub: `Trades refresh ${formatAgo(tradesUpdatedAt)}` },
            { label: 'Signals Acted', value: `${kpis.actedRate.toFixed(1)}%`, sub: `${signals.filter(s => s.acted_on).length}/${signals.length}` },
            { label: 'Avg Confidence', value: `${kpis.avgConfidence.toFixed(1)}%`, sub: `Signals refresh ${formatAgo(signalsUpdatedAt)}` },
            { label: 'Selected Coin', value: selectedCoin, sub: `${chartSource.toUpperCase()} chart source` },
          ].map((k) => (
            <div key={k.label} className="glass-card rounded-xl p-3">
              <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{k.label}</p>
              <p className="text-xl font-semibold mt-1 text-[var(--text-primary)]">{k.value}</p>
              <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 font-mono-trade">{k.sub}</p>
            </div>
          ))}
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

        {/* Ops */}
        <section>
          <OpsPanel />
        </section>

        {/* Tabs: Signals + Trades */}
        <section>
          <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
            <div className="flex items-center gap-1">
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
            <p className="text-[11px] text-[var(--text-muted)] font-mono-trade">
              {bottomTab === 'signals' ? `Signals refreshed ${formatAgo(signalsUpdatedAt)}` : `Trades refreshed ${formatAgo(tradesUpdatedAt)}`}
            </p>
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
  const kpis = useMemo(() => {
    const openTrades = trades.filter((t) => t.status === 'open').length;
    const closed = trades.filter((t) => t.status === 'closed');
    const winRate = closed.length ? (closed.filter((t) => (t.net_pnl ?? 0) > 0).length / closed.length) * 100 : 0;
    return { openTrades, closedTrades: closed.length, winRate };
  }, [trades]);

  return <div className="min-h-screen bg-grid font-sans antialiased"><main className="max-w-[1400px] mx-auto px-5 py-6 space-y-6">
    <section><div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">{COINS.map((coin) => (
      <PriceCard key={coin} coin={coin} icon={coinIcons[coin]} price={prices?.[coin]?.price ?? null} change24h={prices?.[coin]?.change24h ?? null} loading={loadingPrices} error={error ?? undefined} cdeSpec={cdeSpecs?.[coin]} dataSource={cardSources[coin]} onDataSourceChange={(src) => setCardSources((prev) => ({ ...prev, [coin]: src }))} selected={selectedCoin === coin} onClick={() => setSelectedCoin(coin)} />
    ))}</div></section>

    <section className="grid grid-cols-2 lg:grid-cols-5 gap-3">{[
      { label: 'Open Trades', value: kpis.openTrades.toString(), sub: `Closed ${kpis.closedTrades}` },
      { label: 'Win Rate', value: `${kpis.winRate.toFixed(1)}%`, sub: `Trades ${formatAgo(tradesUpdatedAt)}` },
      { label: 'Signals', value: signals.length.toString(), sub: `Updated ${formatAgo(signalsUpdatedAt)}` },
      { label: 'Paper Positions', value: paperPositions.length.toString(), sub: `Paper ${formatAgo(paperUpdatedAt)}` },
      { label: 'Selected Coin', value: selectedCoin, sub: `${chartSource.toUpperCase()} chart source` },
    ].map((k) => <div key={k.label} className="glass-card rounded-xl p-3"><p className="text-[10px] uppercase text-[var(--text-muted)]">{k.label}</p><p className="text-xl font-semibold">{k.value}</p><p className="text-[11px] text-[var(--text-secondary)]">{k.sub}</p></div>)}</section>

    <PriceChart data={history} symbol={selectedCoin} loading={loadingHistory} timeRange={timeRange} setTimeRange={setTimeRange} dataSource={chartSource} onDataSourceChange={setChartSource} cdeSpec={cdeSpecs?.[selectedCoin]} />
    <WalletInfo loading={loadingWallet} />

    <section>
      <div className="flex flex-wrap gap-2 mb-4">
        {[
          ['signals', `Signals (${signals.length})`],
          ['trades', `Trades (${trades.length})`],
          ['paperPositions', `Paper Positions (${paperPositions.length})`],
          ['paperEquity', `Paper Equity (${paperEquity.length})`],
          ['paperPerformance', 'Paper Performance'],
        ].map(([key, label]) => <button key={key} onClick={() => setBottomTab(key as BottomTab)} className={`px-4 py-2 rounded-lg text-sm ${bottomTab === key ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]' : 'text-[var(--text-muted)]'}`}>{label}</button>)}
      </div>
      {bottomTab === 'signals' && <SignalsTable signals={signals} loading={loadingSignals} />}
      {bottomTab === 'trades' && <TradesTable trades={trades} loading={loadingTrades} />}
      {bottomTab === 'paperPositions' && <PaperPositionsTable positions={paperPositions} loading={loadingPaper} />}
      {bottomTab === 'paperEquity' && <PaperEquityTable points={paperEquity} loading={loadingPaper} />}
      {bottomTab === 'paperPerformance' && <PaperPerformancePanel equity={paperEquity} fills={paperFills} loading={loadingPaper} />}
    </section>

    <p className="text-xs text-[var(--text-muted)]">Prices refresh {formatAgo(pricesUpdatedAt)}</p>
  </main></div>;
}

export default App;
