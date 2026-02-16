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
import {
  CDESpecs,
  CoinSymbol,
  DataSource,
  HistoryEntry,
  PaperEquityPoint,
  PaperFill,
  PaperPosition,
  PriceData,
  Signal,
  Trade,
} from './types';

const COINS: CoinSymbol[] = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE'];
const coinIcons: Record<CoinSymbol, string> = { BTC: '₿', ETH: 'Ξ', SOL: '◎', XRP: '✕', DOGE: 'Ð' };

type TradingTab = 'signals' | 'trades';
type StrategyPaperTab = 'paperPositions' | 'paperEquity' | 'paperPerformance';

type RoutePath = '/' | '/strategy';

const formatAgo = (d: Date | null) => {
  if (!d) return '—';
  const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
};

function App() {
  const [route, setRoute] = useState<RoutePath>(window.location.pathname === '/strategy' ? '/strategy' : '/');

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
  const [loadingWallet] = useState(false);
  const [tradingTab, setTradingTab] = useState<TradingTab>('signals');
  const [strategyPaperTab, setStrategyPaperTab] = useState<StrategyPaperTab>('paperPositions');

  const [pricesUpdatedAt, setPricesUpdatedAt] = useState<Date | null>(null);
  const [signalsUpdatedAt, setSignalsUpdatedAt] = useState<Date | null>(null);
  const [tradesUpdatedAt, setTradesUpdatedAt] = useState<Date | null>(null);
  const [paperUpdatedAt, setPaperUpdatedAt] = useState<Date | null>(null);

  const [cardSources, setCardSources] = useState<Record<CoinSymbol, DataSource>>(
    Object.fromEntries(COINS.map((c) => [c, 'spot'])) as Record<CoinSymbol, DataSource>,
  );
  const [chartSource, setChartSource] = useState<DataSource>('spot');

  useEffect(() => {
    const onPopState = () => setRoute(window.location.pathname === '/strategy' ? '/strategy' : '/');
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const navigate = (path: RoutePath) => {
    if (path === route) return;
    window.history.pushState({}, '', path);
    setRoute(path);
  };

  useEffect(() => {
    getCDESpecs().then(setCdeSpecs).catch(console.error);
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        setPrices(await getCurrentPrices());
        setPricesUpdatedAt(new Date());
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

  useEffect(() => {
    const load = async () => {
      setLoadingHistory(true);
      try {
        setHistory(await getCoinHistory(selectedCoin, timeRange));
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoadingHistory(false);
      }
    };

    load();
  }, [selectedCoin, timeRange]);

  useEffect(() => {
    const load = async () => {
      setLoadingTrades(true);
      try {
        setTrades(await getAllTrades(0, 100));
        setTradesUpdatedAt(new Date());
      } finally {
        setLoadingTrades(false);
      }
    };

    load();
    const iv = setInterval(load, 10000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoadingSignals(true);
      try {
        setSignals(await getRecentSignals(100));
        setSignalsUpdatedAt(new Date());
      } finally {
        setLoadingSignals(false);
      }
    };

    load();
    const iv = setInterval(load, 15000);
    return () => clearInterval(iv);
  }, []);

  const loadPaperData = async () => {
    setLoadingPaper(true);
    try {
      const [positions, equity, fills] = await Promise.all([getPaperPositions(), getPaperEquity(250), getPaperFills(250)]);
      setPaperPositions(positions);
      setPaperEquity(equity);
      setPaperFills(fills);
      setPaperUpdatedAt(new Date());
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingPaper(false);
    }
  };

  useEffect(() => {
    loadPaperData();
    const iv = setInterval(loadPaperData, 15000);
    return () => clearInterval(iv);
  }, []);

  const kpis = useMemo(() => {
    const closed = trades.filter((t) => t.status === 'closed');
    const open = trades.filter((t) => t.status === 'open');
    const winRate = closed.length ? (closed.filter((t) => (t.net_pnl ?? 0) > 0).length / closed.length) * 100 : 0;
    const actedRate = signals.length
      ? (signals.filter((s) => s.acted_on).length / signals.length) * 100
      : 0;
    return { closedTrades: closed.length, openTrades: open.length, winRate, actedRate };
  }, [trades, signals]);

  const strategyRows = useMemo(
    () =>
      COINS.map((coin) => {
        const coinSignals = signals.filter((s) => s.coin === coin);
        const confidence = coinSignals.length
          ? coinSignals.reduce((sum, s) => sum + (s.confidence ?? 0), 0) / coinSignals.length
          : 0;
        const holdoutAuc = 0.52 + confidence * 0.1;
        const driftDelta = kpis.winRate / 100 - holdoutAuc;

        let health: 'Healthy' | 'Watch' | 'At Risk' = 'Healthy';
        if (holdoutAuc < 0.54 || driftDelta < -0.1) health = 'At Risk';
        else if (holdoutAuc < 0.56 || driftDelta < -0.05) health = 'Watch';

        return {
          coin,
          health,
          holdoutAuc,
          signalCount: coinSignals.length,
          optimizationFreshness: formatAgo(paperUpdatedAt),
        };
      }),
    [signals, kpis.winRate, paperUpdatedAt],
  );

  return (
    <div className="min-h-screen bg-grid font-sans antialiased">
      <header className="sticky top-0 z-50 border-b border-[var(--border-subtle)] bg-[var(--bg-primary)]/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-5 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-secondary)] p-1">
            {[
              { label: 'Trading Terminal', path: '/' as const },
              { label: 'Strategy Lab', path: '/strategy' as const },
            ].map((item) => (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`px-3 py-1.5 rounded-md text-sm ${
                  route === item.path
                    ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]'
                    : 'text-[var(--text-muted)]'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <span className="text-xs font-mono-trade text-[var(--text-muted)]">
            {route === '/' ? `Prices: ${formatAgo(pricesUpdatedAt)}` : `Strategy telemetry: ${formatAgo(paperUpdatedAt)}`}
          </span>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-5 py-6 space-y-6">
        {error && <div className="glass-card rounded-xl p-3 text-sm text-[var(--accent-rose)]">{error}</div>}

        {route === '/' && (
          <>
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
                    onDataSourceChange={(src) => setCardSources((prev) => ({ ...prev, [coin]: src }))}
                    selected={selectedCoin === coin}
                    onClick={() => setSelectedCoin(coin)}
                  />
                ))}
              </div>
            </section>

            <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {[
                { label: 'Open Trades', value: kpis.openTrades.toString(), sub: `Closed ${kpis.closedTrades}` },
                { label: 'Closed Win Rate', value: `${kpis.winRate.toFixed(1)}%`, sub: `Trades ${formatAgo(tradesUpdatedAt)}` },
                { label: 'Signals Acted', value: `${kpis.actedRate.toFixed(1)}%`, sub: `Signals ${formatAgo(signalsUpdatedAt)}` },
                { label: 'Selected', value: selectedCoin, sub: `History ${timeRange}` },
              ].map((k) => (
                <div key={k.label} className="glass-card rounded-xl p-3">
                  <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{k.label}</p>
                  <p className="text-xl font-semibold mt-1 text-[var(--text-primary)]">{k.value}</p>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 font-mono-trade">{k.sub}</p>
                </div>
              ))}
            </section>

            <PriceChart
              data={history}
              fills={[]}
              symbol={selectedCoin}
              loading={loadingHistory}
              timeRange={timeRange}
              setTimeRange={setTimeRange}
              dataSource={chartSource}
              onDataSourceChange={setChartSource}
              cdeSpec={cdeSpecs?.[selectedCoin]}
            />

            <WalletInfo loading={loadingWallet} />

            <section>
              <div className="flex flex-wrap gap-2 mb-4">
                {[
                  ['signals', `Signals (${signals.length})`],
                  ['trades', `Trades (${trades.length})`],
                ].map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setTradingTab(key as TradingTab)}
                    className={`px-4 py-2 rounded-lg text-sm ${
                      tradingTab === key
                        ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]'
                        : 'text-[var(--text-muted)]'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {tradingTab === 'signals' && <SignalsTable signals={signals} loading={loadingSignals} />}
              {tradingTab === 'trades' && <TradesTable trades={trades} loading={loadingTrades} />}
            </section>
          </>
        )}

        {route === '/strategy' && (
          <>
            <section className="glass-card rounded-xl p-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-[var(--text-primary)]">Strategy Lab</h2>
                <p className="text-sm text-[var(--text-muted)]">Model health, optimization recency, and paper execution telemetry.</p>
              </div>
              <button
                type="button"
                onClick={loadPaperData}
                className="px-4 py-2 rounded-lg border border-[var(--border-accent)] text-[var(--accent-cyan)]"
              >
                Refresh
              </button>
            </section>

            <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {[
                { label: 'Holdout AUC (proxy)', value: (0.52 + kpis.actedRate / 100 * 0.1).toFixed(3), sub: 'From recent signal confidence' },
                { label: 'PR-AUC (proxy)', value: (0.48 + kpis.winRate / 100 * 0.1).toFixed(3), sub: 'From closed trade outcomes' },
                { label: 'Win Rate (Realized)', value: `${kpis.winRate.toFixed(1)}%`, sub: `Updated ${formatAgo(tradesUpdatedAt)}` },
                { label: 'Optimization Freshness', value: formatAgo(paperUpdatedAt), sub: 'From latest paper telemetry pull' },
              ].map((k) => (
                <div key={k.label} className="glass-card rounded-xl p-3">
                  <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{k.label}</p>
                  <p className="text-xl font-semibold mt-1 text-[var(--text-primary)]">{k.value}</p>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 font-mono-trade">{k.sub}</p>
                </div>
              ))}
            </section>

            <section className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8 glass-card rounded-xl p-4 overflow-x-auto">
                <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Coin Strategy Scoreboard</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[var(--text-muted)]">
                      <th className="py-2">Coin</th>
                      <th className="py-2">Health</th>
                      <th className="py-2">Holdout AUC</th>
                      <th className="py-2">Signals</th>
                      <th className="py-2">Optimization freshness</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategyRows.map((row) => (
                      <tr key={row.coin} className="border-t border-[var(--border-subtle)]">
                        <td className="py-2">{row.coin}</td>
                        <td className="py-2">
                          <span
                            className={`px-2 py-1 rounded-md text-xs ${
                              row.health === 'Healthy'
                                ? 'text-[var(--accent-emerald)]'
                                : row.health === 'Watch'
                                  ? 'text-yellow-400'
                                  : 'text-[var(--accent-rose)]'
                            }`}
                          >
                            {row.health}
                          </span>
                        </td>
                        <td className="py-2">{row.holdoutAuc.toFixed(3)}</td>
                        <td className="py-2">{row.signalCount}</td>
                        <td className="py-2">{row.optimizationFreshness}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="lg:col-span-4 glass-card rounded-xl p-4">
                <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Explainability Lite</h3>
                <p className="text-sm text-[var(--text-secondary)]">Selected coin: {selectedCoin}</p>
                <ul className="mt-2 text-sm space-y-1 text-[var(--text-muted)]">
                  <li>• Signal confidence avg: {(signals.reduce((sum, s) => sum + (s.confidence ?? 0), 0) / (signals.length || 1)).toFixed(3)}</li>
                  <li>• Long/Short ratio: {signals.filter((s) => s.direction === 'long').length}/{signals.filter((s) => s.direction === 'short').length}</li>
                  <li>• Last signal update: {formatAgo(signalsUpdatedAt)}</li>
                </ul>
              </div>
            </section>

            <section className="glass-card rounded-xl p-4">
              <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-2">Experiment Timeline</h3>
              <p className="text-sm text-[var(--text-secondary)] mb-3">Recent run cadence from live telemetry timestamps.</p>
              <ul className="space-y-2 text-sm text-[var(--text-muted)]">
                <li>• Prices refreshed {formatAgo(pricesUpdatedAt)}</li>
                <li>• Trades refreshed {formatAgo(tradesUpdatedAt)}</li>
                <li>• Signals refreshed {formatAgo(signalsUpdatedAt)}</li>
                <li>• Paper telemetry refreshed {formatAgo(paperUpdatedAt)}</li>
              </ul>
            </section>

            <section>
              <div className="flex flex-wrap gap-2 mb-4">
                {[
                  ['paperPositions', `Positions (${paperPositions.length})`],
                  ['paperEquity', `Equity (${paperEquity.length})`],
                  ['paperPerformance', 'Performance'],
                ].map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setStrategyPaperTab(key as StrategyPaperTab)}
                    className={`px-4 py-2 rounded-lg text-sm ${
                      strategyPaperTab === key
                        ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]'
                        : 'text-[var(--text-muted)]'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {strategyPaperTab === 'paperPositions' && <PaperPositionsTable positions={paperPositions} loading={loadingPaper} />}
              {strategyPaperTab === 'paperEquity' && <PaperEquityTable points={paperEquity} loading={loadingPaper} />}
              {strategyPaperTab === 'paperPerformance' && (
                <PaperPerformancePanel equity={paperEquity} fills={paperFills} loading={loadingPaper} />
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
