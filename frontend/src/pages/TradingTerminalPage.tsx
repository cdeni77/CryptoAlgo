import { useEffect, useMemo, useState } from 'react';
import { getCDESpecs, getCoinHistory, getCurrentPrices } from '../api/coinsApi';
import { getRecentSignals } from '../api/signalsApi';
import { getAllTrades } from '../api/tradesApi';
import PriceCard from '../components/PriceCard';
import PriceChart from '../components/PriceChart';
import SignalsTable from '../components/SignalsTable';
import TradesTable from '../components/TradesTable';
import WalletInfo from '../components/WalletInfo';
import { CDESpecs, CoinSymbol, DataSource, HistoryEntry, PriceData, Signal, Trade } from '../types';

const COINS: CoinSymbol[] = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE'];
const coinIcons: Record<CoinSymbol, string> = { BTC: '₿', ETH: 'Ξ', SOL: '◎', XRP: '✕', DOGE: 'Ð' };

type BottomTab = 'signals' | 'trades';

const formatAgo = (d: Date | null) => {
  if (!d) return '—';
  const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
};

export default function TradingTerminalPage() {
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
  const [loadingWallet] = useState(false);
  const [bottomTab, setBottomTab] = useState<BottomTab>('signals');
  const [pricesUpdatedAt, setPricesUpdatedAt] = useState<Date | null>(null);
  const [signalsUpdatedAt, setSignalsUpdatedAt] = useState<Date | null>(null);
  const [tradesUpdatedAt, setTradesUpdatedAt] = useState<Date | null>(null);

  const [cardSources, setCardSources] = useState<Record<CoinSymbol, DataSource>>(
    Object.fromEntries(COINS.map((c) => [c, 'spot'])) as Record<CoinSymbol, DataSource>,
  );
  const [chartSource, setChartSource] = useState<DataSource>('spot');

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

  const kpis = useMemo(() => {
    const openTrades = trades.filter((t) => t.status === 'open').length;
    const closedTrades = trades.filter((t) => t.status === 'closed');
    const winners = closedTrades.filter((t) => (t.net_pnl ?? 0) > 0).length;
    const winRate = closedTrades.length > 0 ? (winners / closedTrades.length) * 100 : 0;

    const actedSignals = signals.filter((s) => s.acted_on).length;
    const actedRate = signals.length > 0 ? (actedSignals / signals.length) * 100 : 0;
    const realizedPnl = closedTrades.reduce((sum, trade) => sum + (trade.net_pnl ?? 0), 0);

    return { openTrades, closedTrades: closedTrades.length, winRate, actedRate, realizedPnl };
  }, [trades, signals]);

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[var(--border-subtle)] bg-[var(--bg-primary)]/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-5 py-4 flex items-center justify-between">
          <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">Trading Terminal</h1>
          <span className="text-xs font-mono-trade text-[var(--text-muted)]">Prices: {formatAgo(pricesUpdatedAt)}</span>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-5 py-6 space-y-6">
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
            {
              label: 'Realized PNL',
              value: `${kpis.realizedPnl >= 0 ? '+' : '-'}$${Math.abs(kpis.realizedPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
              sub: 'Closed trades only',
              valueClass: kpis.realizedPnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]',
            },
          ].map((k) => (
            <div key={k.label} className="glass-card rounded-xl p-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{k.label}</p>
              <p className={`text-xl font-semibold mt-1 ${k.valueClass ?? 'text-[var(--text-primary)]'}`}>{k.value}</p>
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

        <WalletInfo loading={loadingWallet} showPaperMetrics={false} showExternalMetrics showHoldingsBreakdown chartMode="portfolio" />

        <section>
          <div className="flex flex-wrap gap-2 mb-4">
            {[
              ['signals', `Signals (${signals.length})`],
              ['trades', `Trades (${trades.length})`],
            ].map(([key, label]) => (
              <button
                key={key}
                onClick={() => setBottomTab(key as BottomTab)}
                className={`px-4 py-2 rounded-lg text-sm ${
                  bottomTab === key
                    ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]'
                    : 'text-[var(--text-muted)]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {bottomTab === 'signals' && <SignalsTable signals={signals} loading={loadingSignals} />}
          {bottomTab === 'trades' && <TradesTable trades={trades} loading={loadingTrades} />}
        </section>
      </main>
    </>
  );
}
