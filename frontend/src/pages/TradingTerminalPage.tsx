import { useCallback, useEffect, useMemo, useState } from 'react';
import { getCDESpecs, getCoinHistory, getCurrentPrices } from '../api/coinsApi';
import { getPaperEquity, getPaperFills, getPaperPositions } from '../api/paperApi';
import { getRecentSignals } from '../api/signalsApi';
import { getAllTrades } from '../api/tradesApi';
import PaperEquityTable from '../components/PaperEquityTable';
import PaperFillsTable from '../components/PaperFillsTable';
import PaperPerformancePanel from '../components/PaperPerformancePanel';
import PaperPositionsTable from '../components/PaperPositionsTable';
import PriceCard from '../components/PriceCard';
import PriceChart from '../components/PriceChart';
import SignalsTable from '../components/SignalsTable';
import TradesTable from '../components/TradesTable';
import WalletInfo from '../components/WalletInfo';
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
} from '../types';

const COINS: CoinSymbol[] = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE'];
const coinIcons: Record<CoinSymbol, string> = { BTC: '₿', ETH: 'Ξ', SOL: '◎', XRP: '✕', DOGE: 'Ð' };

type TradingMode = 'live' | 'paper';
type LiveTab = 'signals' | 'trades';
type PaperTab = 'positions' | 'equity' | 'performance' | 'fills';

const formatAgo = (d: Date | null) => {
  if (!d) return '—';
  const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
};

export default function TradingTerminalPage() {
  const [mode, setMode] = useState<TradingMode>('paper');

  // ── Prices & chart ───────────────────────────────────────────────
  const [prices, setPrices] = useState<PriceData | null>(null);
  const [cdeSpecs, setCdeSpecs] = useState<CDESpecs | null>(null);
  const [selectedCoin, setSelectedCoin] = useState<CoinSymbol>('BTC');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [timeRange, setTimeRange] = useState<'1h' | '1d' | '1w' | '1m' | '1y'>('1d');
  const [loadingPrices, setLoadingPrices] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [pricesUpdatedAt, setPricesUpdatedAt] = useState<Date | null>(null);
  const [cardSources, setCardSources] = useState<Record<CoinSymbol, DataSource>>(
    Object.fromEntries(COINS.map((c) => [c, 'spot'])) as Record<CoinSymbol, DataSource>,
  );
  const [chartSource, setChartSource] = useState<DataSource>('spot');

  // ── Live mode ────────────────────────────────────────────────────
  const [trades, setTrades] = useState<Trade[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loadingTrades, setLoadingTrades] = useState(true);
  const [loadingSignals, setLoadingSignals] = useState(true);
  const [signalsUpdatedAt, setSignalsUpdatedAt] = useState<Date | null>(null);
  const [tradesUpdatedAt, setTradesUpdatedAt] = useState<Date | null>(null);
  const [liveTab, setLiveTab] = useState<LiveTab>('signals');

  // ── Paper mode ───────────────────────────────────────────────────
  const [paperPositions, setPaperPositions] = useState<PaperPosition[]>([]);
  const [paperEquity, setPaperEquity] = useState<PaperEquityPoint[]>([]);
  const [paperFills, setPaperFills] = useState<PaperFill[]>([]);
  const [loadingPaper, setLoadingPaper] = useState(true);
  const [paperUpdatedAt, setPaperUpdatedAt] = useState<Date | null>(null);
  const [paperTab, setPaperTab] = useState<PaperTab>('positions');

  const [error, setError] = useState<string | null>(null);

  // ── Data fetching ────────────────────────────────────────────────
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
      setLoadingSignals(true);
      try {
        setSignals(await getRecentSignals(200));
        setSignalsUpdatedAt(new Date());
      } finally {
        setLoadingSignals(false);
      }
    };
    load();
    const iv = setInterval(load, 15000);
    return () => clearInterval(iv);
  }, []);

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

  const loadPaper = useCallback(async () => {
    setLoadingPaper(true);
    try {
      const [positions, equity, fills] = await Promise.all([
        getPaperPositions(),
        getPaperEquity(250),
        getPaperFills(250),
      ]);
      setPaperPositions(positions);
      setPaperEquity(equity);
      setPaperFills(fills);
      setPaperUpdatedAt(new Date());
    } finally {
      setLoadingPaper(false);
    }
  }, []);

  useEffect(() => {
    loadPaper();
    const iv = setInterval(loadPaper, 20000);
    return () => clearInterval(iv);
  }, [loadPaper]);

  // ── Derived KPIs ─────────────────────────────────────────────────
  const liveKpis = useMemo(() => {
    const openTrades = trades.filter((t) => t.status === 'open').length;
    const closedTrades = trades.filter((t) => t.status === 'closed');
    const winners = closedTrades.filter((t) => (t.net_pnl ?? 0) > 0).length;
    const winRate = closedTrades.length > 0 ? (winners / closedTrades.length) * 100 : 0;
    const actedSignals = signals.filter((s) => s.acted_on).length;
    const actedRate = signals.length > 0 ? (actedSignals / signals.length) * 100 : 0;
    const realizedPnl = closedTrades.reduce((sum, t) => sum + (t.net_pnl ?? 0), 0);
    return { openTrades, closedTrades: closedTrades.length, winRate, actedRate, realizedPnl };
  }, [trades, signals]);

  const paperKpis = useMemo(() => {
    const latest = paperEquity[0];
    const first = paperEquity[paperEquity.length - 1];
    const returnPct = latest && first && first.equity > 0 ? ((latest.equity - first.equity) / first.equity) * 100 : 0;
    const openPositions = paperPositions.length;
    const totalFees = paperFills.reduce((s, f) => s + f.fee, 0);
    const realizedPnl = latest?.realized_pnl ?? 0;
    return { equity: latest?.equity ?? null, returnPct, openPositions, totalFees, realizedPnl };
  }, [paperEquity, paperFills, paperPositions]);

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[var(--border-subtle)] bg-[var(--bg-primary)]/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-5 py-4 flex items-center justify-between gap-4">
          <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">Trading Terminal</h1>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono-trade text-[var(--text-muted)]">Prices: {formatAgo(pricesUpdatedAt)}</span>
            {/* Paper / Live toggle */}
            <div className="flex rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-secondary)] p-0.5 text-xs">
              {(['paper', 'live'] as TradingMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-4 py-1.5 rounded capitalize font-semibold transition-colors ${mode === m ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)]' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}`}
                >
                  {m === 'paper' ? '📋 Paper' : '⚡ Live'}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-5 py-6 space-y-6">
        {error && <div className="glass-card rounded-xl p-3 text-sm text-[var(--accent-rose)]">{error}</div>}

        {/* Price cards — always visible */}
        <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
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
        </section>

        {/* KPI strip — switches between paper and live */}
        {mode === 'live' ? (
          <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { label: 'Open Trades', value: liveKpis.openTrades.toString(), sub: `Closed ${liveKpis.closedTrades}` },
              { label: 'Win Rate', value: `${liveKpis.winRate.toFixed(1)}%`, sub: `Updated ${formatAgo(tradesUpdatedAt)}` },
              { label: 'Signal Act Rate', value: `${liveKpis.actedRate.toFixed(1)}%`, sub: `Signals ${formatAgo(signalsUpdatedAt)}` },
              { label: 'Realized PNL', value: `${liveKpis.realizedPnl >= 0 ? '+' : '-'}$${Math.abs(liveKpis.realizedPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, sub: 'Closed trades only', valueClass: liveKpis.realizedPnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]' },
            ].map((k) => (
              <div key={k.label} className="glass-card rounded-xl p-3 text-center">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{k.label}</p>
                <p className={`text-xl font-semibold mt-1 ${(k as { valueClass?: string }).valueClass ?? 'text-[var(--text-primary)]'}`}>{k.value}</p>
                <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 font-mono-trade">{k.sub}</p>
              </div>
            ))}
          </section>
        ) : (
          <section className="grid grid-cols-2 lg:grid-cols-5 gap-3">
            {[
              { label: 'Paper Equity', value: paperKpis.equity != null ? `$${paperKpis.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—', valueClass: 'text-[var(--text-primary)]' },
              { label: 'Return', value: `${paperKpis.returnPct >= 0 ? '+' : ''}${paperKpis.returnPct.toFixed(2)}%`, valueClass: paperKpis.returnPct >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]' },
              { label: 'Realized PNL', value: `${paperKpis.realizedPnl >= 0 ? '+' : ''}$${Math.abs(paperKpis.realizedPnl).toFixed(2)}`, valueClass: paperKpis.realizedPnl >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]' },
              { label: 'Open Positions', value: paperKpis.openPositions.toString(), valueClass: 'text-[var(--text-primary)]' },
              { label: 'Total Fees', value: `$${paperKpis.totalFees.toFixed(4)}`, valueClass: 'text-[var(--accent-amber)]' },
            ].map((k) => (
              <div key={k.label} className="glass-card rounded-xl p-3 text-center">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-mono-trade">{k.label}</p>
                <p className={`text-xl font-semibold mt-1 ${k.valueClass}`}>{k.value}</p>
                <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 font-mono-trade">Updated {formatAgo(paperUpdatedAt)}</p>
              </div>
            ))}
          </section>
        )}

        {/* Chart — always visible */}
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

        {/* Bottom section — switches between paper and live */}
        {mode === 'live' ? (
          <>
            <WalletInfo loading={false} showPaperMetrics={false} showExternalMetrics showHoldingsBreakdown chartMode="portfolio" />
            <section>
              <div className="flex flex-wrap gap-2 mb-4">
                {[
                  ['signals', `Signals (${signals.length})`],
                  ['trades', `Trades (${trades.length})`],
                ].map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setLiveTab(key as LiveTab)}
                    className={`px-4 py-2 rounded-lg text-sm ${liveTab === key ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {liveTab === 'signals' && <SignalsTable signals={signals} loading={loadingSignals} />}
              {liveTab === 'trades' && <TradesTable trades={trades} loading={loadingTrades} />}
            </section>
          </>
        ) : (
          <section className="glass-card rounded-xl p-4 space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold">Paper Trading</h2>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">Simulated fills from live signals — no real money.</p>
              </div>
              <div className="flex items-center gap-3 text-xs text-[var(--text-muted)] font-mono-trade">
                {paperFills.length > 0 && <span>Last fill: {formatAgo(new Date(paperFills[0].created_at))}</span>}
                <button onClick={loadPaper} className="px-2 py-1 rounded border border-[var(--border-subtle)] hover:border-[var(--border-accent)] hover:text-[var(--accent-cyan)] transition-colors">Refresh</button>
              </div>
            </div>

            <WalletInfo loading={false} showPaperMetrics showExternalMetrics={false} showHoldingsBreakdown={false} chartMode="paper" />

            <div className="flex flex-wrap gap-2 overflow-x-auto">
              {([
                ['positions', `Positions (${paperPositions.length})`],
                ['equity', `Equity (${paperEquity.length})`],
                ['performance', 'Performance'],
                ['fills', `Fills (${paperFills.length})`],
              ] as [PaperTab, string][]).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setPaperTab(key)}
                  className={`px-4 py-2 rounded-lg text-sm ${paperTab === key ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] border border-[var(--border-accent)]' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}`}
                >
                  {label}
                </button>
              ))}
            </div>

            {paperTab === 'positions' && <PaperPositionsTable positions={paperPositions} loading={loadingPaper} />}
            {paperTab === 'equity' && <PaperEquityTable points={paperEquity} loading={loadingPaper} />}
            {paperTab === 'performance' && <PaperPerformancePanel equity={paperEquity} fills={paperFills} loading={loadingPaper} />}
            {paperTab === 'fills' && <PaperFillsTable fills={paperFills} loading={loadingPaper} />}

            {/* Signals always visible in paper mode for monitoring */}
            <div>
              <h3 className="text-xs font-semibold uppercase text-[var(--text-muted)] mb-2">Signal Feed</h3>
              <SignalsTable signals={signals} loading={loadingSignals} />
            </div>
          </section>
        )}
      </main>
    </>
  );
}
