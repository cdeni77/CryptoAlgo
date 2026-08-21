import { useMemo, useState } from 'react';

import { getCDEPrices, getCDESpecs, getCurrentPrices } from '../api/coinsApi';
import {
  getModelStatus,
  getPaperConfig,
  getPaperEquity,
  getPaperFills,
  getPaperPositions,
  getPaperSummary,
} from '../api/paperApi';
import { getRecentSignals } from '../api/signalsApi';
import EquityChart from '../components/EquityChart';
import ModelStatusPanel from '../components/ModelStatusPanel';
import PaperFillsTable from '../components/PaperFillsTable';
import PaperPositionsTable from '../components/PaperPositionsTable';
import PriceCard from '../components/PriceCard';
import SignalsTable from '../components/SignalsTable';
import { Empty, ErrorBlock, Freshness, Panel } from '../components/StateBlock';
import { usePolling } from '../hooks/usePolling';
import { PaperEquityPoint } from '../types';

const ALL_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'ADA', 'LINK', 'LTC'] as const;

/**
 * Fallback contract sizes, used only until `/coins/cde-specs` answers.
 *
 * This table used to be the *only* source, with a comment reading "Must match
 * trading_costs.py" — a file that has since been deleted, so nothing was
 * checking the claim. Contract size multiplies straight into unrealised PnL: a
 * wrong entry here misreports the position's value by that factor, silently.
 * The API now serves the real specs, and these are the last resort.
 */
const FALLBACK_UNITS: Record<string, number> = {
  BTC: 0.01, ETH: 0.1, SOL: 5, XRP: 500, DOGE: 5000,
  AVAX: 10, ADA: 1000, LINK: 50, LTC: 5,
};

const STARTING_BALANCE = 100_000;

const money = (v: number, prefix = '$') =>
  `${prefix}${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="glass-card rounded-xl p-5">
      <div className="mb-2 text-[11px] font-medium uppercase tracking-widest text-tx-muted">
        {label}
      </div>
      <div className={`mb-1 font-mono text-xl font-semibold tabular-nums ${color ?? 'text-tx-primary'}`}>
        {value}
      </div>
      {sub && <div className="font-mono text-xs text-tx-muted">{sub}</div>}
    </div>
  );
}

export default function DashboardPage() {
  // Each source polls at the rate its data actually changes, and all of them
  // stop when the tab is hidden. Previously one 5-second interval refetched
  // everything — including the wallet, which calls Coinbase — forever.
  const summary = usePolling(getPaperSummary, 10_000);
  const equity = usePolling(() => getPaperEquity(300), 15_000);
  const positions = usePolling(getPaperPositions, 10_000);
  const fills = usePolling(() => getPaperFills(50), 20_000);
  const signals = usePolling(() => getRecentSignals(30), 20_000);
  const spot = usePolling(getCurrentPrices, 5_000);
  const cde = usePolling(getCDEPrices, 5_000);
  const specs = usePolling(getCDESpecs, 600_000);
  const config = usePolling(getPaperConfig, 60_000);
  const modelStatus = usePolling(getModelStatus, 30_000);

  const [priceSource, setPriceSource] = useState<'spot' | 'cde'>('spot');
  const prices = priceSource === 'cde' ? cde.data : spot.data;
  const priceState = priceSource === 'cde' ? cde : spot;

  const unitsFor = useMemo(() => {
    const contracts = specs.data?.contracts ?? {};
    return (coin: string): number => {
      const key = coin.toUpperCase();
      const fromApi = contracts[key]?.units_per_contract;
      return typeof fromApi === 'number' && fromApi > 0 ? fromApi : (FALLBACK_UNITS[key] ?? 1);
    };
  }, [specs.data]);

  const openPositions = useMemo(
    () => (positions.data ?? []).filter((p) => p.is_open),
    [positions.data],
  );

  /** Unrealised PnL marked to the live price rather than to the last DB write. */
  const liveUnrealized = useMemo(() => {
    if (!openPositions.length || !prices) return null;
    return openPositions.reduce((sum, p) => {
      const px = prices[p.coin as keyof typeof prices]?.price;
      if (!px) return sum + p.unrealized_pnl;
      const sign = p.side === 'long' ? 1 : -1;
      return sum + p.contracts * unitsFor(p.coin) * (px - p.entry_price) * sign;
    }, 0);
  }, [openPositions, prices, unitsFor]);

  /** The stored curve with a live-priced point prepended, so the chart's right
   *  edge matches the number in the stat card above it. */
  const equityWithLive = useMemo(() => {
    const points = equity.data ?? [];
    if (!points.length || liveUnrealized === null) return points;
    const latest = points[0];
    // Base off the summary's equity, not the last curve point: a restarted engine
    // rewrites cash_balance, and the curve keeps the pre-restart value.
    const base =
      summary.data != null
        ? summary.data.equity - (summary.data.unrealized_pnl ?? 0)
        : latest.equity - latest.unrealized_pnl;
    const livePoint: PaperEquityPoint = {
      ...latest,
      id: -1,
      timestamp: new Date().toISOString(),
      equity: base + liveUnrealized,
      unrealized_pnl: liveUnrealized,
    };
    return [livePoint, ...points];
  }, [equity.data, liveUnrealized, summary.data]);

  const totalReturn = summary.data?.total_return_pct ?? 0;
  const cash =
    summary.data != null
      ? summary.data.equity - (summary.data.unrealized_pnl ?? 0)
      : STARTING_BALANCE;
  const unrealized = liveUnrealized ?? summary.data?.unrealized_pnl ?? 0;
  const portfolio = cash + unrealized;

  const activeCoins = useMemo(
    () => new Set((config.data?.active_coins ?? []).map((c) => c.toUpperCase())),
    [config.data],
  );
  const shownSignals = useMemo(() => {
    const all = signals.data ?? [];
    return activeCoins.size ? all.filter((s) => activeCoins.has(s.coin.toUpperCase())) : all;
  }, [signals.data, activeCoins]);

  return (
    <div className="max-w-[1600px] space-y-5 p-6">
      {/* A failure on the summary is worth a banner: every number below is
          derived from it, so a stale one is a stale screen. */}
      {summary.error && (
        <ErrorBlock error={summary.error} onRetry={summary.refresh} compact />
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Portfolio value"
          value={summary.data ? money(portfolio) : '—'}
          sub={
            summary.data?.initial_equity != null
              ? `started at ${money(summary.data.initial_equity)}`
              : 'starting balance not reported'
          }
        />
        <StatCard
          label="Total return"
          value={summary.data ? `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%` : '—'}
          sub={
            summary.data?.realized_pnl != null
              ? `${money(summary.data.realized_pnl)} realised`
              : 'realised P&L not measured'
          }
          color={
            summary.data?.total_return_pct == null
              ? undefined
              : totalReturn >= 0 ? 'text-accent-emerald' : 'text-accent-rose'
          }
        />
        <StatCard
          label="Unrealised P&L"
          value={summary.data ? `${unrealized >= 0 ? '+' : ''}${money(unrealized)}` : '—'}
          sub={
            summary.data
              ? `${summary.data.open_positions} open${
                  liveUnrealized !== null ? ' · marked live' : ''
                }`
              : 'not measured'
          }
          color={
            !summary.data ? undefined
              : unrealized >= 0 ? 'text-accent-emerald' : 'text-accent-rose'
          }
        />
        <StatCard
          label="Win rate"
          value={
            summary.data?.win_rate != null ? `${(summary.data.win_rate * 100).toFixed(1)}%` : '—'
          }
          sub={summary.data ? `${summary.data.fill_count} fills` : 'not measured'}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="glass-card rounded-xl p-5 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-widest text-tx-secondary">
              Equity curve
            </span>
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-tx-muted">
                {equityWithLive.length} points
              </span>
              <Freshness
                lastUpdated={equity.lastUpdated}
                refreshing={equity.refreshing}
                error={equity.error}
              />
            </div>
          </div>
          <div className="h-56">
            <Panel
              state={equity}
              emptyWhen={() => equityWithLive.length === 0}
              emptyMessage="No equity history yet."
              emptyHint="The paper engine writes a point on every fill and mark."
              loadingLabel="Loading the curve"
            >
              {() => (
                <EquityChart equity={equityWithLive} startingBalance={STARTING_BALANCE} />
              )}
            </Panel>
          </div>
        </div>

        <div className="glass-card rounded-xl p-5">
          <div className="mb-4 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-widest text-tx-secondary">
              Open positions
            </span>
            <Freshness
              lastUpdated={positions.lastUpdated}
              refreshing={positions.refreshing}
              error={positions.error}
            />
          </div>
          <Panel
            state={positions}
            emptyWhen={(rows) => rows.filter((p) => p.is_open).length === 0}
            emptyMessage="Nothing open."
            loadingLabel="Loading positions"
          >
            {(rows) => (
              <PaperPositionsTable
                positions={rows}
                prices={prices ?? null}
                unitsFor={specs.data ? unitsFor : undefined}
              />
            )}
          </Panel>
        </div>
      </div>

      <div>
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[11px] font-medium uppercase tracking-widest text-tx-muted">
            Live prices
          </div>
          <div className="flex items-center gap-3">
            <Freshness
              lastUpdated={priceState.lastUpdated}
              refreshing={priceState.refreshing}
              error={priceState.error}
            />
            <div className="flex gap-0.5 rounded border border-[rgba(56,189,248,0.08)] bg-[rgba(56,189,248,0.05)] p-0.5">
              {(['spot', 'cde'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setPriceSource(s)}
                  className={`rounded px-2.5 py-0.5 font-mono text-[10px] transition-all ${
                    s === priceSource
                      ? 'bg-accent-cyan/15 text-accent-cyan'
                      : 'text-tx-muted hover:text-tx-secondary'
                  }`}
                >
                  {s.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        {priceState.error && (
          <div className="mb-3">
            <ErrorBlock error={priceState.error} onRetry={priceState.refresh} compact />
          </div>
        )}

        <div className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-9">
          {ALL_COINS.map((coin) => (
            <PriceCard
              key={coin}
              coin={coin}
              price={prices?.[coin as keyof typeof prices]?.price ?? null}
              change24h={prices?.[coin as keyof typeof prices]?.change24h ?? null}
            />
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="glass-card rounded-xl p-5">
          <div className="mb-4 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-widest text-tx-secondary">
              Recent signals
            </span>
            <Freshness
              lastUpdated={signals.lastUpdated}
              refreshing={signals.refreshing}
              error={signals.error}
            />
          </div>
          <Panel
            state={signals}
            emptyWhen={() => shownSignals.length === 0}
            emptyMessage={
              activeCoins.size && (signals.data?.length ?? 0) > 0
                ? 'No signals on the active instruments.'
                : 'No signals yet.'
            }
            emptyHint="Run scripts.signals after building the feature panel."
            loadingLabel="Loading signals"
          >
            {() => <SignalsTable signals={shownSignals} limit={15} compact />}
          </Panel>
        </div>

        <div className="glass-card rounded-xl p-5">
          <div className="mb-4 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-widest text-tx-secondary">
              Recent fills
            </span>
            <Freshness
              lastUpdated={fills.lastUpdated}
              refreshing={fills.refreshing}
              error={fills.error}
            />
          </div>
          <Panel
            state={fills}
            emptyWhen={(rows) => rows.length === 0}
            emptyMessage="No fills yet."
            loadingLabel="Loading fills"
          >
            {(rows) => <PaperFillsTable fills={rows} limit={15} />}
          </Panel>
        </div>

        {modelStatus.error ? (
          <div className="glass-card rounded-xl p-5">
            <div className="mb-4 text-xs font-medium uppercase tracking-widest text-tx-secondary">
              Model status
            </div>
            <ErrorBlock error={modelStatus.error} onRetry={modelStatus.refresh} compact />
          </div>
        ) : modelStatus.data ? (
          <ModelStatusPanel data={modelStatus.data} />
        ) : (
          <div className="glass-card rounded-xl p-5">
            <div className="mb-4 text-xs font-medium uppercase tracking-widest text-tx-secondary">
              Model status
            </div>
            <Empty message="No model status." hint="See the Model page for the gates." />
          </div>
        )}
      </div>
    </div>
  );
}
