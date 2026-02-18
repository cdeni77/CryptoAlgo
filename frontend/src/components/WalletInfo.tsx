import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getCoinHistory } from '../api/coinsApi';
import { getWallet } from '../api/walletApi';
import { CoinSymbol, WalletData } from '../types';

type PortfolioRange = '1h' | '1d' | '1w' | '1m' | '1y';
type ChartMode = 'portfolio' | 'paper';

interface WalletInfoProps {
  loading: boolean;
  showPaperMetrics?: boolean;
  showExternalMetrics?: boolean;
  showHoldingsBreakdown?: boolean;
  chartMode?: ChartMode;
}

type PortfolioPoint = {
  timestamp: string;
  paper_equity_usd: number;
  external_usd: number;
  total_value_usd: number;
};

const RANGE_LABELS: Record<PortfolioRange, string> = {
  '1h': '1H',
  '1d': '24H',
  '1w': '7D',
  '1m': '30D',
  '1y': '1Y',
};

export default function WalletInfo({
  loading,
  showPaperMetrics = true,
  showExternalMetrics = true,
  showHoldingsBreakdown = true,
  chartMode = 'portfolio',
}: WalletInfoProps) {
  const [wallet, setWallet] = useState<WalletData | null>(null);
  const [portfolioRange, setPortfolioRange] = useState<PortfolioRange>('1d');
  const [backfilledSeries, setBackfilledSeries] = useState<PortfolioPoint[]>([]);

  useEffect(() => {
    const loadWallet = async () => {
      try {
        const data = await getWallet();
        setWallet(data);
      } catch (err) {
        console.error('Wallet fetch error:', err);
      }
    };

    loadWallet();
    const interval = setInterval(loadWallet, 5000);
    return () => clearInterval(interval);
  }, []);

  const fmt = (v: number) =>
    `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const pnlColor = (v: number) =>
    v >= 0 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]';

  const portfolioSeries = useMemo(() => {
    if (!wallet) return [];
    const seriesFromRange = wallet.portfolio_history_by_range?.[portfolioRange];
    return (seriesFromRange?.length ? seriesFromRange : wallet.portfolio_history ?? []) as PortfolioPoint[];
  }, [wallet, portfolioRange]);

  const shouldBackfill = useMemo(
    () =>
      chartMode === 'portfolio' &&
      portfolioSeries.length > 0 &&
      portfolioSeries.every((point) => (point.total_value_usd ?? 0) === 0),
    [chartMode, portfolioSeries],
  );

  useEffect(() => {
    const supportedCoins = new Set<CoinSymbol>(['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']);

    const normalizeSymbol = (value?: string | null): CoinSymbol | null => {
      if (!value) return null;
      const normalized = value.toUpperCase().replace(/[^A-Z]/g, '');
      if (supportedCoins.has(normalized as CoinSymbol)) {
        return normalized as CoinSymbol;
      }
      return null;
    };

    const buildBackfill = async () => {
      if (!wallet || !shouldBackfill) {
        setBackfilledSeries([]);
        return;
      }

      const holdings = new Map<CoinSymbol, number>();

      for (const asset of wallet.coinbase?.spot?.assets ?? []) {
        const symbol = normalizeSymbol(asset.asset);
        if (!symbol || !Number.isFinite(asset.amount) || asset.amount <= 0) continue;
        holdings.set(symbol, (holdings.get(symbol) ?? 0) + asset.amount);
      }

      for (const asset of wallet.ledger?.assets ?? []) {
        const symbol = normalizeSymbol(asset.asset);
        if (!symbol || !Number.isFinite(asset.amount) || asset.amount <= 0) continue;
        holdings.set(symbol, (holdings.get(symbol) ?? 0) + asset.amount);
      }

      if (!holdings.size) {
        setBackfilledSeries([]);
        return;
      }

      try {
        const histories = await Promise.all(
          Array.from(holdings.entries()).map(async ([symbol, amount]) => ({
            amount,
            history: await getCoinHistory(symbol, portfolioRange),
          })),
        );

        const mergedSeries = new Map<string, { timestamp: string; trackedValueUsd: number }>();

        for (const { amount, history } of histories) {
          for (const candle of history) {
            const existing = mergedSeries.get(candle.timestamp);
            const trackedValueUsd = amount * candle.close;
            mergedSeries.set(candle.timestamp, {
              timestamp: candle.timestamp,
              trackedValueUsd: (existing?.trackedValueUsd ?? 0) + trackedValueUsd,
            });
          }
        }

        const sortedTrackedSeries = Array.from(mergedSeries.values()).sort(
          (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
        );

        if (!sortedTrackedSeries.length) {
          setBackfilledSeries([]);
          return;
        }

        const spotTotal = wallet.coinbase?.spot?.value_usd ?? 0;
        const ledgerTotal = wallet.ledger?.value_usd ?? wallet.wallets?.ledger?.value_usd ?? 0;
        const perpsTotal = wallet.coinbase?.perps?.value_usd ?? 0;
        const externalTotalNow = spotTotal + ledgerTotal + perpsTotal;
        const trackedTailValue = sortedTrackedSeries[sortedTrackedSeries.length - 1].trackedValueUsd;
        const externalOffset = Math.max(0, externalTotalNow - trackedTailValue);

        const nextSeries: PortfolioPoint[] = sortedTrackedSeries.map((point) => {
          const externalUsd = point.trackedValueUsd + externalOffset;
          return {
            timestamp: point.timestamp,
            paper_equity_usd: wallet.wallets?.paper_trading?.value_usd ?? wallet.balance,
            external_usd: externalUsd,
            total_value_usd: externalUsd,
          };
        });

        setBackfilledSeries(nextSeries);
      } catch (error) {
        console.error('Failed to backfill portfolio chart from holdings:', error);
        setBackfilledSeries([]);
      }
    };

    buildBackfill();
  }, [wallet, portfolioRange, shouldBackfill]);

  const rawChartSeries = shouldBackfill && backfilledSeries.length ? backfilledSeries : portfolioSeries;
  const activeSeriesKey = chartMode === 'paper' ? 'paper_equity_usd' : 'total_value_usd';

  const chartSeries = useMemo(() => {
    const filteredPositive = rawChartSeries.filter((point) => (point[activeSeriesKey] ?? 0) > 0);
    return filteredPositive.length >= 2 ? filteredPositive : rawChartSeries;
  }, [rawChartSeries, activeSeriesKey]);

  const chartBounds = useMemo(() => {
    if (!chartSeries.length) return { min: 0, max: 0, pad: 1 };
    const values = chartSeries.map((point) => point[activeSeriesKey] ?? 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(1, max - min);
    return {
      min,
      max,
      pad: span * 0.08,
    };
  }, [chartSeries, activeSeriesKey]);

  if (loading || !wallet) {
    return (
      <div className="glass-card rounded-xl p-10 flex items-center justify-center h-40">
        <div className="w-10 h-10 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const items = [
    ...(showPaperMetrics
      ? [
          {
            label: 'Paper Trading',
            value:
              wallet.wallets?.paper_trading?.value_usd != null
                ? fmt(wallet.wallets.paper_trading.value_usd)
                : fmt(wallet.balance),
            color: 'text-[var(--text-primary)]',
          },
          {
            label: 'Realized PNL',
            value: `${wallet.realized_pnl >= 0 ? '+' : ''}${fmt(wallet.realized_pnl)}`,
            color: pnlColor(wallet.realized_pnl),
          },
          {
            label: 'Unrealized PNL',
            value: `${wallet.unrealized_pnl >= 0 ? '+' : ''}${fmt(wallet.unrealized_pnl)}`,
            color: pnlColor(wallet.unrealized_pnl),
          },
          {
            label: 'Total PNL',
            value: `${wallet.total_pnl >= 0 ? '+' : ''}${fmt(wallet.total_pnl)}`,
            color: pnlColor(wallet.total_pnl),
          },
        ]
      : []),
    ...(showExternalMetrics
      ? [
          {
            label: 'Coinbase Spot',
            value: wallet.coinbase?.spot?.value_usd != null ? fmt(wallet.coinbase.spot.value_usd) : 'N/A',
            color: 'text-[var(--text-primary)]',
          },
          {
            label: 'Coinbase Perps',
            value: wallet.coinbase?.perps?.value_usd != null ? fmt(wallet.coinbase.perps.value_usd) : 'N/A',
            color: 'text-[var(--text-primary)]',
          },
          {
            label: 'Ledger Wallet',
            value:
              wallet.wallets?.ledger?.value_usd != null
                ? fmt(wallet.wallets.ledger.value_usd)
                : wallet.wallets?.ledger?.address_count
                  ? '$0.00'
                  : 'Not configured',
            color: 'text-[var(--text-primary)]',
          },
          {
            label: 'Portfolio Total',
            value: wallet.coinbase?.total_value_usd != null ? fmt(wallet.coinbase.total_value_usd) : 'N/A',
            color: 'text-[var(--accent-cyan)]',
          },
        ]
      : []),
  ];

  return (
    <div className="glass-card rounded-xl p-5 space-y-4">
      <div className={`grid gap-4 ${items.length <= 4 ? 'grid-cols-2 md:grid-cols-4' : 'grid-cols-2 md:grid-cols-4 lg:grid-cols-8'}`}>
        {items.map(({ label, value, color }) => (
          <div key={label} className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
            <p className="text-[10px] font-mono-trade text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
              {label}
            </p>
            <p className={`text-xl font-bold font-mono-trade ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      {showHoldingsBreakdown && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <details className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
            <summary className="cursor-pointer text-sm font-semibold text-[var(--text-primary)]">
              Coinbase Spot Holdings
            </summary>
            <div className="mt-3 space-y-2 max-h-64 overflow-auto">
              {wallet.coinbase?.spot?.assets?.length ? (
                wallet.coinbase.spot.assets.map((asset) => (
                  <div key={asset.asset} className="text-xs flex items-center justify-between gap-3">
                    <span className="text-[var(--text-muted)]">{asset.asset}</span>
                    <span className="text-[var(--text-primary)]">
                      {asset.amount.toLocaleString()} ({fmt(asset.value_usd)})
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-xs text-[var(--text-muted)]">No spot holdings available.</p>
              )}
            </div>
          </details>

          <details className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
            <summary className="cursor-pointer text-sm font-semibold text-[var(--text-primary)]">
              Coinbase Perps Positions
            </summary>
            <div className="mt-3 space-y-2 max-h-64 overflow-auto">
              {wallet.coinbase?.perps?.positions?.length ? (
                wallet.coinbase.perps.positions.map((position) => (
                  <div key={position.symbol} className="text-xs flex items-center justify-between gap-3">
                    <span className="text-[var(--text-muted)]">{position.symbol}</span>
                    <span className="text-[var(--text-primary)]">
                      {position.contracts ?? 0} ({position.notional_usd != null ? fmt(position.notional_usd) : 'N/A'})
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-xs text-[var(--text-muted)]">No perps positions available.</p>
              )}
            </div>
          </details>

          <details className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
            <summary className="cursor-pointer text-sm font-semibold text-[var(--text-primary)]">
              Ledger Wallet Holdings
            </summary>
            <div className="mt-3 space-y-2 max-h-64 overflow-auto">
              {wallet.ledger?.entries?.length ? (
                wallet.ledger.entries.map((entry) => (
                  <div key={`${entry.coin}-${entry.address}`} className="text-xs space-y-0.5">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-[var(--text-muted)]">{entry.coin}</span>
                      <span className="text-[var(--text-primary)]">
                        {entry.amount != null ? entry.amount.toLocaleString() : 'N/A'}
                        {entry.value_usd != null ? ` (${fmt(entry.value_usd)})` : ''}
                      </span>
                    </div>
                    <div className="text-[10px] text-[var(--text-muted)] break-all">{entry.address}</div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-[var(--text-muted)]">No ledger addresses configured.</p>
              )}
            </div>
          </details>
        </div>
      )}

      <div className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)] h-72">
        <div className="mb-3 flex items-center justify-between gap-2">
          <p className="text-sm font-semibold text-[var(--text-primary)]">
            {chartMode === 'paper' ? 'Paper Equity Trend' : 'Total Portfolio Trend'}
          </p>
          <div className="flex gap-1 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-primary)] p-1">
            {(['1h', '1d', '1w', '1m', '1y'] as const).map((range) => (
              <button
                key={range}
                type="button"
                onClick={() => setPortfolioRange(range)}
                className={`px-2.5 py-1 text-xs rounded ${
                  portfolioRange === range
                    ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                {RANGE_LABELS[range]}
              </button>
            ))}
          </div>
        </div>
        {chartSeries.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartSeries} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickFormatter={(v: string) => {
                  const dt = new Date(v);
                  if (portfolioRange === '1y' || portfolioRange === '1m') {
                    return dt.toLocaleDateString([], { month: 'short', day: 'numeric' });
                  }
                  if (portfolioRange === '1w') {
                    return dt.toLocaleDateString([], { weekday: 'short', day: 'numeric' });
                  }
                  return dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                }}
              />
              <YAxis
                domain={[chartBounds.min - chartBounds.pad, chartBounds.max + chartBounds.pad]}
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
              />
              <Tooltip
                formatter={(value: number) => fmt(value)}
                labelFormatter={(label: string) => new Date(label).toLocaleString()}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid rgba(148,163,184,0.35)',
                  borderRadius: 8,
                }}
              />
              <Line
                type="monotone"
                dataKey={activeSeriesKey}
                stroke={chartMode === 'paper' ? '#34d399' : '#22d3ee'}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-[var(--text-muted)]">No portfolio history available yet.</p>
        )}
      </div>
    </div>
  );
}
