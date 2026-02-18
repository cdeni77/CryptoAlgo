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

interface WalletInfoProps {
  loading: boolean;
  showPaperMetrics?: boolean;
}

export default function WalletInfo({ loading, showPaperMetrics = true }: WalletInfoProps) {
  const [wallet, setWallet] = useState<WalletData | null>(null);
  const [portfolioRange, setPortfolioRange] = useState<'1h' | '1d' | '1w' | '1m' | '1y'>('1d');
  const [backfilledSeries, setBackfilledSeries] = useState<Array<{
    timestamp: string;
    paper_equity_usd: number;
    external_usd: number;
    total_value_usd: number;
  }>>([]);

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
    return seriesFromRange?.length ? seriesFromRange : wallet.portfolio_history ?? [];
  }, [wallet, portfolioRange]);

  const shouldBackfill = useMemo(
    () => portfolioSeries.length > 0 && portfolioSeries.every((point) => (point.total_value_usd ?? 0) === 0),
    [portfolioSeries]
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
          }))
        );

        const mergedSeries = new Map<string, { timestamp: string; total_value_usd: number }>();

        for (const { amount, history } of histories) {
          for (const candle of history) {
            const existing = mergedSeries.get(candle.timestamp);
            const value = amount * candle.close;
            mergedSeries.set(candle.timestamp, {
              timestamp: candle.timestamp,
              total_value_usd: (existing?.total_value_usd ?? 0) + value,
            });
          }
        }

        const nextSeries = Array.from(mergedSeries.values())
          .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
          .map((point) => ({
            timestamp: point.timestamp,
            paper_equity_usd: 0,
            external_usd: point.total_value_usd,
            total_value_usd: point.total_value_usd,
          }));

        setBackfilledSeries(nextSeries);
      } catch (error) {
        console.error('Failed to backfill portfolio chart from holdings:', error);
        setBackfilledSeries([]);
      }
    };

    buildBackfill();
  }, [wallet, portfolioRange, shouldBackfill]);

  const chartSeries = shouldBackfill && backfilledSeries.length ? backfilledSeries : portfolioSeries;

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
  ];

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {items.map(({ label, value, color }) => (
          <div key={label} className="p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
            <p className="text-[10px] font-mono-trade text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
              {label}
            </p>
            <p className={`text-xl font-bold font-mono-trade ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)] h-72">
        <div className="mb-3 flex items-center justify-between gap-2">
          <p className="text-sm font-semibold text-[var(--text-primary)]">Total Portfolio Trend</p>
          <div className="flex gap-1 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-primary)] p-1">
            {(['1h', '1d', '1w', '1m', '1y'] as const).map((range) => (
              <button
                key={range}
                type="button"
                onClick={() => setPortfolioRange(range)}
                className={`px-2 py-1 text-xs rounded ${
                  portfolioRange === range
                    ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                {range.toUpperCase()}
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
                  return portfolioRange === '1y'
                    ? dt.toLocaleDateString([], { month: 'short', day: 'numeric' })
                    : dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                }}
              />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v: number) => `$${Math.round(v)}`} />
              <Tooltip
                formatter={(value: number) => fmt(value)}
                labelFormatter={(label: string) => new Date(label).toLocaleString()}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid rgba(148,163,184,0.35)',
                  borderRadius: 8,
                }}
              />
              <Line type="monotone" dataKey="total_value_usd" stroke="#22d3ee" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-[var(--text-muted)]">No portfolio history available yet.</p>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
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
    </div>
  );
}
