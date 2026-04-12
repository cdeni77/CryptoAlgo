import { useEffect, useState, useMemo } from 'react';
import { getPaperSummary, getPaperEquity, getPaperPositions, getPaperFills, getPaperConfig, getModelStatus } from '../api/paperApi';
import { getCurrentPrices, getCDEPrices } from '../api/coinsApi';
import { PaperSummary, PaperEquityPoint, PaperPosition, PaperFill, PriceData, Signal, ModelStatusData } from '../types';
import EquityChart from '../components/EquityChart';
import PaperPositionsTable from '../components/PaperPositionsTable';
import PaperFillsTable from '../components/PaperFillsTable';
import PriceCard from '../components/PriceCard';
import { getRecentSignals as getSignals } from '../api/signalsApi';
import SignalsTable from '../components/SignalsTable';
import ModelStatusPanel from '../components/ModelStatusPanel';

const ALL_COINS = ['BTC','ETH','SOL','XRP','DOGE','AVAX','ADA','LINK','LTC'] as const;

// Must match trading_costs.py CONTRACT_SPECS
const UNITS_PER_CONTRACT: Record<string, number> = {
  BTC: 0.01, ETH: 0.10, SOL: 5, XRP: 500, DOGE: 5000,
  AVAX: 10, ADA: 1000, LINK: 50, LTC: 5,
};

function fmt(v: number, prefix='$') {
  return `${prefix}${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="glass-card rounded-xl p-5">
      <div className="text-tx-muted text-[11px] font-medium tracking-widest uppercase mb-2">{label}</div>
      <div className={`font-mono text-xl font-semibold mb-1 ${color ?? 'text-tx-primary'}`}>{value}</div>
      {sub && <div className="text-tx-muted text-xs font-mono">{sub}</div>}
    </div>
  );
}

export default function DashboardPage() {
  const [summary,     setSummary]     = useState<PaperSummary | null>(null);
  const [equity,      setEquity]      = useState<PaperEquityPoint[]>([]);
  const [positions,   setPositions]   = useState<PaperPosition[]>([]);
  const [fills,       setFills]       = useState<PaperFill[]>([]);
  const [spotPrices,  setSpotPrices]  = useState<PriceData | null>(null);
  const [cdePrices,   setCdePrices]   = useState<PriceData | null>(null);
  const [priceSource, setPriceSource] = useState<'spot' | 'cde'>('spot');
  const [signals,     setSignals]     = useState<Signal[]>([]);
  const [activeCoins, setActiveCoins] = useState<Set<string>>(new Set());
  const [modelStatus, setModelStatus] = useState<ModelStatusData | null>(null);

  useEffect(() => {
    const load = async () => {
      try { setSummary(await getPaperSummary()); } catch { /* empty */ }
      try { setEquity(await getPaperEquity(300)); } catch { /* empty */ }
      try { setPositions(await getPaperPositions()); } catch { /* empty */ }
      try { setFills(await getPaperFills(50)); } catch { /* empty */ }
      try { setSpotPrices(await getCurrentPrices()); } catch { /* empty */ }
      try { setCdePrices(await getCDEPrices()); } catch { /* empty */ }
      try { setSignals(await getSignals(30)); } catch { /* empty */ }
    };
    const loadConfig = async () => {
      try {
        const cfg = await getPaperConfig();
        setActiveCoins(new Set(cfg.active_coins.map(c => c.toUpperCase())));
      } catch { /* empty */ }
    };
    const loadModelStatus = async () => {
      try { setModelStatus(await getModelStatus()); } catch { /* empty */ }
    };
    load();
    loadConfig();
    loadModelStatus();
    const id = setInterval(load, 5000);
    const cfgId = setInterval(loadConfig, 60000);
    const msId = setInterval(loadModelStatus, 30000);
    return () => { clearInterval(id); clearInterval(cfgId); clearInterval(msId); };
  }, []);

  const prices = priceSource === 'cde' ? cdePrices : spotPrices;

  // Compute live unrealized using correct units_per_contract
  const liveUnrealized = useMemo(() => {
    const openPos = positions.filter(p => p.is_open);
    if (!openPos.length || !prices) return null;
    return openPos.reduce((sum, p) => {
      const px = prices[p.coin as keyof typeof prices]?.price;
      if (!px) return sum + p.unrealized_pnl;
      const units = UNITS_PER_CONTRACT[p.coin.toUpperCase()] ?? 1;
      const sign = p.side === 'long' ? 1 : -1;
      return sum + p.contracts * units * (px - p.entry_price) * sign;
    }, 0);
  }, [positions, prices]);

  // Inject live-price equity point into chart
  const equityWithLive = useMemo(() => {
    if (!equity.length) return equity;
    const latest = equity[0];
    if (liveUnrealized === null) return equity;
    // Use summary-derived equity as the correct base (avoids engine-restart cash_balance drift).
    // Fall back to latest DB equity if summary hasn't loaded yet.
    const baseEquity = summary != null
      ? summary.equity - (summary.unrealized_pnl ?? 0)
      : latest.equity - latest.unrealized_pnl;
    const livePoint: PaperEquityPoint = {
      ...latest, id: -1,
      timestamp: new Date().toISOString(),
      equity: baseEquity + liveUnrealized,
      unrealized_pnl: liveUnrealized,
    };
    return [livePoint, ...equity];
  }, [equity, positions, prices, summary, liveUnrealized]);

  const totalReturn = summary?.total_return_pct ?? 0;
  // Portfolio value: cash from DB + live unrealized (or DB unrealized if no live prices yet)
  const cashBalance = summary != null ? summary.equity - (summary.unrealized_pnl ?? 0) : 100000;
  const equity0 = cashBalance + (liveUnrealized ?? summary?.unrealized_pnl ?? 0);
  const startBal = 100000;

  return (
    <div className="p-6 space-y-5 max-w-[1600px]">
      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Portfolio Value"
          value={fmt(equity0)}
          sub={`Started at ${fmt(startBal)}`}
        />
        <StatCard
          label="Total Return"
          value={`${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`}
          sub={`${fmt(summary?.realized_pnl ?? 0)} realized`}
          color={totalReturn >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}
        />
        <StatCard
          label="Unrealized P&L"
          value={`${(liveUnrealized ?? summary?.unrealized_pnl ?? 0) >= 0 ? '+' : ''}${fmt(liveUnrealized ?? summary?.unrealized_pnl ?? 0)}`}
          sub={`${summary?.open_positions ?? 0} open position${(summary?.open_positions ?? 0) !== 1 ? 's' : ''}${liveUnrealized !== null ? ' · live' : ''}`}
          color={(liveUnrealized ?? summary?.unrealized_pnl ?? 0) >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}
        />
        <StatCard
          label="Win Rate"
          value={summary?.win_rate != null ? `${(summary.win_rate * 100).toFixed(1)}%` : '—'}
          sub={`${summary?.fill_count ?? 0} total fills`}
        />
      </div>

      {/* Equity chart + positions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 glass-card rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">Equity Curve</span>
            <span className="text-tx-muted text-xs font-mono">{equityWithLive.length} points</span>
          </div>
          <div className="h-56">
            <EquityChart equity={equityWithLive} startingBalance={startBal} />
          </div>
        </div>
        <div className="glass-card rounded-xl p-5">
          <div className="text-tx-secondary text-xs font-medium tracking-widest uppercase mb-4">Open Positions</div>
          <PaperPositionsTable positions={positions} prices={prices} />
        </div>
      </div>

      {/* Price grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="text-tx-muted text-[11px] font-medium tracking-widest uppercase">Live Prices</div>
          <div className="flex gap-0.5 p-0.5 rounded bg-[rgba(56,189,248,0.05)] border border-[rgba(56,189,248,0.08)]">
            {(['spot', 'cde'] as const).map(s => (
              <button
                key={s}
                onClick={() => setPriceSource(s)}
                className={`px-2.5 py-0.5 rounded text-[10px] font-mono transition-all ${
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
        <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-9 gap-3">
          {ALL_COINS.map(coin => (
            <PriceCard
              key={coin}
              coin={coin}
              price={prices?.[coin as keyof typeof prices]?.price ?? null}
              change24h={prices?.[coin as keyof typeof prices]?.change24h ?? null}
            />
          ))}
        </div>
      </div>

      {/* Signals + Fills + Model Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="glass-card rounded-xl p-5">
          <div className="text-tx-secondary text-xs font-medium tracking-widest uppercase mb-4">Recent Signals — Active Coins</div>
          <SignalsTable signals={activeCoins.size ? signals.filter(s => activeCoins.has(s.coin.toUpperCase())) : signals} limit={15} />
        </div>
        <div className="glass-card rounded-xl p-5">
          <div className="text-tx-secondary text-xs font-medium tracking-widest uppercase mb-4">Recent Fills</div>
          <PaperFillsTable fills={fills} limit={15} />
        </div>
        <ModelStatusPanel data={modelStatus} />
      </div>
    </div>
  );
}
