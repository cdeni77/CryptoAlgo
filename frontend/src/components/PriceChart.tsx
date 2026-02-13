import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Scatter,
} from 'recharts';
import { HistoryEntry, CoinSymbol, DataSource, CDESpec, PaperFill } from '../types';
import DataSourceToggle from './DataSourceToggle';

interface PriceChartProps {
  data: HistoryEntry[];
  fills: PaperFill[];
  symbol: CoinSymbol;
  loading: boolean;
  timeRange: '1h' | '1d' | '1w' | '1m' | '1y';
  setTimeRange: (range: '1h' | '1d' | '1w' | '1m' | '1y') => void;
  dataSource: DataSource;
  onDataSourceChange: (source: DataSource) => void;
  cdeSpec?: CDESpec;
}

export default function PriceChart({
  data, fills, symbol, loading, timeRange, setTimeRange,
  dataSource, onDataSourceChange, cdeSpec,
}: PriceChartProps) {
  const ranges = ['1h', '1d', '1w', '1m', '1y'] as const;

  const rangeLabels: Record<typeof timeRange, string> = {
    '1h': '1H',
    '1d': '24H',
    '1w': '7D',
    '1m': '30D',
    '1y': '1Y',
  };

  if (loading) {
    return (
      <div className="glass-card rounded-xl p-8 flex items-center justify-center h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-[var(--text-muted)]">Loading chart...</span>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="glass-card rounded-xl p-8 text-center h-[400px] flex items-center justify-center">
        <span className="text-[var(--text-muted)]">No data for this period</span>
      </div>
    );
  }

  // Transform data: if CDE mode, multiply close prices by units_per_contract
  const chartData = data.map(d => {
    const multiplier = (dataSource === 'cde' && cdeSpec) ? cdeSpec.units_per_contract : 1;
    return {
      ...d,
      timestampMs: new Date(d.timestamp).getTime(),
      close: d.close * multiplier,
    };
  });

  const prices = chartData.map(d => d.close);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const pad = (maxPrice - minPrice) * 0.06;
  const isPositive = chartData.length >= 2 && chartData[chartData.length - 1].close >= chartData[0].close;
  const strokeColor = isPositive ? '#34d399' : '#fb7185';
  const fillId = isPositive ? 'gradientGreen' : 'gradientRed';

  const formatXAxis = (timestamp: string) => {
    const date = new Date(timestamp);
    switch (timeRange) {
      case '1h':
      case '1d':
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      case '1w':
        return date.toLocaleDateString([], { weekday: 'short', day: 'numeric' });
      case '1m':
      case '1y':
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
      default:
        return date.toLocaleDateString();
    }
  };

  const formatPrice = (value: number) => {
    if (value >= 10000) return `$${(value / 1000).toFixed(1)}k`;
    if (value >= 1) return `$${value.toFixed(2)}`;
    return `$${value.toFixed(4)}`;
  };

  const sourceLabel = dataSource === 'cde' && cdeSpec
    ? `${cdeSpec.code} Contract Value`
    : `${symbol}/USD Spot`;

  const chartStartMs = chartData[0].timestampMs;
  const chartEndMs = chartData[chartData.length - 1].timestampMs;
  const fillMultiplier = (dataSource === 'cde' && cdeSpec) ? cdeSpec.units_per_contract : 1;

  const fillMarkers = fills
    .filter((fill) => fill.coin === symbol)
    .map((fill) => ({
      ...fill,
      timestampMs: new Date(fill.created_at).getTime(),
      plottedPrice: fill.fill_price * fillMultiplier,
    }))
    .filter((fill) => fill.timestampMs >= chartStartMs && fill.timestampMs <= chartEndMs)
    .sort((a, b) => a.timestampMs - b.timestampMs);

  const longFillMarkers = fillMarkers.filter((fill) => fill.side === 'long');
  const shortFillMarkers = fillMarkers.filter((fill) => fill.side === 'short');

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      {/* Chart header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center p-5 pb-0 gap-3">
        <div className="flex items-center gap-4">
          <div>
            <h3 className="text-lg font-bold text-[var(--text-primary)]">{symbol} Price</h3>
            <p
              className={`text-xs font-mono-trade mt-0.5 ${
                dataSource === 'cde' ? 'text-[var(--accent-cyan)]' : 'text-[var(--text-muted)]'
              }`}
            >
              {sourceLabel}
            </p>
          </div>
          <DataSourceToggle source={dataSource} onChange={onDataSourceChange} compact />
        </div>
        <div className="flex gap-1 p-0.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
          {ranges.map((r) => (
            <button
              key={r}
              onClick={() => setTimeRange(r)}
              className={`
                px-3 py-1 rounded-md text-xs font-mono-trade font-medium transition-all duration-200
                ${timeRange === r
                  ? 'bg-[var(--bg-elevated)] text-[var(--accent-cyan)] shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }
              `}
            >
              {rangeLabels[r]}
            </button>
          ))}
        </div>
      </div>

      {/* Chart body */}
      <div className="p-4 pt-2">
        <ResponsiveContainer width="100%" height={340}>
          <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="gradientGreen" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradientRed" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#fb7185" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#fb7185" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(56, 189, 248, 0.04)" vertical={false} />
            <XAxis
              dataKey="timestampMs"
              type="number"
              domain={[chartStartMs, chartEndMs]}
              scale="time"
              tickFormatter={(ts) => formatXAxis(new Date(ts).toISOString())}
              tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              axisLine={{ stroke: 'rgba(56, 189, 248, 0.06)' }}
              tickLine={false}
              minTickGap={40}
            />
            <YAxis
              domain={[minPrice - pad, maxPrice + pad]}
              tickFormatter={formatPrice}
              tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              axisLine={false}
              tickLine={false}
              width={70}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#131927',
                border: '1px solid rgba(56, 189, 248, 0.15)',
                borderRadius: '8px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                fontFamily: 'JetBrains Mono',
                fontSize: '12px',
                color: '#e2e8f0',
              }}
              labelFormatter={(label) => {
                const d = new Date(label);
                return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
              }}
              formatter={(value: number, _name, props) => {
                if (props?.payload?.markerType === 'paper-fill') {
                  const sideLabel = props.payload.side === 'long' ? 'Entry / Buy' : 'Exit / Sell';
                  return [`${formatPrice(value)} (${props.payload.contracts} ctr)`, sideLabel];
                }

                return [formatPrice(value), dataSource === 'cde' ? 'Contract' : 'Price'];
              }}
            />
            <Area
              type="monotone"
              dataKey="close"
              stroke={strokeColor}
              strokeWidth={2}
              fill={`url(#${fillId})`}
              dot={false}
              activeDot={{ r: 4, stroke: strokeColor, strokeWidth: 2, fill: '#0a0e17' }}
            />
            <Scatter
              data={longFillMarkers.map((fill) => ({ ...fill, markerType: 'paper-fill' }))}
              dataKey="plottedPrice"
              fill="#22c55e"
              shape="triangle"
              legendType="triangle"
            />
            <Scatter
              data={shortFillMarkers.map((fill) => ({ ...fill, markerType: 'paper-fill' }))}
              dataKey="plottedPrice"
              fill="#ef4444"
              shape="triangle"
              legendType="triangle"
            />
          </AreaChart>
        </ResponsiveContainer>
        <div className="flex flex-wrap items-center gap-4 px-1 pt-3 text-[11px] text-[var(--text-muted)] font-mono-trade">
          <span className="inline-flex items-center gap-1.5">
            <span className="text-[#22c55e]">▲</span>
            Paper buy/entry
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="text-[#ef4444]">▼</span>
            Paper sell/exit
          </span>
          <span>{fillMarkers.length} fill{fillMarkers.length === 1 ? '' : 's'} in view</span>
        </div>
      </div>
    </div>
  );
}
