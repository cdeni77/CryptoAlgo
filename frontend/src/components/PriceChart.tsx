import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area } from 'recharts';
import { HistoryEntry } from '../types';

interface PriceChartProps {
  data: HistoryEntry[];
  symbol: 'BTC' | 'ETH' | 'SOL';
  loading: boolean;
  timeRange: '1h' | '1d' | '1w' | '1m' | '1y';
  setTimeRange: (range: '1h' | '1d' | '1w' | '1m' | '1y') => void;
}

export default function PriceChart({ 
  data, 
  symbol, 
  loading, 
  timeRange, 
  setTimeRange 
}: PriceChartProps) {
  const ranges = ['1h', '1d', '1w', '1m', '1y'] as const;

  const rangeLabels: Record<typeof timeRange, string> = {
    '1h': 'Last Hour',
    '1d': 'Last 24 Hours',
    '1w': 'Last 7 Days',
    '1m': 'Last 30 Days',
    '1y': 'Last 365 Days',
  };

  if (loading) {
    return (
      <div className="glass rounded-2xl p-12 flex items-center justify-center h-[420px]">
        <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="glass rounded-2xl p-12 text-center h-[420px] flex items-center justify-center text-gray-400">
        No historical data available for this period
      </div>
    );
  }

  // Dynamic Y-domain padding (smaller for short ranges)
  const prices = data.map(d => d.close);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const rangePadding = (maxPrice - minPrice) * (timeRange === '1h' || timeRange === '1d' ? 0.05 : 0.08);

  // Adaptive X-axis formatting
  const formatXAxis = (timestamp: string) => {
    const date = new Date(timestamp);
    switch (timeRange) {
      case '1h':
      case '1d':
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      case '1w':
        return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
      case '1m':
      case '1y':
        return date.toLocaleDateString([], { month: 'short', year: 'numeric' });
      default:
        return date.toLocaleDateString();
    }
  };

  return (
    <div className="glass rounded-2xl p-5 md:p-6 shadow-2xl overflow-hidden border border-indigo-900/30">
      {/* Header with title + buttons */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <h3 className="text-xl md:text-2xl font-semibold bg-gradient-to-r from-indigo-300 to-blue-300 bg-clip-text text-transparent">
          {symbol} Price • {rangeLabels[timeRange]}
        </h3>

        <div className="flex flex-wrap gap-2">
          {ranges.map((r) => (
            <button
              key={r}
              onClick={() => setTimeRange(r)}
              className={`
                px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200
                border border-indigo-700/50
                ${timeRange === r 
                  ? 'bg-gradient-to-r from-indigo-600 to-blue-600 text-white shadow-lg shadow-indigo-600/30' 
                  : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700/80 hover:border-indigo-500/70'}
              `}
            >
              {r.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={420}>
        <LineChart 
          data={data} 
          margin={{ top: 10, right: 20, left: 10, bottom: 30 }}
        >
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.55} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid 
            strokeDasharray="4 4" 
            stroke="#1e293b" 
            opacity={0.5} 
          />

          <XAxis 
            dataKey="timestamp" 
            stroke="#94a3b8"
            tickFormatter={formatXAxis}
            tick={{ fontSize: 11, fill: '#cbd5e1' }}
            angle={-35}
            textAnchor="end"
            height={50}
            interval="preserveStartEnd"
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
          />

          <YAxis 
            stroke="#94a3b8"
            tickFormatter={(val) => `$${val.toLocaleString(undefined, { notation: 'compact' })}`}
            domain={[minPrice - rangePadding, maxPrice + rangePadding]}
            tick={{ fontSize: 11, fill: '#cbd5e1' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            width={60}
          />

          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#0f172a',
              border: '1px solid #4f46e5',
              borderRadius: '10px',
              boxShadow: '0 8px 20px rgba(0,0,0,0.5)',
              padding: '10px 12px',
              fontSize: '13px',
            }}
            labelStyle={{ color: '#c7d2fe', fontWeight: 600, marginBottom: '6px' }}
            itemStyle={{ color: '#e0f2fe' }}
            formatter={(val: number) => [`$${val.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, 'Price']}
            labelFormatter={(label) => new Date(label).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
          />

          <Area 
            type="monotone" 
            dataKey="close" 
            stroke="none" 
            fill="url(#colorPrice)" 
            fillOpacity={0.35}
          />

          <Line 
            type="monotone" 
            dataKey="close" 
            stroke="#818cf8" 
            strokeWidth={2.5}
            dot={false}
            activeDot={{ 
              r: 7, 
              stroke: '#312e81', 
              strokeWidth: 3,
              fill: '#6366f1'
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}