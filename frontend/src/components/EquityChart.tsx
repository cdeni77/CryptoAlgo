import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { PaperEquityPoint } from '../types';

interface Props {
  equity: PaperEquityPoint[];
  startingBalance?: number;
}

function fmt(v: number) {
  return `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtTime(ts: string) {
  const d = new Date(ts);
  return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as PaperEquityPoint;
  const unrealized = d.unrealized_pnl ?? 0;
  const realized   = d.realized_pnl ?? 0;
  return (
    <div className="glass-card rounded-lg p-3 text-xs font-mono min-w-[160px]">
      <div className="text-tx-muted mb-2">{fmtTime(d.timestamp)}</div>
      <div className="flex justify-between gap-4 mb-1">
        <span className="text-tx-secondary">Equity</span>
        <span className="text-tx-primary font-semibold">{fmt(d.equity)}</span>
      </div>
      <div className="flex justify-between gap-4 mb-1">
        <span className="text-tx-secondary">Unrealized</span>
        <span className={unrealized >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}>{fmt(unrealized)}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-tx-secondary">Realized</span>
        <span className={realized >= 0 ? 'text-accent-emerald' : 'text-accent-rose'}>{fmt(realized)}</span>
      </div>
    </div>
  );
}

export default function EquityChart({ equity, startingBalance = 100000 }: Props) {
  const data = useMemo(() => [...equity].reverse(), [equity]);

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-full text-tx-muted text-sm">
        No equity data yet
      </div>
    );
  }

  const latest  = data[data.length - 1]?.equity ?? startingBalance;
  const isUp    = latest >= startingBalance;
  const color   = isUp ? '#34d399' : '#fb7185';
  const gradId  = isUp ? 'equityGradUp' : 'equityGradDown';

  const minVal = Math.min(...data.map(d => d.equity)) * 0.999;
  const maxVal = Math.max(...data.map(d => d.equity)) * 1.001;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="timestamp"
          tickFormatter={v => {
            const d = new Date(v);
            return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
          }}
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          axisLine={false} tickLine={false}
          interval="preserveStartEnd"
          minTickGap={80}
        />
        <YAxis
          domain={[minVal, maxVal]}
          tickFormatter={v => `$${(v/1000).toFixed(1)}k`}
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          axisLine={false} tickLine={false}
          width={52}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={startingBalance} stroke="rgba(56,189,248,0.25)" strokeDasharray="4 4" />
        <Area
          type="monotone"
          dataKey="equity"
          stroke={color}
          strokeWidth={1.5}
          fill={`url(#${gradId})`}
          dot={false}
          activeDot={{ r: 3, fill: color, stroke: 'transparent' }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
