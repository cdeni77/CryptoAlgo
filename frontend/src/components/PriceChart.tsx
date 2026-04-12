import { useMemo } from 'react';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { HistoryEntry, PaperFill } from '../types';

interface Props {
  data: HistoryEntry[];
  fills?: PaperFill[];
  coin: string;
  mode?: 'candle' | 'line';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as HistoryEntry;
  if (!d) return null;
  const chg = ((d.close - d.open) / d.open) * 100;
  const up = d.close >= d.open;
  return (
    <div className="glass-card rounded-lg p-3 text-xs font-mono min-w-[140px]">
      <div className="text-tx-muted mb-1.5">
        {new Date(d.timestamp).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}
      </div>
      {[['O', d.open], ['H', d.high], ['L', d.low], ['C', d.close]].map(([k, v]) => (
        <div key={String(k)} className="flex justify-between gap-3">
          <span className="text-tx-muted">{k}</span>
          <span className="text-tx-primary">${Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
        </div>
      ))}
      <div className={`mt-1 ${up ? 'text-accent-emerald' : 'text-accent-rose'}`}>{up ? '+' : ''}{chg.toFixed(2)}%</div>
    </div>
  );
}

export default function PriceChart({ data, fills = [], coin, mode = 'candle' }: Props) {
  const chartData = useMemo(() => data.map(d => ({
    ...d,
    upBody:   d.close >= d.open ? [d.open, d.close] : [null, null],
    downBody: d.close <  d.open ? [d.close, d.open] : [null, null],
  })), [data]);

  const fillPrices = useMemo(() =>
    fills.filter(f => f.coin === coin).map(f => f.fill_price),
    [fills, coin]
  );

  // Determine overall trend color for line mode
  const lineColor = useMemo(() => {
    if (!data.length) return '#38bdf8';
    const chg = data[data.length - 1].close - data[0].open;
    return chg >= 0 ? '#34d399' : '#fb7185';
  }, [data]);

  if (!data.length) {
    return <div className="flex items-center justify-center h-full text-tx-muted text-sm">Loading chart…</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <XAxis
          dataKey="timestamp"
          tickFormatter={v => {
            const d = new Date(v);
            return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
          }}
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={90}
        />
        <YAxis
          domain={['auto', 'auto']}
          tickFormatter={v => v >= 1000 ? `$${(v/1000).toFixed(1)}k` : `$${v.toFixed(2)}`}
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          axisLine={false} tickLine={false} width={56} orientation="right"
        />
        <Tooltip content={<CustomTooltip />} />
        {fillPrices.map((p, i) => (
          <ReferenceLine key={i} y={p} stroke="rgba(56,189,248,0.4)" strokeDasharray="3 4" />
        ))}
        {mode === 'candle' ? (
          <>
            <Bar dataKey="upBody"   fill="#34d399" opacity={0.85} radius={[1,1,0,0]} />
            <Bar dataKey="downBody" fill="#fb7185" opacity={0.85} radius={[1,1,0,0]} />
            <Line dataKey="close" stroke="rgba(148,163,184,0.15)" dot={false} strokeWidth={0.5} />
          </>
        ) : (
          <Line dataKey="close" stroke={lineColor} dot={false} strokeWidth={1.5} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
