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

/** One candle: the high-low wick and the open-close body, drawn to scale.
 *
 * The bar's `dataKey` is the full `[low, high]` range, so recharts hands this
 * shape a rectangle spanning exactly that range in pixels — which is enough to
 * recover the price-to-pixel scale locally and place the body inside it. The
 * previous version drew two bars covering open-to-close only: the tooltip
 * reported H and L that appeared nowhere on the chart, so every bar looked like
 * its entire range was the body, and recharts also allocated a category slot to
 * each of the two bars, offsetting every candle from its own timestamp.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function Candle({ x, y, width, height, payload }: any) {
  const d = payload as HistoryEntry;
  if (
    typeof x !== 'number' || typeof y !== 'number' ||
    typeof width !== 'number' || typeof height !== 'number' ||
    !d
  ) return null;

  const span = d.high - d.low;
  // A bar that never moved has no scale to recover. Draw it as a hairline at
  // its single price rather than dividing by zero.
  const priceToY = (price: number) =>
    span > 0 ? y + ((d.high - price) / span) * height : y + height / 2;

  const up = d.close >= d.open;
  const colour = up ? '#34d399' : '#fb7185';
  const bodyTop = priceToY(Math.max(d.open, d.close));
  const bodyBottom = priceToY(Math.min(d.open, d.close));
  const centre = x + width / 2;

  return (
    <g opacity={0.85}>
      <line
        x1={centre} x2={centre}
        y1={priceToY(d.high)} y2={priceToY(d.low)}
        stroke={colour} strokeWidth={1}
      />
      <rect
        x={x} width={Math.max(width, 1)}
        y={bodyTop}
        // A doji has zero body height and would render as nothing.
        height={Math.max(bodyBottom - bodyTop, 1)}
        fill={colour}
      />
    </g>
  );
}

export default function PriceChart({ data, fills = [], coin, mode = 'candle' }: Props) {
  const chartData = useMemo(() => data.map(d => ({
    ...d,
    range: [d.low, d.high],
  })), [data]);

  // Keyed by fill id, not array index: the list is filtered and repolled, so an
  // index key reuses one fill's DOM node for another fill's price.
  const coinFills = useMemo(() =>
    fills.filter(f => f.coin === coin),
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
        {coinFills.map(f => (
          <ReferenceLine
            key={f.id} y={f.fill_price}
            stroke="rgba(56,189,248,0.4)" strokeDasharray="3 4"
          />
        ))}
        {mode === 'candle' ? (
          <Bar dataKey="range" shape={<Candle />} isAnimationActive={false} />
        ) : (
          <Line dataKey="close" stroke={lineColor} dot={false} strokeWidth={1.5} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
