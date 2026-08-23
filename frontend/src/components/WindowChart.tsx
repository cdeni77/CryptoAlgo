/** The price path against the line it has to finish above.
 *
 * This is the picture a barrier problem asks for, and the one the rest of the
 * dashboard abstracts away. Everything else here reports a probability; this
 * reports the thing the probability is about.
 *
 * Three decisions make it readable rather than merely present:
 *
 * **The strike is drawn per window, not once across the chart.** It is reset
 * every fifteen minutes, so a single horizontal line spanning the whole series
 * would be a different number in every window and true in none of them. Each
 * segment spans exactly its own window.
 *
 * **The area between price and its own strike is filled in the two poles.** Teal
 * above, clay below. That is the only encoding on this chart that carries
 * direction, and it is the same one the probability scale uses, so a glance at
 * either says the same thing.
 *
 * **Window boundaries are marked.** The settlement instants are where the whole
 * question resolves, so they are structural rather than incidental — a faint
 * rule every fifteen minutes, which also gives the eye the quarter-hour grid the
 * system runs on.
 */
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { MinuteBar, WindowStrike } from '../types';
import { Empty } from './Primitives';

const AXIS = { stroke: 'var(--rule-firm)', strokeWidth: 1 };
const TICK = { fill: 'var(--ink-3)', fontSize: 10, fontFamily: '"Kode Mono", monospace' };

interface Point {
  t: number;
  close: number;
  strike: number | null;
  above: number | null;
  below: number | null;
}

export function WindowChart({
  bars,
  strikes,
  height = 200,
}: {
  bars: MinuteBar[];
  /** One per window. The window length is read off these rather than passed in:
   *  the boundaries come from the venue's own open and close times, so a
   *  configured length could disagree with the data being drawn. */
  strikes: WindowStrike[];
  height?: number;
}) {
  if (bars.length === 0) {
    return <Empty what="No minute prices recorded yet." next="python -m scripts.live" />;
  }

  const byWindow = strikes
    .map((s) => ({
      from: new Date(s.window_open).getTime(),
      to: new Date(s.settle_time).getTime(),
      strike: s.strike,
    }))
    .sort((a, b) => a.from - b.from);

  const strikeAt = (t: number): number | null => {
    // Linear scan over at most a few hundred windows, and the series is sorted,
    // so this is cheaper than building an index and easier to be sure about.
    for (const w of byWindow) if (t >= w.from && t < w.to) return w.strike;
    return null;
  };

  const data: Point[] = bars.map((b) => {
    const t = new Date(b.minute).getTime();
    const strike = strikeAt(t);
    return {
      t,
      close: b.close,
      strike,
      // Two series rather than one signed series: recharts fills an area toward
      // a baseline, and splitting them is what lets each half take its own pole
      // colour instead of one colour flipping meaning at the crossing.
      above: strike != null && b.close >= strike ? b.close : null,
      below: strike != null && b.close < strike ? b.close : null,
    };
  });

  const values = data.map((d) => d.close).concat(
    data.map((d) => d.strike).filter((v): v is number => v != null),
  );
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = (high - low) * 0.12 || high * 0.0005;
  const digits = high > 1000 ? 0 : high > 10 ? 2 : 3;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--rule)" strokeDasharray="2 3" vertical={false} />
        <XAxis
          dataKey="t"
          type="number"
          domain={['dataMin', 'dataMax']}
          tick={TICK}
          axisLine={AXIS}
          tickLine={false}
          minTickGap={56}
          tickFormatter={(v) =>
            new Date(v).toLocaleTimeString(undefined, {
              hour: '2-digit',
              minute: '2-digit',
              hour12: false,
            })
          }
        />
        <YAxis
          domain={[low - pad, high + pad]}
          tick={TICK}
          axisLine={false}
          tickLine={false}
          width={64}
          tickFormatter={(v) => Number(v).toFixed(digits)}
        />

        {/* Every settlement instant. The quarter-hour grid the system runs on. */}
        {byWindow.map((w) => (
          <ReferenceLine key={w.to} x={w.to} stroke="var(--rule)" />
        ))}

        <Tooltip
          contentStyle={{
            background: 'var(--surface)',
            border: '1px solid var(--rule-firm)',
            borderRadius: 2,
            fontSize: 11,
            fontFamily: '"Kode Mono", monospace',
            color: 'var(--ink)',
            boxShadow: 'none',
          }}
          labelStyle={{ color: 'var(--ink-3)', fontSize: 10 }}
          labelFormatter={(v) =>
            new Date(Number(v)).toLocaleTimeString(undefined, { hour12: false })
          }
          formatter={(v: number, name) => [
            v == null ? '—' : v.toFixed(digits),
            name === 'strike' ? 'strike' : 'price',
          ]}
        />

        <Area
          type="monotone"
          dataKey="above"
          stroke="none"
          fill="var(--above-wash)"
          fillOpacity={1}
          connectNulls={false}
          isAnimationActive={false}
          baseValue={low - pad}
        />
        <Area
          type="monotone"
          dataKey="below"
          stroke="none"
          fill="var(--below-wash)"
          fillOpacity={1}
          connectNulls={false}
          isAnimationActive={false}
          baseValue={low - pad}
        />

        {/* The strike, stepped: it holds for a window and jumps at settlement. */}
        <Line
          type="stepAfter"
          dataKey="strike"
          stroke="var(--ink-2)"
          strokeWidth={1}
          strokeDasharray="4 2"
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="close"
          stroke="var(--ink)"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
