/** The dashboard charts, in Recharts.
 *
 * Plain styling: no gradients, shadows, rounded bars or animation. The two
 * poles are the only directional colours; the accent is never one of them.
 */
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import type {
  CalibrationBin,
  EquityPoint,
  FunnelStage,
  VenuePnlPoint,
} from '../types';
import { Empty } from './Primitives';
import { pct, stamp } from '../lib/format';

const AXIS = { stroke: 'var(--rule-firm)', strokeWidth: 1 };
const TICK = { fill: 'var(--ink-3)', fontSize: 10, fontFamily: '"Kode Mono", monospace' };
const GRID = { stroke: 'var(--rule)', strokeDasharray: '2 3' };

const TOOLTIP = {
  contentStyle: {
    background: 'var(--surface)',
    border: '1px solid var(--rule-firm)',
    borderRadius: 2,
    fontSize: 11,
    fontFamily: '"Kode Mono", monospace',
    color: 'var(--ink)',
    boxShadow: 'none',
  },
  labelStyle: { color: 'var(--ink-3)', fontSize: 10 },
};


/* ----------------------------------------------------------- venue account */

/** The account chart: cumulative realised P&L from the venue's ledger.
 *
 *  Built from settlement P&L, not balance differences, because a deposit and a
 *  profit look the same in the balance. It steps once per settled market and is
 *  flat in between.
 */
export function VenueAccountChart({ points }: { points: VenuePnlPoint[] }) {
  if (points.length === 0) {
    return (
      <Empty
        what="No venue ledger yet, so there is no chart. A paper account has none — the venue only has a ledger once real orders fill."
        next="python -m scripts.sync_venue"
      />
    );
  }

  // One row per settlement instant.
  type Row = { t: number; cumulative_pnl?: number };
  const rows = new Map<number, Row>();
  const at = (t: number): Row => {
    const existing = rows.get(t);
    if (existing) return existing;
    const created: Row = { t };
    rows.set(t, created);
    return created;
  };
  for (const p of points) at(new Date(p.timestamp).getTime()).cumulative_pnl = p.cumulative_pnl;

  const data = [...rows.values()].sort((a, b) => a.t - b.t);
  // Forward-fill: P&L is a step function, flat between settlements.
  let carried: number | undefined;
  for (const row of data) {
    if (row.cumulative_pnl == null) row.cumulative_pnl = carried;
    else carried = row.cumulative_pnl;
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <ComposedChart data={data} margin={{ top: 8, right: 0, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis
          dataKey="t"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={(v) => stamp(new Date(v).toISOString())}
          tick={TICK}
          axisLine={AXIS}
          tickLine={false}
          minTickGap={48}
        />
        <YAxis
          yAxisId="pnl"
          tick={TICK}
          axisLine={false}
          tickLine={false}
          width={56}
          tickFormatter={(v) => `${Number(v) >= 0 ? '+' : ''}$${Number(v).toFixed(0)}`}
          domain={['auto', 'auto']}
        />
        {/* Break-even. */}
        <ReferenceLine
          yAxisId="pnl"
          y={0}
          stroke="var(--rule-firm)"
          strokeDasharray="3 3"
          label={{
            value: 'break-even',
            position: 'insideLeft',
            fill: 'var(--ink-3)',
            fontSize: 10,
          }}
        />
        <Tooltip
          {...TOOLTIP}
          labelFormatter={(v) => stamp(new Date(Number(v)).toISOString())}
          formatter={(v: number, name) => [
            `${name === 'realised p&l' && v >= 0 ? '+' : ''}$${v.toFixed(2)}`,
            name,
          ]}
        />
        <Area
          yAxisId="pnl"
          type="stepAfter"
          dataKey="cumulative_pnl"
          name="realised p&l"
          stroke="var(--accent)"
          strokeWidth={1.5}
          fill="var(--accent-wash)"
          fillOpacity={1}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ---------------------------------------------------------------- equity */

/**
 * P&L against the start of the series, not the equity level, which a deposit
 * or withdrawal moves.
 */
export function EquityChart({ points }: { points: EquityPoint[] }) {
  if (points.length === 0) {
    return (
      <Empty
        what="No settled positions yet, so there is no P&L curve."
        next="python -m scripts.live"
      />
    );
  }
  const start = points[0].equity;
  const data = points.map((p) => ({
    ...p,
    t: new Date(p.timestamp).getTime(),
    pnl: p.equity - start,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis
          dataKey="t"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={(v) => stamp(new Date(v).toISOString())}
          tick={TICK}
          axisLine={AXIS}
          tickLine={false}
          minTickGap={48}
        />
        <YAxis
          tick={TICK}
          axisLine={false}
          tickLine={false}
          width={52}
          tickFormatter={(v) => `${Number(v) >= 0 ? '+' : ''}$${Number(v).toFixed(0)}`}
          domain={['auto', 'auto']}
        />
        {/* Break-even. */}
        <ReferenceLine
          y={0}
          stroke="var(--rule-firm)"
          strokeDasharray="3 3"
          label={{
            value: 'break-even',
            position: 'insideLeft',
            fill: 'var(--ink-3)',
            fontSize: 10,
          }}
        />
        <Tooltip
          {...TOOLTIP}
          labelFormatter={(v) => stamp(new Date(Number(v)).toISOString())}
          formatter={(v: number, name) => [`${v >= 0 ? '+' : ''}$${v.toFixed(2)}`, name]}
        />
        <Area
          type="stepAfter"
          dataKey="pnl"
          name="p&l"
          stroke="var(--accent)"
          strokeWidth={1.5}
          fill="var(--accent-wash)"
          fillOpacity={1}
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ----------------------------------------------------------- reliability */

/** Observed frequency against predicted probability, model and baseline,
 * drawn against the diagonal.
 */
export function ReliabilityChart({ bins }: { bins: CalibrationBin[] }) {
  if (bins.length === 0) {
    return (
      <Empty
        what="No reliability table stored."
        next="python -m scripts.evaluate --out runs/latest"
      />
    );
  }
  const series = (source: 'model' | 'baseline') =>
    bins
      .filter((b) => b.source === source && b.predicted != null && b.observed != null)
      .map((b) => ({ x: b.predicted!, y: b.observed!, n: b.count }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 20, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="x"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tick={TICK}
          axisLine={AXIS}
          tickLine={false}
          label={{
            value: 'predicted',
            position: 'insideBottom',
            offset: -12,
            fill: 'var(--ink-3)',
            fontSize: 10,
          }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tick={TICK}
          axisLine={false}
          tickLine={false}
          width={40}
          label={{
            value: 'observed',
            angle: -90,
            position: 'insideLeft',
            fill: 'var(--ink-3)',
            fontSize: 10,
          }}
        />
        <ZAxis dataKey="n" range={[24, 220]} />
        {/* Perfect calibration. */}
        <ReferenceLine
          segment={[
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ]}
          stroke="var(--rule-firm)"
        />
        <Tooltip
          {...TOOLTIP}
          formatter={(v: number, name) => [
            name === 'n' ? v.toLocaleString() : pct(v, 2),
            name === 'x' ? 'predicted' : name === 'y' ? 'observed' : 'windows',
          ]}
        />
        <Scatter
          name="baseline"
          data={series('baseline')}
          fill="var(--mid)"
          shape="circle"
          isAnimationActive={false}
        />
        <Scatter
          name="model"
          data={series('model')}
          fill="var(--accent)"
          shape="square"
          isAnimationActive={false}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* ----------------------------------------------------------------- funnel */

/** Why the system declined, in gate order.
 *
 * `traded` uses the accent, which separates it from the refusals without
 * implying a direction.
 */
export function FunnelChart({ stages }: { stages: FunnelStage[] }) {
  if (stages.length === 0) {
    return <Empty what="No decisions recorded in this window." />;
  }
  const data = stages.map((s) => ({ ...s, label: s.reason.replace(/_/g, ' ') }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 26)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 48, bottom: 0, left: 4 }}>
        <CartesianGrid {...GRID} horizontal={false} />
        <XAxis type="number" tick={TICK} axisLine={AXIS} tickLine={false} />
        <YAxis
          type="category"
          dataKey="label"
          tick={{ ...TICK, fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={132}
        />
        <Tooltip
          {...TOOLTIP}
          formatter={(v: number, _n, item) => [
            `${v.toLocaleString()}  (${pct(item?.payload?.share, 1)})`,
            'decisions',
          ]}
        />
        <Bar dataKey="count" isAnimationActive={false} barSize={12}>
          {data.map((d) => (
            <Cell
              key={d.reason}
              fill={d.reason === 'traded' ? 'var(--accent)' : 'var(--ink-3)'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
