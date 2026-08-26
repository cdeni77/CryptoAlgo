/** The three charts, and what each is for.
 *
 * Recharts, styled down hard: no gradients, no drop shadows, no rounded bars, no
 * animated entrances. A faint grid, hairline axes, and an emphasised endpoint
 * where one carries meaning. The colour rules from the palette hold — the two
 * poles are the only directional colours and the accent is never one of them.
 */
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
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
  VenueBalancePoint,
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

/** The account chart, drawn from the venue's ledger. Two series, two questions.
 *
 *  **Realised P&L** steps once per settled market and is the answer to "how is
 *  the portfolio doing". It is built from settlement P&L rather than from balance
 *  differences, and that is the load-bearing choice: nothing in the ledger
 *  distinguishes a deposit from a profit, so a balance-difference curve reports
 *  the first deposit as the best day the strategy ever had.
 *
 *  **Balance** is the venue's cash, sampled every live cycle. It is the real
 *  money and it gives the chart a present rather than ending at the last
 *  settlement — but on its own it sawtooths, because entering a position debits
 *  it and settling credits it back. So it is the thin second line on its own
 *  axis, never the headline.
 *
 *  The P&L series is forward-filled between settlements. That is not invented
 *  data: cumulative realised P&L genuinely does not change until something
 *  settles, which is what a step means.
 */
export function VenueAccountChart({
  points,
  balances,
}: {
  points: VenuePnlPoint[];
  balances: VenueBalancePoint[];
}) {
  if (points.length === 0 && balances.length === 0) {
    return (
      <Empty
        what="No venue ledger yet, so there is no chart. A paper account has none — the venue only has a ledger once real orders fill."
        next="python -m scripts.sync_venue"
      />
    );
  }

  // One row per instant either series has a reading at, so both can be drawn
  // against a shared time axis without resampling either onto the other's grid.
  type Row = { t: number; cumulative_pnl?: number; balance?: number };
  const rows = new Map<number, Row>();
  const at = (t: number): Row => {
    const existing = rows.get(t);
    if (existing) return existing;
    const created: Row = { t };
    rows.set(t, created);
    return created;
  };
  for (const p of points) at(new Date(p.timestamp).getTime()).cumulative_pnl = p.cumulative_pnl;
  for (const b of balances) at(new Date(b.timestamp).getTime()).balance = b.balance;

  const data = [...rows.values()].sort((a, b) => a.t - b.t);
  // Forward-fill the P&L only. The balance is left with gaps and `connectNulls`
  // draws through them, because an unsampled balance is unknown rather than
  // unchanged — the venue's cash moves whether or not we looked.
  let carried: number | undefined;
  for (const row of data) {
    if (row.cumulative_pnl == null) row.cumulative_pnl = carried;
    else carried = row.cumulative_pnl;
  }

  const hasBalance = balances.length > 0;

  return (
    <ResponsiveContainer width="100%" height={240}>
      <ComposedChart data={data} margin={{ top: 8, right: hasBalance ? 8 : 0, bottom: 0, left: 0 }}>
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
        {hasBalance && (
          <YAxis
            yAxisId="cash"
            orientation="right"
            tick={TICK}
            axisLine={false}
            tickLine={false}
            width={52}
            tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
            domain={['auto', 'auto']}
          />
        )}
        {/* Break-even. The line that matters on a P&L chart, and the one a
            currency axis alone does not make legible. */}
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
        {hasBalance && (
          <Line
            yAxisId="cash"
            type="linear"
            dataKey="balance"
            name="venue cash"
            stroke="var(--ink-3)"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ---------------------------------------------------------------- equity */

export function EquityChart({ points }: { points: EquityPoint[] }) {
  if (points.length === 0) {
    return (
      <Empty
        what="No settled positions yet, so there is no equity curve."
        next="python -m scripts.live"
      />
    );
  }
  const start = points[0].equity;
  const data = points.map((p) => ({ ...p, t: new Date(p.timestamp).getTime() }));

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
          tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
          domain={['auto', 'auto']}
        />
        {/* The starting bankroll, so above and below the line are legible without
            reading the axis. */}
        <ReferenceLine
          y={start}
          stroke="var(--rule-firm)"
          strokeDasharray="3 3"
          label={{ value: 'start', position: 'insideLeft', fill: 'var(--ink-3)', fontSize: 10 }}
        />
        <Tooltip
          {...TOOLTIP}
          labelFormatter={(v) => stamp(new Date(Number(v)).toISOString())}
          formatter={(v: number, name) => [`$${v.toFixed(2)}`, name]}
        />
        <Area
          type="stepAfter"
          dataKey="equity"
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

/** Observed frequency against predicted probability, model and baseline.
 *
 * The one diagnostic that cannot be faked by a good average: a model can hit the
 * base rate exactly while being wrong at every level of confidence. Since this
 * system only trades its confident predictions, a deviation in the 0.85–0.95
 * band matters far more than the headline number — so both curves are drawn
 * against the diagonal and the diagonal is the subject of the chart.
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
        {/* Perfect calibration. The chart is about distance from this line. */}
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
 * Expected to be dominated by `edge_below_gate`, and that is the system working
 * rather than failing: the forecast does not cover the fee, so it abstains. On
 * the predecessor of this system the equivalent count said what no Sharpe ratio
 * said. `traded` is drawn in the accent so it separates from the refusals
 * without implying a direction.
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

/* ------------------------------------------------------------ skill by fold */

export function FoldSkillChart({
  folds,
}: {
  folds: { fold: number; skill: number }[];
}) {
  if (folds.length === 0) return <Empty what="No per-fold skill recorded." />;
  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={folds} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="fold" tick={TICK} axisLine={AXIS} tickLine={false} />
        <YAxis tick={TICK} axisLine={false} tickLine={false} width={62} />
        <ReferenceLine y={0} stroke="var(--rule-firm)" />
        <Tooltip {...TOOLTIP} formatter={(v: number) => [v.toFixed(5), 'log loss skill']} />
        <Line
          type="linear"
          dataKey="skill"
          stroke="var(--accent)"
          strokeWidth={1.5}
          dot={{ r: 2.5, fill: 'var(--accent)', strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
