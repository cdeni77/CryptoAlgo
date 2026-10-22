/** One model's own record: its P&L from zero, its stats and its positions.
 *
 *  The curve is rebuilt from this model's positions rather than cut out of
 *  the account curve. Two models traded on the same day overlap at the
 *  handover, so they do not partition the account and a slice would double
 *  count or drop the join.
 */
import { fetchModelDetail } from '../api/serving';
import { usePolling } from '../hooks/usePolling';
import type { ModelCurvePoint } from '../types';
import { Empty, Failed, Loading, Metric, Panel, SectionHead } from './Primitives';

const money = (v: number) => `${v >= 0 ? '+' : '−'}$${Math.abs(v).toFixed(2)}`;

function tone(v: number | null | undefined): string {
  if (v == null || v === 0) return 'text-ink-3';
  return v > 0 ? 'text-up' : 'text-down';
}

function when(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** A dependency-free sparkline. The shared charts take their own point
 *  shapes; this series is a model's own cumulative P&L and has no zero line
 *  in common with them. */
function Curve({ points }: { points: ModelCurvePoint[] }) {
  if (points.length < 2) return null;
  const w = 720;
  const h = 120;
  const values = points.map((p) => p.cumulative_pnl);
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (points.length - 1)) * w;
  const y = (v: number) => h - ((v - lo) / span) * h;
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.cumulative_pnl).toFixed(1)}`).join(' ');
  const last = values[values.length - 1];
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-3 h-28 w-full"
      preserveAspectRatio="none"
      role="img"
      aria-label="Cumulative profit and loss for this model"
    >
      <line
        x1={0}
        x2={w}
        y1={y(0)}
        y2={y(0)}
        className="stroke-line"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <path
        d={d}
        fill="none"
        className={last >= 0 ? 'stroke-up' : 'stroke-down'}
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

export function ModelDetailPanel({
  version,
  pollMs,
  onClose,
}: {
  version: string;
  pollMs: number;
  onClose: () => void;
}) {
  const detail = usePolling(() => fetchModelDetail(version, true), pollMs, [version]);
  const s = detail.data?.summary;

  return (
    <Panel>
      <SectionHead
        eyebrow="Model"
        title={version}
        note="This model's own record, from its first position to its last."
        right={
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-line px-2 py-1 text-tiny text-ink-3"
          >
            Back to all models
          </button>
        }
      />

      {detail.loading ? (
        <Loading what="this model" />
      ) : detail.error ? (
        <Failed error={detail.error} what="this model" />
      ) : !s || s.positions === 0 ? (
        <Empty what="This model has no positions recorded against it." />
      ) : (
        <>
          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-4">
            <Metric
              label="p&l"
              value={s.pnl}
              unit="$"
              size="lg"
              tone={s.pnl >= 0 ? 'above' : 'below'}
              hint="settled positions only, this model alone"
            />
            <Metric
              label="per position"
              value={s.pnl_per_position}
              unit="$"
              digits={3}
              tone={(s.pnl_per_position ?? 0) >= 0 ? 'above' : 'below'}
            />
            <Metric
              label="won"
              value={s.win_rate == null ? null : 100 * s.win_rate}
              unit="%"
              digits={0}
              hint={`${s.settled} settled. A win is a position that MADE MONEY, not the market going up.`}
            />
            <Metric
              label="traded"
              value={s.positions}
              digits={0}
              hint={
                s.open > 0
                  ? `${s.open} still open, ${s.contracts} contracts`
                  : `${s.contracts} contracts, $${s.fees.toFixed(2)} fees`
              }
            />
          </div>

          <Curve points={detail.data?.curve ?? []} />
          <p className="text-tiny text-ink-3">
            This model&apos;s cumulative P&L from zero, by settlement. Not a
            slice of the account curve — models overlap at a handover, so they
            do not partition the account. Fees ${s.fees.toFixed(2)}.
          </p>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-tiny">
              <thead className="text-ink-3">
                <tr className="text-left">
                  <th className="py-1 pr-3 font-normal">Window</th>
                  <th className="py-1 pr-3 font-normal">Symbol</th>
                  <th className="py-1 pr-3 font-normal">Side</th>
                  <th className="py-1 pr-3 text-right font-normal">+m</th>
                  <th className="py-1 pr-3 text-right font-normal">Ct</th>
                  <th className="py-1 pr-3 text-right font-normal">Price</th>
                  <th className="py-1 text-right font-normal">P&L</th>
                </tr>
              </thead>
              <tbody>
                {(detail.data?.positions ?? []).slice(0, 60).map((p, i) => (
                  <tr key={`${p.symbol}-${p.window_open}-${i}`} className="border-t border-line">
                    <td className="py-1 pr-3 text-ink-3">{when(p.window_open)}</td>
                    <td className="py-1 pr-3">{p.symbol.replace('-USD', '')}</td>
                    <td className="py-1 pr-3">{p.side}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">{p.offset_minutes}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">{p.contracts}</td>
                    <td className="py-1 pr-3 text-right tabular-nums">{p.price.toFixed(2)}</td>
                    <td className={`py-1 text-right tabular-nums ${tone(p.pnl)}`}>
                      {p.pnl == null ? 'open' : money(p.pnl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {(detail.data?.positions.length ?? 0) > 60 && (
              <p className="mt-2 text-tiny text-ink-3">
                Newest 60 of {detail.data?.positions.length}.
              </p>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}
