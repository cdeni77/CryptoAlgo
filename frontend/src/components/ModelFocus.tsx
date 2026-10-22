/** What the page shows while it is filtered to one model.
 *
 *  Everything here is computed from OUR positions, because that is the only
 *  place a model is recorded. The venue's settlement ledger -- which supplies
 *  the account-wide settled count, win rate and contracts -- is keyed by
 *  market and has no idea which model placed an order, so those panels cannot
 *  narrow and say so instead of quietly showing the account total next to a
 *  model's name.
 */
import type { Position } from '../types';
import { Metric, Panel, SectionHead } from './Primitives';

/** A model's cumulative P&L from zero, by settlement.
 *
 *  Not a slice of the account curve: models overlap at a handover, so cutting
 *  the account series by date would double count the join or drop it. */
function Curve({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 720;
  const h = 140;
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (values.length - 1)) * w;
  const y = (v: number) => h - ((v - lo) / span) * h;
  const d = values
    .map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
    .join(' ');
  const last = values[values.length - 1];
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-2 h-36 w-full"
      preserveAspectRatio="none"
      role="img"
      aria-label="Cumulative profit and loss for the selected model"
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
        className={last >= 0 ? 'stroke-above' : 'stroke-below'}
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

export function ModelFocus({
  version,
  positions,
  onClear,
}: {
  version: string;
  positions: Position[];
  onClear: () => void;
}) {
  // Settled only, oldest first, so the curve runs forward in time.
  const settled = positions
    .filter((p) => p.settled_at != null && p.pnl != null)
    .slice()
    .sort((a, b) => (a.settled_at! < b.settled_at! ? -1 : 1));

  let running = 0;
  const curve = settled.map((p) => (running += p.pnl ?? 0));

  const pnl = running;
  const wins = settled.filter((p) => (p.pnl ?? 0) > 0).length;
  const contracts = positions.reduce((n, p) => n + (p.contracts ?? 0), 0);
  const open = positions.length - settled.length;

  return (
    <Panel>
      <SectionHead
        eyebrow="Filtered"
        title={version}
        note="This model alone, from our own positions."
        right={
          <button
            type="button"
            onClick={onClear}
            className="rounded-md border border-line px-2 py-1 text-tiny text-ink-3"
          >
            Show the whole account
          </button>
        }
      />

      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-4">
        <Metric
          label="p&l"
          value={settled.length ? pnl : null}
          unit="$"
          size="lg"
          tone={pnl >= 0 ? 'above' : 'below'}
          hint="settled positions decided by this model"
        />
        <Metric
          label="per position"
          value={settled.length ? pnl / settled.length : null}
          unit="$"
          digits={3}
          tone={pnl >= 0 ? 'above' : 'below'}
        />
        <Metric
          label="win rate"
          value={settled.length ? (100 * wins) / settled.length : null}
          unit="%"
          digits={2}
          hint="positions that MADE MONEY, not the share where the market rose"
        />
        <Metric
          label="settled"
          value={settled.length}
          digits={0}
          hint={`${contracts} contracts${open ? `, ${open} still open` : ''}`}
        />
      </div>

      <Curve values={curve} />
      <p className="text-tiny text-ink-3">
        This model&apos;s cumulative P&amp;L from zero, one step per settled
        position. The account chart above it is the venue ledger and stays
        account-wide: the venue records a settlement against a market, not
        against whichever model decided to be there.
      </p>
    </Panel>
  );
}
