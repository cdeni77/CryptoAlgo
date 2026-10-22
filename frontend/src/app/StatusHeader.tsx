/** The operational header: what the system is, and whether it is current.
 *
 *  Three facts a trader needs before reading any number below: is this real
 *  money, is the loop still deciding, and how long is left in the window.
 *
 *  **Freshness comes from the payload's own timestamps.** A 200 from the API
 *  proves Postgres answered, not that the trader is running -- the venue
 *  ladder recorder died for thirty-one days while every heartbeat stayed
 *  green, because its loop ran happily and wrote nothing.
 */
import { DECISION_STALE_SECONDS, secondsSince } from '../lib/status';

function ago(seconds: number | null): string {
  if (seconds == null) return '—';
  if (seconds < 90) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

export function StatusHeader({
  title,
  hint,
  mode,
  halted,
  lastDecisionAt,
  secondsLeft,
}: {
  title: string;
  hint: string;
  mode: 'paper' | 'live' | null;
  halted: boolean;
  lastDecisionAt: string | null;
  secondsLeft: number;
}) {
  const age = secondsSince(lastDecisionAt);
  const stale = age != null && age > DECISION_STALE_SECONDS;

  return (
    <header className="sticky top-0 z-10 border-b border-rule bg-paper/95 backdrop-blur-sm">
      <div className="flex items-center justify-between gap-3 px-4 py-3 md:gap-6 md:px-8 md:py-4">
        <div className="min-w-0">
          <div className="eyebrow hidden truncate sm:block">{hint}</div>
          <h1 className="truncate text-base font-semibold text-ink md:text-lg">{title}</h1>
        </div>

        <div className="flex shrink-0 items-center gap-3 md:gap-5">
          {/* Real money is stated in words, not implied by a colour. */}
          {mode && (
            <div className="hidden text-right md:block">
              <div className="eyebrow">mode</div>
              <div
                className={`font-mono text-tiny font-medium uppercase tracking-wider ${
                  mode === 'live' ? 'text-below' : 'text-ink-2'
                }`}
              >
                {mode === 'live' ? 'live · real money' : 'paper'}
              </div>
            </div>
          )}

          {halted && (
            <div className="hidden text-right md:block">
              <div className="eyebrow">state</div>
              <div className="font-mono text-tiny font-medium uppercase tracking-wider text-fail">
                halted
              </div>
            </div>
          )}

          {/* The age of the newest DECISION, not of the fetch. Held back to
              `lg`: at 768 the four slots together crushed the page title to a
              single letter and an ellipsis. */}
          <div className="hidden text-right lg:block">
            <div className="eyebrow">last decision</div>
            <div
              className={`font-mono text-tiny tabular-nums ${
                stale ? 'text-fail' : 'text-ink-2'
              }`}
            >
              {ago(age)}
            </div>
          </div>

          <div className="text-right">
            {/* Every time here is US Eastern, the venue's own day boundary.
                Said because a clock with no zone is read as the viewer's. */}
            <div className="eyebrow">next settlement · ET</div>
            <div className="font-mono text-mid font-medium tabular-nums text-ink">
              {String(Math.floor(secondsLeft / 60)).padStart(2, '0')}:
              {String(secondsLeft % 60).padStart(2, '0')}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
