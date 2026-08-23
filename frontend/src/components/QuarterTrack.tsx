/** The fifteen-minute band: what has happened, what is left, and when it settles.
 *
 * This is the app's primary structural device, and it is information rather than
 * ornament. A Kalshi crypto up/down market opens on a quarter-hour boundary and
 * settles on the next one, so *how much of the window is already gone* is half
 * of what determines the probability — the other half being volatility. A number
 * ("9 of 15 minutes") states that; a band shows it, and shows where the four
 * decision offsets sit relative to it, which a number cannot.
 *
 * Elapsed minutes are filled. The decision offsets carry a taller tick. The
 * current position is a firm rule. Remaining minutes are empty, and their count
 * is the thing the barrier divides by.
 */
interface QuarterTrackProps {
  windowMinutes: number;
  elapsed: number;
  offsets: number[];
  /** Seconds until settlement, when the caller is tracking a live clock. */
  secondsToSettle?: number | null;
  compact?: boolean;
}

export function QuarterTrack({
  windowMinutes,
  elapsed,
  offsets,
  secondsToSettle,
  compact = false,
}: QuarterTrackProps) {
  const cells = Array.from({ length: windowMinutes }, (_, i) => i);
  const offsetSet = new Set(offsets);
  const height = compact ? 'h-3' : 'h-5';

  return (
    <div className="flex items-center gap-3">
      <div className={`flex flex-1 items-stretch gap-px ${height}`} aria-hidden>
        {cells.map((minute) => {
          const past = minute < elapsed;
          const isOffset = offsetSet.has(minute + 1);
          return (
            <div
              key={minute}
              className={[
                'relative flex-1',
                past ? 'bg-ink-3' : 'bg-sunken',
                minute + 1 === elapsed ? 'outline outline-1 outline-accent' : '',
              ].join(' ')}
            >
              {isOffset && (
                <span
                  className={`absolute -bottom-1 left-1/2 h-1 w-px -translate-x-1/2 ${
                    past ? 'bg-ink-2' : 'bg-rule-firm'
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-micro text-ink-3">
        {secondsToSettle == null
          ? `${windowMinutes - elapsed}m left`
          : formatCountdown(secondsToSettle)}
      </span>
    </div>
  );
}

function formatCountdown(seconds: number): string {
  if (seconds <= 0) return 'settling';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
