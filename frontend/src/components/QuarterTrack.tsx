/** The fifteen-minute band: what has happened, what is left, and when it settles.
 *
 * Elapsed minutes are filled, decision offsets carry a tick, the current minute
 * is outlined, and remaining minutes are empty.
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
