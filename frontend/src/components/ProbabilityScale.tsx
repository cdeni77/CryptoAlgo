/** The single most important picture in the app: forecast against price.
 *
 * A binary's price *is* a probability, so the forecast and the quote live on one
 * dimensionless axis and the edge is the *distance between two marks*. Nothing
 * else in this interface makes the thesis legible at a glance: the bar is where
 * the model thinks the probability is, the caret is what the market charges, and
 * whether the caret sits inside or outside the bar is whether there is a trade.
 *
 * The axis is anchored at 50% because that is the structural midpoint of a
 * two-outcome market, and the bar grows from there toward whichever pole the
 * forecast favours. A left-anchored progress bar would imply 0 is the reference,
 * which is false — 0 and 1 are symmetric outcomes and 0.5 is the pivot.
 *
 * `breakEven` is drawn as a hairline rather than a second bar, because it is the
 * threshold the forecast has to clear rather than a competing estimate.
 */
interface ProbabilityScaleProps {
  /** Model probability that the window settles above its strike. */
  probability: number;
  /** The market's implied probability — the quote, on the same axis. */
  price?: number | null;
  /** Quote plus half-spread plus fee: what the forecast actually has to beat. */
  breakEven?: number | null;
  height?: number;
  showAxis?: boolean;
}

export function ProbabilityScale({
  probability,
  price,
  breakEven,
  height = 22,
  showAxis = true,
}: ProbabilityScaleProps) {
  const p = clamp(probability);
  const above = p >= 0.5;
  const left = above ? 50 : p * 100;
  const width = Math.abs(p - 0.5) * 100;

  return (
    <div className="w-full">
      <div
        className="relative w-full border border-rule bg-sunken"
        style={{ height }}
        role="img"
        aria-label={`Forecast ${(p * 100).toFixed(1)} percent${
          price != null ? `, market ${(price * 100).toFixed(0)} percent` : ''
        }`}
      >
        {/* The 50% pivot. */}
        <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-rule-firm" />

        {/* The forecast, growing from the pivot toward its pole. */}
        <span
          className={`absolute inset-y-0 ${above ? 'bg-above' : 'bg-below'}`}
          style={{ left: `${left}%`, width: `${width}%` }}
        />

        {/* Break-even: the hairline the forecast must clear. */}
        {breakEven != null && (
          <span
            className="absolute inset-y-0 w-px bg-ink-3"
            style={{ left: `${clamp(breakEven) * 100}%` }}
            title={`break-even ${(breakEven * 100).toFixed(2)}%`}
          />
        )}

        {/* The quote. A caret, so it reads as a position on the axis rather
            than as a competing quantity. */}
        {price != null && (
          <span
            className="absolute -top-px bottom-[-1px] w-0 -translate-x-1/2"
            style={{ left: `${clamp(price) * 100}%` }}
            title={`market ${(price * 100).toFixed(0)}c`}
          >
            <span className="absolute -top-1 left-0 h-[calc(100%+0.5rem)] w-0.5 -translate-x-1/2 bg-ink" />
          </span>
        )}
      </div>
      {showAxis && (
        <div className="mt-1 flex justify-between font-mono text-micro text-ink-3">
          <span>0</span>
          <span>50</span>
          <span>100</span>
        </div>
      )}
    </div>
  );
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}
