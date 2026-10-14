/** Forecast against price, on one probability axis.
 *
 * The bar is the forecast, growing from the 50% pivot toward the pole it
 * favours. The caret is the market price; the edge is the distance between
 * them. `breakEven` is a hairline: the threshold the forecast must clear.
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

        {/* The quote, as a caret. */}
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
