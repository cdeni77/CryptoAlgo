/** Forecast against price, on one probability axis.
 *
 * The bar is the forecast, growing from the 50% pivot toward the pole it
 * favours. The caret is the market price; the edge is the distance between
 * them. `breakEven` is a hairline: the threshold the forecast must clear.
 */
interface ProbabilityScaleProps {
  /** Model probability of the SIDE TAKEN — not of "up".
   *
   *  The controller reads `model_probability` side-relative for display, so a
   *  DOWN row arrives here as P(down). `price` must therefore also be
   *  side-relative: feeding it a raw P(up) put the bar and the caret on
   *  opposite axes for every down row, and a +0.51pp edge drew as a 59-point
   *  gap with the caret on the wrong side of the hairline. */
  probability: number;
  /** The market's implied probability for the same side — the venue's MID,
   *  not an executable price. `market_ask_up`/`market_ask_down` are what a
   *  trade would actually cost; this is the venue's belief. */
  price?: number | null;
  /** What the forecast has to clear. Since the maker switch this is the quote
   *  alone: `effective_cost == price` on every row since the epoch, because
   *  resting at the bid pays no fee and crosses no spread. It is NOT net of
   *  cost for the ~19% of orders that cross. */
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
            title={`market mid ${(price * 100).toFixed(1)}c — the venue's belief, not a tradeable price. The ask is what a trade would cost.`}
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
