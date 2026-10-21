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

        {/* The quote. A caret, so it reads as a position on the axis rather
            than as a competing quantity. */}
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
