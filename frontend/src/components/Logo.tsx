/** The mark: a window, its strike, and price sitting above it.
 *
 * Every element means something, which is the test for whether a mark is a mark
 * or a decoration. The square is the fifteen-minute window. The rule across it
 * at 62% height is the strike — the price the venue recorded when the window
 * opened. The filled block above the rule and to the right is where price is
 * now: above the strike, late in the window. That is the system's whole subject
 * in three shapes, and it survives being 16 pixels wide in a browser tab.
 *
 * The block takes the `above` pole colour, so the mark is literally reporting a
 * settle-up state rather than being tinted for looks.
 */
export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role="img"
      aria-label="Quarter"
      className="shrink-0"
    >
      <rect
        x="2.5"
        y="2.5"
        width="27"
        height="27"
        rx="1.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
      />
      <path d="M7 20 H25" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square" />
      <rect x="19" y="9" width="6" height="6" className="fill-above" />
    </svg>
  );
}

export function Logo() {
  return (
    <div className="flex items-baseline gap-2.5 text-ink">
      <span className="translate-y-[3px]">
        <Mark />
      </span>
      <span className="font-sans text-mid font-bold uppercase tracking-[0.2em]">
        Quarter
      </span>
    </div>
  );
}
