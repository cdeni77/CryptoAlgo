/** The mark: a window, its strike, and price sitting above it.
 *
 * The square is the fifteen-minute window, the rule is the strike, and the
 * block, in the `above` pole colour, is price late in the window.
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
