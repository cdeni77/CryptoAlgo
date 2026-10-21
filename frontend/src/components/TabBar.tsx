/** The mobile navigation, which is a bottom tab bar rather than a drawer.
 *
 * `Rail` is 13.5rem wide and does not collapse, so on a 375px phone it took
 * 216px and left 159px for the page — every heading wrapped to one word per
 * line and the right-hand column ran off the screen. It is hidden below `md`
 * and this takes over.
 *
 * A bar, not a hamburger: there are five destinations and they all fit, so a
 * drawer would hide a complete menu behind a tap for no gain. Fixed to the
 * bottom because this is read one-handed on a phone, and the bottom of the
 * screen is the part a thumb reaches.
 *
 * `pb-[env(safe-area-inset-bottom)]` is what keeps the labels off the iPhone
 * home indicator; it needs `viewport-fit=cover` on the viewport meta to be
 * anything but zero.
 */
export interface TabBarProps {
  routes: { path: string; label: string; short?: string; hint: string }[];
  current: string;
  onNavigate: (path: string) => void;
}

export function TabBar({ routes, current, onNavigate }: TabBarProps) {
  return (
    // Fully opaque, not `bg-surface/95`: at 95% the page scrolled visibly
    // behind the labels and the coloured chips read as part of the nav.
    <nav
      className="fixed inset-x-0 bottom-0 z-20 border-t border-rule bg-surface pb-[env(safe-area-inset-bottom)] md:hidden"
      aria-label="Sections"
    >
      <ul className="flex">
        {routes.map((route) => {
          const active = route.path === current;
          return (
            <li key={route.path} className="min-w-0 flex-1">
              <button
                type="button"
                onClick={() => onNavigate(route.path)}
                aria-current={active ? 'page' : undefined}
                title={route.hint}
                className={[
                  // 44px minimum: anything smaller is below every platform's
                  // touch-target guidance and is genuinely hard to hit.
                  'relative flex h-14 w-full flex-col items-center justify-center gap-1 px-1',
                  'text-[11px] leading-none transition-colors',
                  active ? 'font-semibold text-ink' : 'text-ink-3',
                ].join(' ')}
              >
                {/* The active marker is on the TOP edge here, where the rail
                    puts it on the leading edge — same structural accent, moved
                    to the side of the control that faces the content. */}
                {active && (
                  <span className="absolute inset-x-3 top-0 h-0.5 bg-accent" aria-hidden />
                )}
                <span className="w-full truncate text-center">
                  {route.short ?? route.label}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
