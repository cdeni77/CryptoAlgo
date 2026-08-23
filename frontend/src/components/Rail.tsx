/** The navigation rail.
 *
 * A list of words, not a column of icons. Five destinations do not need
 * pictograms, and a pictogram beside a word is decoration twice. The active item
 * is marked by a 2px accent bar on its leading edge and by ink-weight text —
 * structural accent, never a directional colour.
 *
 * The footer carries the fee schedule, because it is the denominator of every
 * number on every page and it is an assumption rather than a measurement.
 */
import { Logo } from './Logo';

export interface RailProps {
  routes: { path: string; label: string; hint: string }[];
  current: string;
  onNavigate: (path: string) => void;
  lastUpdated: Date | null;
}

export function Rail({ routes, current, onNavigate, lastUpdated }: RailProps) {
  return (
    <nav className="flex w-rail shrink-0 flex-col border-r border-rule bg-surface">
      <div className="border-b border-rule px-5 py-5">
        <Logo />
        <p className="mt-2 font-mono text-micro leading-4 text-ink-3">
          barrier probability
          <br />
          15-minute binaries
        </p>
      </div>

      <ul className="flex-1 py-2">
        {routes.map((route) => {
          const active = route.path === current;
          return (
            <li key={route.path}>
              <button
                type="button"
                onClick={() => onNavigate(route.path)}
                aria-current={active ? 'page' : undefined}
                title={route.hint}
                className={[
                  'relative w-full px-5 py-2 text-left text-base transition-colors',
                  active
                    ? 'font-medium text-ink'
                    : 'text-ink-2 hover:bg-sunken hover:text-ink',
                ].join(' ')}
              >
                {active && (
                  <span className="absolute inset-y-1 left-0 w-0.5 bg-accent" aria-hidden />
                )}
                {route.label}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="border-t border-rule px-5 py-4 font-mono text-micro leading-4 text-ink-3">
        <div>fee 0.07·p(1−p)</div>
        <div>+0.5¢ half-spread</div>
        <div className="mt-2 text-ink-3/70">spread measured; depth read from the book</div>
        {lastUpdated && (
          <div className="mt-3 border-t border-rule pt-3">
            {lastUpdated.toLocaleTimeString(undefined, { hour12: false })}
          </div>
        )}
      </div>
    </nav>
  );
}
