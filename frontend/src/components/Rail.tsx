/** The navigation rail: a list of words, the active one marked by an accent bar.
 *
 * The footer shows the fee schedule and half-spread that every edge figure uses.
 */
import { clock } from '../lib/format';
import { Logo } from './Logo';

export interface RailProps {
  routes: { path: string; label: string; hint: string }[];
  current: string;
  onNavigate: (path: string) => void;
  lastUpdated: Date | null;
}

export function Rail({ routes, current, onNavigate, lastUpdated }: RailProps) {
  return (
    // Hidden below `md`, where `TabBar` takes over. 13.5rem of a 375px phone
    // left 159px for the page, which wrapped every heading to one word a line.
    <nav className="hidden w-rail shrink-0 flex-col border-r border-rule bg-surface md:flex">
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
            {/* `clock`, not `toLocaleTimeString(undefined)`: the latter is the
                VIEWER's zone, so this line disagreed with every other time on
                the page as soon as the dashboard was opened from elsewhere. */}
            {clock(lastUpdated.toISOString())} ET
          </div>
        )}
      </div>
    </nav>
  );
}
