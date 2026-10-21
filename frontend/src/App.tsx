/** Routing, and the shell.
 *
 * Routing uses `window.history.pushState`, without react-router. `RoutePath`
 * derives from `ROUTES`, and `PAGES` is a `Record<RoutePath, ComponentType>`, so
 * a route without a component fails to compile.
 */
import { useCallback, useEffect, useState } from 'react';
import type { ComponentType } from 'react';
import { Rail } from './components/Rail';
import { TabBar } from './components/TabBar';
import { AccountPage } from './pages/AccountPage';
import { CalibrationPage } from './pages/CalibrationPage';
import { DecisionsPage } from './pages/DecisionsPage';
import { LivePage } from './pages/LivePage';
import { ModelPage } from './pages/ModelPage';

// `short` is only for the mobile tab bar, where five labels share 375px.
// Carried here rather than mapped inside `TabBar` so a new route cannot get a
// tab with no label and no compile error.
export const ROUTES = {
  '/': { label: 'Live', short: 'Live', hint: 'the barrier state now, per symbol' },
  '/decisions': { label: 'Decisions', short: 'Decide', hint: 'every decision point, traded or refused' },
  '/calibration': { label: 'Calibration', short: 'Calib', hint: 'is the forecast honest about its own confidence' },
  '/model': { label: 'Model', short: 'Model', hint: 'gates, and every candidate ever evaluated' },
  '/account': { label: 'Account', short: 'Acct', hint: 'what the account is worth, and how it got there' },
} as const;

export type RoutePath = keyof typeof ROUTES;

/** A Record, not a lookup with a fallback, so every route must have a page. */
const PAGES: Record<RoutePath, ComponentType> = {
  '/': LivePage,
  '/decisions': DecisionsPage,
  '/calibration': CalibrationPage,
  '/model': ModelPage,
  '/account': AccountPage,
};

function currentPath(): RoutePath {
  const path = window.location.pathname;
  return (path in ROUTES ? path : '/') as RoutePath;
}

export default function App() {
  const [route, setRoute] = useState<RoutePath>(currentPath);
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const onPop = () => setRoute(currentPath());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  // The header clock counts down to the next quarter-hour settlement.
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const navigate = useCallback((path: string) => {
    const next = (path in ROUTES ? path : '/') as RoutePath;
    window.history.pushState({}, '', next);
    setRoute(next);
  }, []);

  const Page = PAGES[route];
  const secondsIntoWindow = (now.getUTCMinutes() % 15) * 60 + now.getUTCSeconds();
  const secondsLeft = 15 * 60 - secondsIntoWindow;

  return (
    <div className="flex min-h-screen bg-paper">
      <Rail
        routes={Object.entries(ROUTES).map(([path, meta]) => ({
          path,
          label: meta.label,
          short: meta.short,
          hint: meta.hint,
        }))}
        current={route}
        onNavigate={navigate}
        lastUpdated={null}
      />

      <main className="min-w-0 flex-1">
        <header className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-rule bg-paper/95 px-4 py-3 backdrop-blur-sm md:gap-6 md:px-8 md:py-4">
          {/* `min-w-0` on the flex child, or the long hint refuses to shrink and
              pushes the clock off the right edge instead of truncating. */}
          <div className="min-w-0">
            {/* The hint is a full sentence; on a phone it is one line or it is
                three. Hidden rather than wrapped — the h1 already names the
                page, and the hint is still the button's title in the rail. */}
            <div className="eyebrow hidden truncate sm:block">{ROUTES[route].hint}</div>
            <h1 className="truncate text-base font-semibold text-ink md:text-lg">
              {ROUTES[route].label}
            </h1>
          </div>
          <div className="shrink-0 text-right">
            {/* Every time on this dashboard is US Eastern -- the venue's own
                day boundary, and the zone the account reconciles against. Said
                here because a clock with no zone is read as the viewer's. */}
            <div className="eyebrow">next settlement · ET</div>
            <div className="font-mono text-mid font-medium tabular-nums text-ink">
              {String(Math.floor(secondsLeft / 60)).padStart(2, '0')}:
              {String(secondsLeft % 60).padStart(2, '0')}
            </div>
          </div>
        </header>

        {/* `pb-20` clears the fixed tab bar, which would otherwise sit on top of
            the last row of every page. */}
        <div className="mx-auto max-w-shell px-4 pb-20 pt-5 md:px-8 md:pb-8 md:pt-8">
          <Page />
        </div>
      </main>

      <TabBar
        routes={Object.entries(ROUTES).map(([path, meta]) => ({
          path,
          label: meta.label,
          short: meta.short,
          hint: meta.hint,
        }))}
        current={route}
        onNavigate={navigate}
      />
    </div>
  );
}
