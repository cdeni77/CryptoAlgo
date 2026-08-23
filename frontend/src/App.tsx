/** Routing, and the shell.
 *
 * No react-router. Routing is manual through `window.history.pushState`, and the
 * two maps below are what keep it honest: `RoutePath` derives from `ROUTES`, and
 * `PAGES` is a `Record<RoutePath, ComponentType>` — so adding a route without a
 * component is a `tsc` error rather than a blank screen. The render used to be a
 * chain of `route === '/x' && <XPage />` and this file used to claim that was
 * exhaustive. It was not.
 */
import { useCallback, useEffect, useState } from 'react';
import type { ComponentType } from 'react';
import { Rail } from './components/Rail';
import { AccountPage } from './pages/AccountPage';
import { CalibrationPage } from './pages/CalibrationPage';
import { DecisionsPage } from './pages/DecisionsPage';
import { LivePage } from './pages/LivePage';
import { ModelPage } from './pages/ModelPage';

export const ROUTES = {
  '/': { label: 'Live', hint: 'the barrier state now, per symbol' },
  '/decisions': { label: 'Decisions', hint: 'every decision point, traded or refused' },
  '/calibration': { label: 'Calibration', hint: 'is the forecast honest about its own confidence' },
  '/model': { label: 'Model', hint: 'gates, and every candidate ever evaluated' },
  '/account': { label: 'Account', hint: 'equity over time and settled positions' },
} as const;

export type RoutePath = keyof typeof ROUTES;

/** A Record, not a lookup with a fallback. A route with no component will not
 *  compile, which is the whole reason this is typed this way. */
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

  // The header clock ticks on the quarter-hour grid the whole system runs on, so
  // a stale screen is visible as a stale clock rather than as plausible numbers.
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
          hint: meta.hint,
        }))}
        current={route}
        onNavigate={navigate}
        lastUpdated={null}
      />

      <main className="min-w-0 flex-1">
        <header className="sticky top-0 z-10 flex items-center justify-between gap-6 border-b border-rule bg-paper/95 px-8 py-4 backdrop-blur-sm">
          <div>
            <div className="eyebrow">{ROUTES[route].hint}</div>
            <h1 className="text-lg font-semibold text-ink">{ROUTES[route].label}</h1>
          </div>
          <div className="text-right">
            <div className="eyebrow">next settlement</div>
            <div className="font-mono text-mid font-medium tabular-nums text-ink">
              {String(Math.floor(secondsLeft / 60)).padStart(2, '0')}:
              {String(secondsLeft % 60).padStart(2, '0')}
            </div>
          </div>
        </header>

        <div className="mx-auto max-w-shell px-8 py-8">
          <Page />
        </div>
      </main>
    </div>
  );
}
