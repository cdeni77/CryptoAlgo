/** Routing, and the shell.
 *
 * Routing uses `window.history.pushState`, without react-router. `RoutePath`
 * derives from `ROUTES`, and `PAGES` is a `Record<RoutePath, ComponentType>`, so
 * a route without a component fails to compile.
 */
import { useCallback, useEffect, useState } from 'react';
import type { ComponentType } from 'react';
import { AlertStrip } from './app/AlertStrip';
import { StatusHeader } from './app/StatusHeader';
import { fetchCollectionHealth, fetchLive } from './api/serving';
import { usePolling } from './hooks/usePolling';
import { deriveAlerts } from './lib/status';
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

  // **Polled at the shell, not per page.** The header and the alert strip are
  // on every route, and each page fetching its own copy made the same fact
  // arrive at a different second on each screen. Five seconds: these drive a
  // banner, not a chart, and `/collection` is the heaviest read on the API.
  const live = usePolling(fetchLive, 5_000);
  const collection = usePolling(fetchCollectionHealth, 15_000);

  const account = live.data?.account ?? null;
  // The newest decision across symbols. Its AGE is the liveness signal; the
  // fetch succeeding is not.
  const decisionTimes = (live.data?.windows ?? [])
    .map((w) => w.decision_time)
    .filter((t): t is string => Boolean(t))
    .sort();
  const lastDecisionAt = decisionTimes.length
    ? decisionTimes[decisionTimes.length - 1]
    : null;
  const alerts = deriveAlerts(account, collection.data ?? null, lastDecisionAt);

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
        <StatusHeader
          title={ROUTES[route].label}
          hint={ROUTES[route].hint}
          mode={account?.mode ?? null}
          halted={Boolean(account?.halted)}
          lastDecisionAt={lastDecisionAt}
          secondsLeft={secondsLeft}
        />

        <AlertStrip alerts={alerts} />

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
