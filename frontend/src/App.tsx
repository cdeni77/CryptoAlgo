import { useCallback, useEffect, useState } from 'react';

import { getPaperConfig } from './api/paperApi';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import ModelPage from './pages/ModelPage';
import ResearchPage from './pages/ResearchPage';
import TradingPage from './pages/TradingPage';

export const ROUTES = {
  '/': 'Dashboard',
  '/trading': 'Trading',
  '/research': 'Research',
  '/model': 'Model',
} as const;

export type RoutePath = keyof typeof ROUTES;

function isRoute(path: string): path is RoutePath {
  return path in ROUTES;
}

function currentRoute(): RoutePath {
  const path = window.location.pathname;
  return isRoute(path) ? path : '/';
}

export default function App() {
  const [route, setRoute] = useState<RoutePath>(currentRoute);
  const [utc, setUtc] = useState('');
  const [activeCoins, setActiveCoins] = useState<string[] | null>(null);
  const [activeCoinsError, setActiveCoinsError] = useState<string | null>(null);

  const navigate = useCallback((path: RoutePath) => {
    if (path === window.location.pathname) return;
    window.history.pushState(null, '', path);
    setRoute(path);
  }, []);

  // Without this, the browser's back button changed the URL and left the page
  // rendering the route it was already on — history entries that went nowhere.
  useEffect(() => {
    const onPopState = () => setRoute(currentRoute());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    document.title = `${ROUTES[route]} · CryptoAlgo`;
  }, [route]);

  // The engine's active-coin list, for the sidebar. The previous version
  // swallowed the error on the grounds that it is "just a label" — but the label
  // renders "Loading…" whenever the fetch fails, in the persistent chrome of
  // every page, so a dead backend was indistinguishable from a slow one forever.
  // It also bypassed usePolling and so kept polling a hidden tab.
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      getPaperConfig()
        .then((cfg) => {
          if (cancelled) return;
          setActiveCoins(cfg.active_coins);
          setActiveCoinsError(null);
        })
        .catch((caught: unknown) => {
          if (cancelled) return;
          setActiveCoins(null);
          setActiveCoinsError(caught instanceof Error ? caught.message : String(caught));
        });

    let id: number | undefined;
    const start = () => {
      if (id === undefined) id = window.setInterval(load, 60_000);
    };
    const stop = () => {
      if (id !== undefined) {
        window.clearInterval(id);
        id = undefined;
      }
    };
    const onVisibilityChange = () => {
      if (document.hidden) {
        stop();
      } else {
        void load();
        start();
      }
    };

    load();
    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      cancelled = true;
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, []);

  // Ticking only while the tab is visible. A background tab re-rendering the
  // whole shell every second for a clock nobody is reading is pure waste.
  useEffect(() => {
    let id: number | undefined;
    const tick = () => setUtc(`${new Date().toUTCString().slice(17, 25)} UTC`);

    const start = () => {
      if (id === undefined) id = window.setInterval(tick, 1000);
    };
    const stop = () => {
      if (id !== undefined) window.clearInterval(id);
      id = undefined;
    };
    const onVisibilityChange = () => {
      if (document.hidden) {
        stop();
      } else {
        tick();
        start();
      }
    };

    tick();
    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-[#080c14] font-sans antialiased">
      <Sidebar route={route} navigate={navigate} activeCoins={activeCoins} activeCoinsError={activeCoinsError} />

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex flex-shrink-0 items-center justify-between border-b border-[rgba(56,189,248,0.08)] bg-[#0c1120] px-6 py-3.5">
          <span className="text-sm font-medium uppercase tracking-widest text-tx-secondary">
            {ROUTES[route]}
          </span>
          <span className="font-mono text-xs text-tx-muted">{utc}</span>
        </header>

        <main className="bg-grid flex-1 overflow-y-auto">
          {route === '/' && <DashboardPage />}
          {route === '/trading' && <TradingPage />}
          {route === '/research' && <ResearchPage />}
          {route === '/model' && <ModelPage />}
        </main>
      </div>
    </div>
  );
}
