import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import TradingPage from './pages/TradingPage';
import ResearchPage from './pages/ResearchPage';
import { getPaperConfig } from './api/paperApi';

export type RoutePath = '/' | '/trading' | '/research';

function getInitialRoute(): RoutePath {
  const p = window.location.pathname as RoutePath;
  return ['/', '/trading', '/research'].includes(p) ? p : '/';
}

export default function App() {
  const [route, setRoute] = useState<RoutePath>(getInitialRoute);
  const [utc, setUtc] = useState('');
  const [activeCoins, setActiveCoins] = useState<string[]>([]);

  useEffect(() => {
    const load = () => getPaperConfig().then(cfg => setActiveCoins(cfg.active_coins)).catch(() => {});
    load();
    const id = setInterval(load, 60000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setUtc(`${now.toUTCString().slice(17, 25)} UTC`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  function navigate(path: RoutePath) {
    window.history.pushState(null, '', path);
    setRoute(path);
  }

  const pageTitle = route === '/' ? 'Dashboard' : route === '/trading' ? 'Trading' : 'Research Lab';

  return (
    <div className="flex h-screen overflow-hidden bg-[#080c14] font-sans antialiased">
      <Sidebar route={route} navigate={navigate} activeCoins={activeCoins} />

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <header className="flex-shrink-0 flex items-center justify-between px-6 py-3.5 border-b border-[rgba(56,189,248,0.08)] bg-[#0c1120]">
          <span className="text-tx-secondary text-sm font-medium tracking-widest uppercase">{pageTitle}</span>
          <span className="font-mono text-tx-muted text-xs">{utc}</span>
        </header>

        <main className="flex-1 overflow-y-auto bg-grid">
          {route === '/'        && <DashboardPage />}
          {route === '/trading' && <TradingPage />}
          {route === '/research'&& <ResearchPage />}
        </main>
      </div>
    </div>
  );
}
