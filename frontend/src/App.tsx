import { useEffect, useState } from 'react';
import StrategyLabPage from './pages/StrategyLabPage';
import TradingTerminalPage from './pages/TradingTerminalPage';

type RoutePath = '/' | '/strategy';

function normalizePath(pathname: string): RoutePath {
  return pathname.startsWith('/strategy') ? '/strategy' : '/';
}

function NavButton({ href, active, label, onNavigate }: { href: RoutePath; active: boolean; label: string; onNavigate: (path: RoutePath) => void }) {
  return (
    <button
      onClick={() => onNavigate(href)}
      className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
        active
          ? 'border-[var(--border-accent)] text-[var(--accent-cyan)] bg-[var(--bg-elevated)]'
          : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
      }`}
    >
      {label}
    </button>
  );
}

function TopNavigation({ current, onNavigate }: { current: RoutePath; onNavigate: (path: RoutePath) => void }) {
  return (
    <div className="max-w-[1400px] mx-auto px-5 pt-4 pb-1 flex items-center gap-2">
      <NavButton href="/" active={current === '/'} label="Trading Terminal" onNavigate={onNavigate} />
      <NavButton href="/strategy" active={current === '/strategy'} label="Strategy Lab" onNavigate={onNavigate} />
    </div>
  );
}

function App() {
  const [route, setRoute] = useState<RoutePath>(normalizePath(window.location.pathname));

  useEffect(() => {
    const updateRoute = () => setRoute(normalizePath(window.location.pathname));
    window.addEventListener('popstate', updateRoute);
    return () => window.removeEventListener('popstate', updateRoute);
  }, []);

  const navigate = (path: RoutePath) => {
    if (route === path) return;
    window.history.pushState({}, '', path);
    setRoute(path);
  };

  return (
    <div className="min-h-screen bg-grid font-sans antialiased">
      <TopNavigation current={route} onNavigate={navigate} />
      {route === '/' ? <TradingTerminalPage /> : <StrategyLabPage />}
    </div>
  );
}

export default App;
