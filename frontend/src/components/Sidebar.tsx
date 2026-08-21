import { RoutePath } from '../App';

const GRID_ICON = (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM14 11a1 1 0 011 1v1h1a1 1 0 110 2h-1v1a1 1 0 11-2 0v-1h-1a1 1 0 110-2h1v-1a1 1 0 011-1z"/>
  </svg>
);

const CHART_ICON = (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>
  </svg>
);

const FLASK_ICON = (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path fillRule="evenodd" d="M7 2a1 1 0 00-.707 1.707L7 4.414v3.758a1 1 0 01-.293.707l-4 4C1.1 14.584 2.022 17 4.08 17h11.84c2.058 0 2.98-2.416 1.373-3.793l-4-3.586A1 1 0 0113 9.172V5.414l.707-.707A1 1 0 0013 3H7zm2 6.172V5h2v3.172a3 3 0 00.879 2.12l1.027.95A4 4 0 0115 17H5a4 4 0 012.094-3.563l1.027-.95A3 3 0 009 8.172z" clipRule="evenodd"/>
  </svg>
);

const SHIELD_ICON = (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path fillRule="evenodd" d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.589 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75zm4.196 5.954a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd"/>
  </svg>
);

const NAV_ITEMS: { path: RoutePath; label: string; icon: JSX.Element }[] = [
  { path: '/',         label: 'Dashboard', icon: GRID_ICON   },
  { path: '/trading',  label: 'Trading',   icon: CHART_ICON  },
  { path: '/research', label: 'Research',  icon: FLASK_ICON  },
  // The gates screen: what is live, why it was allowed to be, and what else was
  // tried. Given a shield rather than a chart icon because the decision it shows
  // is a refusal by default.
  { path: '/model',    label: 'Model',     icon: SHIELD_ICON },
];

interface Props {
  route: RoutePath;
  navigate: (path: RoutePath) => void;
  activeCoins?: string[];
}

export default function Sidebar({ route, navigate, activeCoins }: Props) {
  return (
    <aside className="w-52 flex-shrink-0 flex flex-col bg-[#0c1120] border-r border-[rgba(56,189,248,0.08)]">
      {/* Brand */}
      <div className="px-5 py-5 border-b border-[rgba(56,189,248,0.08)]">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-accent-cyan/20 border border-accent-cyan/30 flex items-center justify-center">
            <span className="text-accent-cyan text-xs font-bold">CA</span>
          </div>
          <div>
            <div className="text-tx-primary text-sm font-semibold leading-tight">CryptoAlgo</div>
            <div className="text-tx-muted text-[10px] leading-tight">Paper Trading</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ path, label, icon }) => {
          const active = route === path;
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 ${
                active
                  ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20'
                  : 'text-tx-secondary hover:text-tx-primary hover:bg-[rgba(56,189,248,0.05)] border border-transparent'
              }`}
            >
              <span className={active ? 'text-accent-cyan' : 'text-tx-muted'}>{icon}</span>
              {label}
            </button>
          );
        })}
      </nav>

      {/* Footer status */}
      <div className="px-4 py-4 border-t border-[rgba(56,189,248,0.08)]">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-emerald animate-pulse" />
          <span className="text-tx-muted text-[11px]">Live</span>
        </div>
        <div className="text-tx-muted text-[10px] leading-relaxed">
          {activeCoins && activeCoins.length > 0 ? activeCoins.join(' · ') + ' active' : 'Loading…'}
        </div>
      </div>
    </aside>
  );
}
