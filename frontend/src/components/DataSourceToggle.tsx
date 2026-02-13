import { DataSource } from '../types';

interface DataSourceToggleProps {
  source: DataSource;
  onChange: (source: DataSource) => void;
  compact?: boolean;
}

export default function DataSourceToggle({ source, onChange, compact = false }: DataSourceToggleProps) {
  return (
    <div
      className="flex items-center gap-1 p-0.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        onClick={() => onChange('spot')}
        className={`
          ${compact ? 'px-3 py-1 text-xs' : 'px-4 py-1.5 text-sm'}
          rounded-md font-medium font-mono-trade transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan)]/40
          ${source === 'spot'
            ? 'bg-[var(--accent-cyan)] text-[var(--bg-primary)] shadow-md shadow-cyan-500/20'
            : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
          }
        `}
      >
        SPOT
      </button>
      <button
        onClick={() => onChange('cde')}
        className={`
          ${compact ? 'px-3 py-1 text-xs' : 'px-4 py-1.5 text-sm'}
          rounded-md font-medium font-mono-trade transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan)]/40
          ${source === 'cde'
            ? 'bg-[var(--accent-cyan)] text-[var(--bg-primary)] shadow-md shadow-cyan-500/20'
            : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
          }
        `}
      >
        CDE
      </button>
    </div>
  );
}