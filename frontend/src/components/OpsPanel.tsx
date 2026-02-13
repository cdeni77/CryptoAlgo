import { useEffect, useMemo, useState } from 'react';
import { getOpsLogs, getOpsStatus, startPipeline, stopPipeline, triggerRetrain } from '../api/opsApi';
import { OpsLogEntry, OpsStatus } from '../types';

const fmt = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '—');

export default function OpsPanel() {
  const [status, setStatus] = useState<OpsStatus | null>(null);
  const [logs, setLogs] = useState<OpsLogEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [s, l] = await Promise.all([getOpsStatus(), getOpsLogs(120)]);
      setStatus(s);
      setLogs(l.entries);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, []);

  const onAction = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const metrics = useMemo(() => {
    const m = status?.metrics ?? {};
    return {
      auc: m.AUC,
      sharpe: m.OOS_SHARPE,
      trades: m.TRADE_COUNT,
    };
  }, [status]);

  return (
    <section className="glass-card rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm uppercase tracking-wider font-mono-trade text-[var(--text-muted)]">Operations</h2>
        <div className="flex items-center gap-2">
          <button
            disabled={busy}
            onClick={() => onAction(startPipeline)}
            className="px-3 py-1.5 rounded-lg text-xs font-mono-trade bg-emerald-500/10 text-[var(--accent-emerald)] border border-emerald-500/30 disabled:opacity-50"
          >
            Start Pipeline
          </button>
          <button
            disabled={busy}
            onClick={() => onAction(stopPipeline)}
            className="px-3 py-1.5 rounded-lg text-xs font-mono-trade bg-rose-500/10 text-[var(--accent-rose)] border border-rose-500/30 disabled:opacity-50"
          >
            Stop Pipeline
          </button>
          <button
            disabled={busy}
            onClick={() => onAction(triggerRetrain)}
            className="px-3 py-1.5 rounded-lg text-xs font-mono-trade bg-cyan-500/10 text-[var(--accent-cyan)] border border-cyan-500/30 disabled:opacity-50"
          >
            Retrain
          </button>
        </div>
      </div>

      {error && <p className="text-xs text-[var(--accent-rose)]">{error}</p>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="rounded-lg bg-[var(--bg-secondary)]/60 p-3 space-y-1 text-xs font-mono-trade">
          <p className="text-[var(--text-muted)]">Current training state</p>
          <p className="text-[var(--text-primary)]">Phase: {status?.phase ?? 'loading'}</p>
          <p className="text-[var(--text-secondary)]">Symbol: {status?.symbol ?? '—'}</p>
          <p className="text-[var(--text-secondary)]">Pipeline: {status?.pipeline_running ? 'running' : 'stopped'}</p>
          <p className="text-[var(--text-secondary)]">Training: {status?.training_running ? 'running' : 'idle'}</p>
          <p className="text-[var(--text-secondary)]">Last run: {fmt(status?.last_run_time ?? null)}</p>
          <p className="text-[var(--text-secondary)]">Next run: {fmt(status?.next_run_time ?? null)}</p>
        </div>

        <div className="rounded-lg bg-[var(--bg-secondary)]/60 p-3 space-y-1 text-xs font-mono-trade">
          <p className="text-[var(--text-muted)]">Last model metrics</p>
          <p className="text-[var(--text-primary)]">AUC: {metrics.auc?.toFixed(3) ?? '—'}</p>
          <p className="text-[var(--text-primary)]">OOS Sharpe: {metrics.sharpe?.toFixed(3) ?? '—'}</p>
          <p className="text-[var(--text-primary)]">Trade count: {metrics.trades ?? '—'}</p>
          <p className="text-[var(--text-secondary)] mt-2">Log file: {status?.log_file ?? '—'}</p>
        </div>
      </div>

      <div className="rounded-lg bg-[var(--bg-secondary)]/50 border border-[var(--border-subtle)]">
        <div className="px-3 py-2 border-b border-[var(--border-subtle)] text-xs font-mono-trade text-[var(--text-muted)]">
          Live log stream
        </div>
        <div className="h-56 overflow-auto p-3 space-y-1 font-mono-trade text-xs">
          {logs.length === 0 ? (
            <p className="text-[var(--text-muted)]">No logs yet.</p>
          ) : (
            logs.slice().reverse().map((line, idx) => (
              <p key={`${line.raw}-${idx}`} className="text-[var(--text-secondary)] break-all">
                {line.raw}
              </p>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
