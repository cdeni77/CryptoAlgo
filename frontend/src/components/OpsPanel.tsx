import { useEffect, useMemo, useState } from 'react';
import {
  getOpsLogs,
  getOpsStatus,
  launchParallel,
  ParallelLaunchOptions,
  RetrainOptions,
  startPipeline,
  stopPipeline,
  trainScratch,
  TrainScratchOptions,
  triggerRetrain,
} from '../api/opsApi';
import { OpsLogEntry, OpsStatus } from '../types';

const fmt = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '—');

const defaultParallel: ParallelLaunchOptions = {
  trials: 200,
  jobs: 16,
  coins: 'BTC,ETH,SOL,XRP,DOGE',
  plateau_patience: 80,
  plateau_min_delta: 0.02,
  plateau_warmup: 40,
};

const defaultScratch: TrainScratchOptions = {
  backfill_days: 30,
  include_oi: true,
  debug: false,
  threshold: 0.74,
  min_auc: 0.54,
  leverage: 4,
  exclude_symbols: 'BIP,DOP',
};

const defaultRetrain: RetrainOptions = {
  train_window_days: 90,
  retrain_every_days: 7,
  debug: false,
};

export default function OpsPanel() {
  const [status, setStatus] = useState<OpsStatus | null>(null);
  const [logs, setLogs] = useState<OpsLogEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parallelForm, setParallelForm] = useState<ParallelLaunchOptions>(defaultParallel);
  const [scratchForm, setScratchForm] = useState<TrainScratchOptions>(defaultScratch);
  const [retrainForm, setRetrainForm] = useState<RetrainOptions>(defaultRetrain);

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
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <h2 className="text-sm uppercase tracking-wider font-mono-trade text-[var(--text-muted)]">Operations</h2>
        <div className="flex items-center gap-2 flex-wrap">
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
            onClick={() => onAction(() => launchParallel(parallelForm))}
            className="px-3 py-1.5 rounded-lg text-xs font-mono-trade bg-amber-500/10 text-[var(--accent-amber)] border border-amber-500/30 disabled:opacity-50"
          >
            Parallel Launch
          </button>
          <button
            disabled={busy}
            onClick={() => onAction(() => trainScratch(scratchForm))}
            className="px-3 py-1.5 rounded-lg text-xs font-mono-trade bg-cyan-500/10 text-[var(--accent-cyan)] border border-cyan-500/30 disabled:opacity-50"
          >
            Train Scratch
          </button>
          <button
            disabled={busy}
            onClick={() => onAction(() => triggerRetrain(retrainForm))}
            className="px-3 py-1.5 rounded-lg text-xs font-mono-trade bg-sky-500/10 text-sky-300 border border-sky-500/30 disabled:opacity-50"
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
          <p className="text-[var(--text-secondary)]">Parallel: {status?.parallel_running ? 'running' : 'idle'}</p>
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

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono-trade">
        <div className="rounded-lg bg-[var(--bg-secondary)]/60 p-3 space-y-2">
          <p className="text-[var(--text-muted)]">Parallel params</p>
          <input className="w-full bg-[var(--bg-primary)] rounded px-2 py-1" value={parallelForm.coins} onChange={(e) => setParallelForm((p) => ({ ...p, coins: e.target.value }))} />
          <div className="grid grid-cols-2 gap-2">
            <input type="number" className="bg-[var(--bg-primary)] rounded px-2 py-1" value={parallelForm.trials} onChange={(e) => setParallelForm((p) => ({ ...p, trials: Number(e.target.value) }))} />
            <input type="number" className="bg-[var(--bg-primary)] rounded px-2 py-1" value={parallelForm.jobs} onChange={(e) => setParallelForm((p) => ({ ...p, jobs: Number(e.target.value) }))} />
          </div>
        </div>

        <div className="rounded-lg bg-[var(--bg-secondary)]/60 p-3 space-y-2">
          <p className="text-[var(--text-muted)]">Scratch params</p>
          <div className="grid grid-cols-2 gap-2">
            <input type="number" className="bg-[var(--bg-primary)] rounded px-2 py-1" value={scratchForm.backfill_days} onChange={(e) => setScratchForm((p) => ({ ...p, backfill_days: Number(e.target.value) }))} />
            <input type="number" step="0.01" className="bg-[var(--bg-primary)] rounded px-2 py-1" value={scratchForm.threshold} onChange={(e) => setScratchForm((p) => ({ ...p, threshold: Number(e.target.value) }))} />
          </div>
          <input className="w-full bg-[var(--bg-primary)] rounded px-2 py-1" value={scratchForm.exclude_symbols} onChange={(e) => setScratchForm((p) => ({ ...p, exclude_symbols: e.target.value }))} />
        </div>

        <div className="rounded-lg bg-[var(--bg-secondary)]/60 p-3 space-y-2">
          <p className="text-[var(--text-muted)]">Retrain params</p>
          <div className="grid grid-cols-2 gap-2">
            <input type="number" className="bg-[var(--bg-primary)] rounded px-2 py-1" value={retrainForm.train_window_days} onChange={(e) => setRetrainForm((p) => ({ ...p, train_window_days: Number(e.target.value) }))} />
            <input type="number" className="bg-[var(--bg-primary)] rounded px-2 py-1" value={retrainForm.retrain_every_days} onChange={(e) => setRetrainForm((p) => ({ ...p, retrain_every_days: Number(e.target.value) }))} />
          </div>
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
