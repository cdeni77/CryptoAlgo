import { useEffect, useMemo, useRef, useState } from 'react';
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
  trials: 250,
  jobs: 16,
  coins: 'BTC,ETH,SOL,XRP,DOGE',
  plateau_patience: 120,
  plateau_min_delta: 0.01,
  plateau_warmup: 80,
  holdout_days: 90,
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

type ActiveModal = 'parallel' | 'scratch' | 'retrain' | null;

/* ── Tiny labeled input ─────────────────────────────────────────── */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
      {children}
    </label>
  );
}

const inputCls =
  'w-full bg-[var(--bg-primary)] border border-[var(--border-subtle)] rounded-lg px-3 py-1.5 text-xs font-mono-trade text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]/50 focus:ring-1 focus:ring-[var(--accent-cyan)]/20 transition-colors';

/* ── Status pill ────────────────────────────────────────────────── */
function StatusDot({ active, label }: { active: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-mono-trade">
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          active ? 'bg-[var(--accent-emerald)] shadow-[0_0_6px_rgba(52,211,153,0.5)]' : 'bg-[var(--text-muted)]/40'
        }`}
      />
      <span className={active ? 'text-[var(--accent-emerald)]' : 'text-[var(--text-muted)]'}>{label}</span>
    </span>
  );
}

/* ══════════════════════════════════════════════════════════════════ */

export default function OpsPanel() {
  const [expanded, setExpanded] = useState(false);
  const [status, setStatus] = useState<OpsStatus | null>(null);
  const [logs, setLogs] = useState<OpsLogEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeModal, setActiveModal] = useState<ActiveModal>(null);

  const [parallelForm, setParallelForm] = useState<ParallelLaunchOptions>(defaultParallel);
  const [scratchForm, setScratchForm] = useState<TrainScratchOptions>(defaultScratch);
  const [retrainForm, setRetrainForm] = useState<RetrainOptions>(defaultRetrain);

  const logEndRef = useRef<HTMLDivElement>(null);

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

  // auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const onAction = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setActiveModal(null);
    try {
      await fn();
      setExpanded(true); // auto-expand to show logs
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const metrics = useMemo(() => {
    const m = status?.metrics ?? {};
    return { auc: m.AUC, sharpe: m.OOS_SHARPE, trades: m.TRADE_COUNT };
  }, [status]);

  const anyRunning =
    status?.pipeline_running || status?.training_running || status?.parallel_running;

  /* ── Collapsed summary bar ─────────────────────────────────────── */
  const summaryBar = (
    <div
      onClick={() => setExpanded(!expanded)}
      className="flex items-center justify-between cursor-pointer select-none group"
    >
      <div className="flex items-center gap-3">
        <h2 className="text-sm uppercase tracking-wider font-mono-trade text-[var(--text-muted)] group-hover:text-[var(--text-secondary)] transition-colors">
          Operations
        </h2>
        <div className="flex items-center gap-3">
          <StatusDot active={!!status?.pipeline_running} label="Pipeline" />
          <StatusDot active={!!status?.training_running} label="Training" />
          <StatusDot active={!!status?.parallel_running} label="Optuna" />
        </div>
      </div>
      <div className="flex items-center gap-3">
        {status?.phase && status.phase !== 'idle' && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] font-mono-trade border border-[var(--accent-cyan)]/20">
            {status.phase}
          </span>
        )}
        <svg
          className={`w-4 h-4 text-[var(--text-muted)] transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </div>
  );

  /* ── Action buttons ────────────────────────────────────────────── */
  const actions = (
    <div className="flex items-center gap-2 flex-wrap pt-3 border-t border-[var(--border-subtle)]">
      <button
        disabled={busy}
        onClick={() => onAction(startPipeline)}
        className="ops-btn bg-[var(--accent-emerald)]/8 text-[var(--accent-emerald)] border-[var(--accent-emerald)]/25 hover:bg-[var(--accent-emerald)]/15"
      >
        <span className="w-1.5 h-1.5 rounded-full bg-current" />
        Start Pipeline
      </button>
      <button
        disabled={busy}
        onClick={() => onAction(stopPipeline)}
        className="ops-btn bg-[var(--accent-rose)]/8 text-[var(--accent-rose)] border-[var(--accent-rose)]/25 hover:bg-[var(--accent-rose)]/15"
      >
        <span className="w-1.5 h-1.5 rounded-full bg-current" />
        Stop Pipeline
      </button>

      <div className="w-px h-5 bg-[var(--border-subtle)] mx-1" />

      <button
        disabled={busy}
        onClick={() => setActiveModal('parallel')}
        className="ops-btn bg-[var(--accent-cyan)]/8 text-[var(--accent-cyan)] border-[var(--accent-cyan)]/25 hover:bg-[var(--accent-cyan)]/15"
      >
        Parallel Optimize
      </button>
      <button
        disabled={busy}
        onClick={() => setActiveModal('scratch')}
        className="ops-btn bg-[var(--accent-cyan)]/8 text-[var(--accent-cyan)] border-[var(--accent-cyan)]/25 hover:bg-[var(--accent-cyan)]/15"
      >
        Train Scratch
      </button>
      <button
        disabled={busy}
        onClick={() => setActiveModal('retrain')}
        className="ops-btn bg-[var(--accent-cyan)]/8 text-[var(--accent-cyan)] border-[var(--accent-cyan)]/25 hover:bg-[var(--accent-cyan)]/15"
      >
        Retrain
      </button>
    </div>
  );

  /* ── Inline param drawer ───────────────────────────────────────── */
  const paramDrawer = activeModal && (
    <div className="mt-3 p-4 rounded-lg bg-[var(--bg-primary)] border border-[var(--accent-cyan)]/20 animate-in">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-mono-trade uppercase tracking-wider text-[var(--accent-cyan)]">
          {activeModal === 'parallel' && 'Parallel Optimization'}
          {activeModal === 'scratch' && 'Train From Scratch'}
          {activeModal === 'retrain' && 'Retrain Models'}
        </h3>
        <button
          onClick={() => setActiveModal(null)}
          className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* ── Parallel form ──────────────────────────────────────── */}
      {activeModal === 'parallel' && (
        <div className="space-y-3">
          <Field label="Coins">
            <input className={inputCls} value={parallelForm.coins} onChange={(e) => setParallelForm((p) => ({ ...p, coins: e.target.value }))} />
          </Field>
          <div className="grid grid-cols-3 gap-2">
            <Field label="Trials/coin">
              <input type="number" className={inputCls} value={parallelForm.trials} onChange={(e) => setParallelForm((p) => ({ ...p, trials: Number(e.target.value) }))} />
            </Field>
            <Field label="Workers">
              <input type="number" className={inputCls} value={parallelForm.jobs} onChange={(e) => setParallelForm((p) => ({ ...p, jobs: Number(e.target.value) }))} />
            </Field>
            <Field label="Holdout days">
              <input type="number" className={inputCls} value={parallelForm.holdout_days} onChange={(e) => setParallelForm((p) => ({ ...p, holdout_days: Number(e.target.value) }))} />
            </Field>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Field label="Patience">
              <input type="number" className={inputCls} value={parallelForm.plateau_patience} onChange={(e) => setParallelForm((p) => ({ ...p, plateau_patience: Number(e.target.value) }))} />
            </Field>
            <Field label="Min delta">
              <input type="number" step="0.001" className={inputCls} value={parallelForm.plateau_min_delta} onChange={(e) => setParallelForm((p) => ({ ...p, plateau_min_delta: Number(e.target.value) }))} />
            </Field>
            <Field label="Warmup">
              <input type="number" className={inputCls} value={parallelForm.plateau_warmup} onChange={(e) => setParallelForm((p) => ({ ...p, plateau_warmup: Number(e.target.value) }))} />
            </Field>
          </div>
          <button
            disabled={busy}
            onClick={() => onAction(() => launchParallel(parallelForm))}
            className="w-full mt-1 px-4 py-2 rounded-lg text-xs font-mono-trade font-medium bg-[var(--accent-cyan)] text-[var(--bg-primary)] hover:brightness-110 disabled:opacity-50 transition-all"
          >
            {busy ? 'Launching…' : 'Launch Optimization'}
          </button>
        </div>
      )}

      {/* ── Scratch form ───────────────────────────────────────── */}
      {activeModal === 'scratch' && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <Field label="Backfill days">
              <input type="number" className={inputCls} value={scratchForm.backfill_days} onChange={(e) => setScratchForm((p) => ({ ...p, backfill_days: Number(e.target.value) }))} />
            </Field>
            <Field label="Threshold">
              <input type="number" step="0.01" className={inputCls} value={scratchForm.threshold} onChange={(e) => setScratchForm((p) => ({ ...p, threshold: Number(e.target.value) }))} />
            </Field>
            <Field label="Min AUC">
              <input type="number" step="0.01" className={inputCls} value={scratchForm.min_auc} onChange={(e) => setScratchForm((p) => ({ ...p, min_auc: Number(e.target.value) }))} />
            </Field>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Field label="Leverage">
              <input type="number" className={inputCls} value={scratchForm.leverage} onChange={(e) => setScratchForm((p) => ({ ...p, leverage: Number(e.target.value) }))} />
            </Field>
            <Field label="Exclude">
              <input className={inputCls} value={scratchForm.exclude_symbols} onChange={(e) => setScratchForm((p) => ({ ...p, exclude_symbols: e.target.value }))} />
            </Field>
            <Field label="Options">
              <div className="flex items-center gap-3 h-full">
                <label className="flex items-center gap-1 text-[11px] text-[var(--text-secondary)] cursor-pointer">
                  <input type="checkbox" checked={scratchForm.include_oi} onChange={(e) => setScratchForm((p) => ({ ...p, include_oi: e.target.checked }))} className="accent-[var(--accent-cyan)]" />
                  OI
                </label>
                <label className="flex items-center gap-1 text-[11px] text-[var(--text-secondary)] cursor-pointer">
                  <input type="checkbox" checked={scratchForm.debug} onChange={(e) => setScratchForm((p) => ({ ...p, debug: e.target.checked }))} className="accent-[var(--accent-cyan)]" />
                  Debug
                </label>
              </div>
            </Field>
          </div>
          <button
            disabled={busy}
            onClick={() => onAction(() => trainScratch(scratchForm))}
            className="w-full mt-1 px-4 py-2 rounded-lg text-xs font-mono-trade font-medium bg-[var(--accent-cyan)] text-[var(--bg-primary)] hover:brightness-110 disabled:opacity-50 transition-all"
          >
            {busy ? 'Starting…' : 'Start Training'}
          </button>
        </div>
      )}

      {/* ── Retrain form ───────────────────────────────────────── */}
      {activeModal === 'retrain' && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <Field label="Window (days)">
              <input type="number" className={inputCls} value={retrainForm.train_window_days} onChange={(e) => setRetrainForm((p) => ({ ...p, train_window_days: Number(e.target.value) }))} />
            </Field>
            <Field label="Frequency (days)">
              <input type="number" className={inputCls} value={retrainForm.retrain_every_days} onChange={(e) => setRetrainForm((p) => ({ ...p, retrain_every_days: Number(e.target.value) }))} />
            </Field>
            <Field label="Options">
              <label className="flex items-center gap-1 text-[11px] text-[var(--text-secondary)] cursor-pointer h-full">
                <input type="checkbox" checked={retrainForm.debug} onChange={(e) => setRetrainForm((p) => ({ ...p, debug: e.target.checked }))} className="accent-[var(--accent-cyan)]" />
                Debug
              </label>
            </Field>
          </div>
          <button
            disabled={busy}
            onClick={() => onAction(() => triggerRetrain(retrainForm))}
            className="w-full mt-1 px-4 py-2 rounded-lg text-xs font-mono-trade font-medium bg-[var(--accent-cyan)] text-[var(--bg-primary)] hover:brightness-110 disabled:opacity-50 transition-all"
          >
            {busy ? 'Starting…' : 'Start Retrain'}
          </button>
        </div>
      )}
    </div>
  );

  /* ── Status + metrics strip ────────────────────────────────────── */
  const statusStrip = (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-3">
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Phase</span>
        <span className="text-xs font-mono-trade text-[var(--text-primary)]">{status?.phase ?? 'idle'}</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Last AUC</span>
        <span className="text-xs font-mono-trade text-[var(--text-primary)]">{metrics.auc?.toFixed(3) ?? '—'}</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">OOS Sharpe</span>
        <span className="text-xs font-mono-trade text-[var(--text-primary)]">{metrics.sharpe?.toFixed(3) ?? '—'}</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Last run</span>
        <span className="text-xs font-mono-trade text-[var(--text-secondary)]">{fmt(status?.last_run_time ?? null)}</span>
      </div>
    </div>
  );

  /* ── Log viewer ────────────────────────────────────────────────── */
  const logViewer = (
    <div className="mt-3 rounded-lg bg-[var(--bg-primary)] border border-[var(--border-subtle)] overflow-hidden">
      <div className="px-3 py-1.5 border-b border-[var(--border-subtle)] flex items-center justify-between">
        <span className="text-[10px] font-mono-trade text-[var(--text-muted)] uppercase tracking-wider">Live Logs</span>
        {anyRunning && (
          <span className="flex items-center gap-1.5 text-[10px] text-[var(--accent-emerald)] font-mono-trade">
            <span className="w-1 h-1 rounded-full bg-[var(--accent-emerald)] animate-pulse" />
            streaming
          </span>
        )}
      </div>
      <div className="h-48 overflow-auto p-3 space-y-px font-mono-trade text-[11px] leading-relaxed">
        {logs.length === 0 ? (
          <p className="text-[var(--text-muted)]">No logs yet.</p>
        ) : (
          logs.map((line, idx) => (
            <p
              key={idx}
              className={`break-all ${
                line.raw.includes('❌') || line.raw.includes('Error')
                  ? 'text-[var(--accent-rose)]'
                  : line.raw.includes('✅') || line.raw.includes('BEST')
                  ? 'text-[var(--accent-emerald)]'
                  : line.raw.includes('⚠️')
                  ? 'text-[var(--accent-cyan)]'
                  : 'text-[var(--text-secondary)]/80'
              }`}
            >
              {line.raw}
            </p>
          ))
        )}
        <div ref={logEndRef} />
      </div>
    </div>
  );

  return (
    <section className="glass-card rounded-xl p-4 space-y-0">
      <style>{`
        .ops-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 5px 12px;
          border-radius: 8px;
          font-size: 11px;
          font-family: 'JetBrains Mono', monospace;
          font-weight: 500;
          border: 1px solid;
          transition: all 0.15s ease;
          cursor: pointer;
        }
        .ops-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
        .animate-in {
          animation: slideDown 0.15s ease-out;
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {summaryBar}
      {error && <p className="text-xs text-[var(--accent-rose)] pt-2">{error}</p>}

      {expanded && (
        <div className="animate-in">
          {statusStrip}
          {actions}
          {paramDrawer}
          {logViewer}
        </div>
      )}
    </section>
  );
}