import { ModelCoinInfo, ModelStatusData } from '../types';

const STATUS_CONFIG: Record<string, { label: string; dot: string; text: string }> = {
  active:       { label: 'Active',        dot: 'bg-accent-emerald', text: 'text-accent-emerald' },
  gate_rejected:{ label: 'Gate Rejected', dot: 'bg-amber-400',      text: 'text-amber-400' },
  stale:        { label: 'Stale',         dot: 'bg-amber-400',      text: 'text-amber-400' },
  auc_rejected: { label: 'AUC Rejected',  dot: 'bg-accent-rose',    text: 'text-accent-rose' },
};

function fmtRelTime(isoStr: string | null): string {
  if (!isoStr) return '—';
  const diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
  if (diff < 90) return 'just now';
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${(diff / 3600).toFixed(1)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

function fmtAbsTime(isoStr: string | null): string {
  if (!isoStr) return '—';
  const d = new Date(isoStr);
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short',
  });
}

function timeUntil(isoStr: string | null): string {
  if (!isoStr) return '—';
  const diff = (new Date(isoStr).getTime() - Date.now()) / 1000;
  if (diff <= 0) return 'due now';
  const h = Math.floor(diff / 3600);
  const d = Math.floor(h / 24);
  if (d >= 1) return `in ${d}d ${h % 24}h`;
  return `in ${h}h ${Math.floor((diff % 3600) / 60)}m`;
}

function CoinRow({ c }: { c: ModelCoinInfo }) {
  const cfg = STATUS_CONFIG[c.status] ?? STATUS_CONFIG.stale;
  const gateLabel = c.gate_failure_reason
    ? c.gate_failure_reason.replace(/_/g, ' ')
    : c.status === 'auc_rejected' ? 'no signal written' : null;

  return (
    <div className="flex items-center gap-3 py-2 border-b border-[rgba(56,189,248,0.06)] last:border-0">
      {/* Coin */}
      <span className="font-mono text-xs font-semibold text-tx-primary w-12 flex-shrink-0">{c.coin}</span>

      {/* Status badge */}
      <div className="flex items-center gap-1.5 w-28 flex-shrink-0">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot} ${c.status === 'active' ? 'animate-pulse' : ''}`} />
        <span className={`text-[10px] font-medium ${cfg.text}`}>{cfg.label}</span>
      </div>

      {/* AUC */}
      <span className="font-mono text-xs text-tx-muted w-16 flex-shrink-0">
        {c.model_auc != null ? `AUC ${c.model_auc.toFixed(3)}` : 'AUC —'}
      </span>

      {/* Gate failure / last signal */}
      <div className="flex-1 min-w-0">
        {gateLabel ? (
          <span className="text-[10px] text-tx-muted font-mono truncate">{gateLabel}</span>
        ) : null}
      </div>

      {/* Time */}
      <span className="text-[10px] font-mono text-tx-muted flex-shrink-0 text-right w-20">
        {fmtRelTime(c.last_signal_at)}
      </span>
    </div>
  );
}

interface Props {
  data: ModelStatusData | null;
}

export default function ModelStatusPanel({ data }: Props) {
  if (!data) {
    return (
      <div className="glass-card rounded-xl p-5">
        <div className="text-tx-secondary text-xs font-medium tracking-widest uppercase mb-4">ML Model Status</div>
        <div className="text-tx-muted text-xs">Loading…</div>
      </div>
    );
  }

  const retrain = data.last_retrain;
  const retrainStatusColor = !retrain ? 'text-tx-muted'
    : retrain.status === 'success' ? 'text-accent-emerald'
    : 'text-accent-rose';

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-tx-secondary text-xs font-medium tracking-widest uppercase">ML Model Status</span>
        <span className="text-tx-muted text-[10px] font-mono">inference mode · retrain every {data.retrain_every_days}d</span>
      </div>

      {/* Per-coin rows */}
      {data.coins.length === 0 ? (
        <div className="text-tx-muted text-xs">No active coins configured.</div>
      ) : (
        <div className="mb-4">
          {data.coins.map(c => <CoinRow key={c.coin} c={c} />)}
        </div>
      )}

      {/* Retrain info footer */}
      <div className="pt-3 border-t border-[rgba(56,189,248,0.08)] grid grid-cols-2 gap-x-4 gap-y-1">
        <div>
          <div className="text-tx-muted text-[9px] tracking-widest uppercase mb-0.5">Last Retrain</div>
          <div className={`text-[10px] font-mono ${retrainStatusColor}`}>
            {retrain ? `${retrain.status} · ${retrain.symbols_trained}/${retrain.symbols_total} coins` : '—'}
          </div>
          {retrain?.started_at && (
            <div className="text-tx-muted text-[9px] font-mono">{fmtAbsTime(retrain.started_at)}</div>
          )}
          {retrain?.error && (
            <div className="text-accent-rose text-[9px] font-mono truncate mt-0.5" title={retrain.error}>
              {retrain.error.slice(0, 60)}…
            </div>
          )}
        </div>
        <div>
          <div className="text-tx-muted text-[9px] tracking-widest uppercase mb-0.5">Next Retrain</div>
          <div className="text-tx-primary text-[10px] font-mono">{timeUntil(data.next_retrain_at)}</div>
          {data.next_retrain_at && (
            <div className="text-tx-muted text-[9px] font-mono">{fmtAbsTime(data.next_retrain_at)}</div>
          )}
          {!data.next_retrain_at && (
            <div className="text-amber-400 text-[9px] font-mono">no successful retrain yet</div>
          )}
        </div>
      </div>
    </div>
  );
}
