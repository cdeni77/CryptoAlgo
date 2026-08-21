import { ModelCoinInfo, ModelStatusData } from '../types';

/**
 * Per-instrument model state, from the most recent signal each one produced.
 *
 * The AUC column is gone: it was a classification metric on a model that
 * regresses net return, so it read "AUC —" on every row. What each instrument
 * has instead is the edge the model forecast and the round trip that edge has to
 * clear, which is the pair that decides whether a trade happens.
 *
 * The layout is a grid rather than a flex row with fixed widths. The old one
 * pinned a `w-20` time column beside a `flex-1` reason and let both overflow, so
 * "volatility regime" and "3.1h ago" printed on top of each other.
 */

const STATUS = {
  active: { label: 'Active', dot: 'bg-accent-emerald', text: 'text-accent-emerald' },
  gate_rejected: { label: 'Blocked', dot: 'bg-accent-amber', text: 'text-accent-amber' },
  stale: { label: 'Stale', dot: 'bg-accent-amber', text: 'text-accent-amber' },
  no_signal: { label: 'No signal', dot: 'bg-tx-muted', text: 'text-tx-muted' },
} as const;

function relative(iso: string | null): string {
  if (!iso) return '—';
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 90) return 'now';
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function absolute(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-GB', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    timeZone: 'UTC', hour12: false,
  });
}

function until(iso: string | null): string {
  if (!iso) return '—';
  const seconds = (new Date(iso).getTime() - Date.now()) / 1000;
  if (seconds <= 0) return 'due now';
  const hours = Math.floor(seconds / 3600);
  const days = Math.floor(hours / 24);
  return days >= 1 ? `in ${days}d ${hours % 24}h` : `in ${hours}h`;
}

function CoinRow({ c }: { c: ModelCoinInfo }) {
  const style = STATUS[c.status] ?? STATUS.no_signal;
  const detail = c.gate_failure_reason?.replace(/_/g, ' ') ?? null;
  const net = c.expected_net_bps;

  return (
    <div className="grid grid-cols-[2.6rem_1fr_4.2rem_2.4rem] items-center gap-2 border-b border-[rgba(56,189,248,0.06)] py-2 last:border-0">
      <span className="font-mono text-xs font-semibold text-tx-primary">{c.coin}</span>

      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${style.dot} ${
              c.status === 'active' ? 'animate-pulse' : ''
            }`}
            aria-hidden
          />
          <span className={`text-[10px] font-medium ${style.text}`}>{style.label}</span>
        </div>
        {detail && (
          <div className="truncate font-mono text-[10px] text-tx-muted" title={detail}>
            {detail}
          </div>
        )}
      </div>

      {/* Net edge against the cost it has to clear. Shown together because a
          +12bp forecast is a trade on DOGE at 5bp and a loss on ETH at 54bp. */}
      <div className="text-right">
        <div
          className={`font-mono text-[11px] tabular-nums ${
            net === null ? 'text-tx-muted' : net > 0 ? 'text-accent-emerald' : 'text-accent-rose'
          }`}
        >
          {net === null ? '—' : `${net >= 0 ? '+' : ''}${net.toFixed(1)}`}
        </div>
        <div className="font-mono text-[9px] text-tx-muted tabular-nums">
          {c.cost_bps === null ? '' : `/ ${c.cost_bps.toFixed(1)}bp`}
        </div>
      </div>

      <span className="text-right font-mono text-[10px] text-tx-muted tabular-nums">
        {relative(c.last_signal_at)}
      </span>
    </div>
  );
}

export default function ModelStatusPanel({ data }: { data: ModelStatusData }) {
  const retrain = data.last_retrain;
  const tone = !retrain
    ? 'text-tx-muted'
    : retrain.status === 'success'
      ? 'text-accent-emerald'
      : 'text-accent-rose';

  return (
    <div className="glass-card rounded-xl p-5">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-xs font-medium uppercase tracking-widest text-tx-secondary">
          Signal status
        </span>
        <span className="font-mono text-[10px] text-tx-muted">
          retrain every {data.retrain_every_days}d
        </span>
      </div>
      <p className="mb-3 text-[10px] leading-snug text-tx-muted">
        Latest signal per instrument: expected net edge over the round trip it has
        to clear.
      </p>

      {data.coins.length === 0 ? (
        <div className="py-6 text-center text-xs text-tx-muted">
          No active instruments configured.
        </div>
      ) : (
        <div className="mb-4">
          {data.coins.map((c) => (
            <CoinRow key={c.coin} c={c} />
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-[rgba(56,189,248,0.08)] pt-3">
        <div className="min-w-0">
          <div className="mb-0.5 text-[9px] uppercase tracking-widest text-tx-muted">
            Last retrain
          </div>
          <div className={`font-mono text-[10px] ${tone}`}>
            {retrain
              ? `${retrain.status} · ${retrain.symbols_trained}/${retrain.symbols_total}`
              : '—'}
          </div>
          <div className="font-mono text-[9px] text-tx-muted">
            {absolute(retrain?.started_at ?? null)}
          </div>
          {retrain?.error && (
            <div
              className="mt-0.5 truncate font-mono text-[9px] text-accent-rose"
              title={retrain.error}
            >
              {retrain.error}
            </div>
          )}
        </div>
        <div className="min-w-0">
          <div className="mb-0.5 text-[9px] uppercase tracking-widest text-tx-muted">
            Next retrain
          </div>
          <div className="font-mono text-[10px] text-tx-primary">
            {until(data.next_retrain_at)}
          </div>
          <div className="font-mono text-[9px] text-tx-muted">
            {data.next_retrain_at ? absolute(data.next_retrain_at) : 'no successful retrain yet'}
          </div>
        </div>
      </div>
    </div>
  );
}
