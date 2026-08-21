import { ApiError } from '../api/client';

/**
 * Loading, error and empty states, in one place.
 *
 * Every panel used to handle a failure with `.catch(() => {})`, which meant a
 * backend that had stopped responding rendered identically to a market that had
 * stopped moving: the last values stayed on screen with nothing to say they were
 * stale. And every "no data" case rendered as "Loading…", so an empty result was
 * indistinguishable from a request that never came back.
 */

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 py-10 text-tx-muted text-sm">
      <span
        className="h-3 w-3 animate-spin rounded-full border-2 border-accent-cyan/30 border-t-accent-cyan"
        aria-hidden
      />
      <span>{label}…</span>
    </div>
  );
}

export function Empty({ message, hint }: { message: string; hint?: string }) {
  return (
    <div className="py-10 text-center">
      <div className="text-sm text-tx-secondary">{message}</div>
      {hint && <div className="mt-1.5 text-xs text-tx-muted">{hint}</div>}
    </div>
  );
}

/**
 * An error, with the fix where the message can name one.
 *
 * A 401 or 503 from the launch endpoint is not a generic failure — it means the
 * API token is missing on one side or the other — so it says that instead.
 */
export function ErrorBlock({
  error,
  onRetry,
  compact = false,
}: {
  error: Error;
  onRetry?: () => void;
  compact?: boolean;
}) {
  const isAuth = error instanceof ApiError && error.isAuthProblem;
  return (
    <div
      role="alert"
      className={`rounded-lg border border-accent-rose/25 bg-accent-rose/5 ${compact ? 'px-3 py-2' : 'p-4'}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-medium uppercase tracking-widest text-accent-rose">
            {isAuth ? 'Not authorised' : 'Request failed'}
          </div>
          <div className="mt-1 break-words text-xs text-tx-secondary">{error.message}</div>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex-shrink-0 rounded border border-accent-rose/30 px-2 py-1 text-[11px] text-accent-rose transition-colors hover:bg-accent-rose/10"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * The common wrapper: show the error, then the spinner, then the content.
 *
 * Order matters. Existing data stays on screen underneath an error banner rather
 * than being replaced by it, because a stale price is worth more than a blank
 * panel — but the banner has to be visible so nobody trades on the stale value
 * believing it is current.
 */
export function Panel<T>({
  state,
  children,
  emptyWhen,
  emptyMessage = 'Nothing to show yet',
  emptyHint,
  loadingLabel,
}: {
  state: { data: T | null; error: Error | null; loading: boolean; refresh: () => void };
  children: (data: T) => JSX.Element;
  emptyWhen?: (data: T) => boolean;
  emptyMessage?: string;
  emptyHint?: string;
  loadingLabel?: string;
}) {
  const { data, error, loading, refresh } = state;

  return (
    <>
      {error && (
        <div className={data ? 'mb-3' : ''}>
          <ErrorBlock error={error} onRetry={refresh} compact={Boolean(data)} />
        </div>
      )}
      {loading && !data && <Spinner label={loadingLabel} />}
      {data && (emptyWhen?.(data) ? <Empty message={emptyMessage} hint={emptyHint} /> : children(data))}
      {!loading && !data && !error && <Empty message={emptyMessage} hint={emptyHint} />}
    </>
  );
}

/** A dot plus a timestamp, so a screen can say how fresh it is. */
export function Freshness({
  lastUpdated,
  refreshing,
  error,
}: {
  lastUpdated: Date | null;
  refreshing?: boolean;
  error?: Error | null;
}) {
  const tone = error ? 'bg-accent-rose' : refreshing ? 'bg-accent-amber' : 'bg-accent-emerald';
  return (
    <span className="flex items-center gap-1.5 font-mono text-[10px] text-tx-muted">
      <span className={`h-1.5 w-1.5 rounded-full ${tone}`} aria-hidden />
      {lastUpdated
        ? lastUpdated.toLocaleTimeString('en-GB', { hour12: false })
        : '—'}
    </span>
  );
}
