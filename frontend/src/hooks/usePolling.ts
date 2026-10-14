import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '../api/client';

/**
 * Poll an endpoint.
 *
 * Stops while the tab is hidden. Reports a failure as `error`, and keeps the
 * last data on screen beside it. `loading` is true only until the first
 * response settles, so an empty result renders as empty, not as a spinner.
 */

export interface PollingState<T> {
  data: T | null;
  error: ApiError | Error | null;
  /** True until the first response settles, and never again, so a background
   *  refresh never blanks the screen. */
  loading: boolean;
  /** True while a refresh is in flight over existing data. */
  refreshing: boolean;
  lastUpdated: Date | null;
  refresh: () => void;
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  deps: unknown[] = [],
): PollingState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // The fetcher is usually an inline closure, so it is a new function every
  // render. Holding it in a ref keeps the effect from re-subscribing each time.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const mounted = useRef(true);
  const inFlight = useRef(false);
  // Which set of `deps` the in-flight request belongs to, so a response for a
  // previous subject is dropped instead of shown under the new one.
  const generation = useRef(0);
  const firstRun = useRef(true);

  const load = useCallback(async () => {
    // Skip if a request is already out, so requests never stack up.
    if (inFlight.current) return;
    const requested = generation.current;
    inFlight.current = true;
    setRefreshing(true);
    try {
      const next = await fetcherRef.current();
      // Stale if the dependencies moved on while this was in flight.
      if (!mounted.current || generation.current !== requested) return;
      setData(next);
      setError(null);
      setLastUpdated(new Date());
    } catch (caught) {
      if (!mounted.current || generation.current !== requested) return;
      setError(caught instanceof Error ? caught : new Error(String(caught)));
    } finally {
      if (generation.current === requested) inFlight.current = false;
      if (mounted.current && generation.current === requested) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    // A new generation invalidates any in-flight response and frees the guard,
    // so the new subject is fetched at once.
    generation.current += 1;
    inFlight.current = false;

    if (firstRun.current) {
      firstRun.current = false;
    } else {
      // The previous value describes a different subject: clear it.
      setData(null);
      setError(null);
      setLoading(true);
      setLastUpdated(null);
    }

    let timer: number | undefined;

    const start = () => {
      if (timer !== undefined) return;
      timer = window.setInterval(load, intervalMs);
    };
    const stop = () => {
      if (timer === undefined) return;
      window.clearInterval(timer);
      timer = undefined;
    };

    const onVisibilityChange = () => {
      if (document.hidden) {
        stop();
      } else {
        // Refresh at once on return, not at the next interval.
        void load();
        start();
      }
    };

    void load();
    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      mounted.current = false;
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, intervalMs, ...deps]);

  return { data, error, loading, refreshing, lastUpdated, refresh: load };
}
