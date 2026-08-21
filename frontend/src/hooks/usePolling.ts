import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '../api/client';

/**
 * Poll an endpoint, with the three things every screen here was missing.
 *
 * **It stops when nobody is looking.** Every page set up its own `setInterval`
 * and left it running forever. A dashboard on a background tab was hitting six
 * endpoints every three seconds, indefinitely — which on this stack means the
 * wallet endpoint calling Coinbase on a timer for a screen nobody is reading.
 *
 * **It reports failure.** The old handlers were `.catch(() => {})`. A backend
 * that had stopped responding looked exactly like a market that had stopped
 * moving: the last values simply stayed on screen. An error now surfaces, and
 * the stale data stays visible alongside it rather than being blanked, because
 * the last known price is still worth more than an empty panel.
 *
 * **It distinguishes "loading" from "empty".** `loading` is true only until the
 * first settled response, so an empty array after a successful fetch renders as
 * "nothing here" rather than as a spinner that never resolves.
 */

export interface PollingState<T> {
  data: T | null;
  error: ApiError | Error | null;
  /** True until the first response settles, and never again — a refresh in the
   *  background must not blank a populated screen. */
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

  const load = useCallback(async () => {
    // Skip if a request is already out. A slow endpoint on a short interval
    // otherwise stacks requests until one of them wins arbitrarily, and the
    // value that lands is whichever finished last, not whichever is newest.
    if (inFlight.current) return;
    inFlight.current = true;
    setRefreshing(true);
    try {
      const next = await fetcherRef.current();
      if (!mounted.current) return;
      setData(next);
      setError(null);
      setLastUpdated(new Date());
    } catch (caught) {
      if (!mounted.current) return;
      setError(caught instanceof Error ? caught : new Error(String(caught)));
    } finally {
      inFlight.current = false;
      if (mounted.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
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
        // Refresh immediately on return: the interval alone would leave a stale
        // value on screen for up to one full period after the tab is focused.
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
