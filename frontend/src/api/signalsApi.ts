import { Signal } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getRecentSignals(limit = 50): Promise<Signal[]> {
  return fetchWithError(`${API_BASE}/signals?limit=${limit}`);
}

export async function getSignalsByCoin(coin: string, hours?: number, limit = 100): Promise<Signal[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (hours) params.set('hours', String(hours));
  return fetchWithError(`${API_BASE}/signals/coin/${coin}?${params}`);
}