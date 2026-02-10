import { PriceData, HistoryEntry } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getCurrentPrices(): Promise<PriceData> {
  return fetchWithError(`${API_BASE}/coins/prices`);
}

export async function getCoinHistory(
  symbol: 'BTC' | 'ETH' | 'SOL',
  range: '1h' | '1d' | '1w' | '1m' | '1y' = '1d'
): Promise<HistoryEntry[]> {
  const params = new URLSearchParams();

  switch (range) {
    case '1h':
      params.set('hours', '1');
      break;

    case '1d':
      params.set('days', '1');
      break;

    case '1w':
      params.set('days', '7');
      break;

    case '1m':
      params.set('days', '30');
      break;

    case '1y':
      params.set('days', '365');
      break;

    default:
      params.set('days', '60');
  }

  const url = `${API_BASE}/coins/history/${symbol}?${params.toString()}`;
  return fetchWithError(url);
}