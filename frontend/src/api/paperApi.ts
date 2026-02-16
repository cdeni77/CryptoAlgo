import { PaperEquityPoint, PaperFill, PaperOrder, PaperPosition } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getPaperPositions(): Promise<PaperPosition[]> {
  return fetchWithError(`${API_BASE}/paper/positions`);
}

export async function getPaperEquity(limit = 250): Promise<PaperEquityPoint[]> {
  return fetchWithError(`${API_BASE}/paper/equity?limit=${limit}`);
}

export async function getPaperOrders(limit = 100): Promise<PaperOrder[]> {
  return fetchWithError(`${API_BASE}/paper/orders?limit=${limit}`);
}

export async function getPaperFills(limit = 100): Promise<PaperFill[]> {
  return fetchWithError(`${API_BASE}/paper/fills?limit=${limit}`);
}
