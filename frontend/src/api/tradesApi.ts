import { Trade } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getAllTrades(skip = 0, limit = 50): Promise<Trade[]> {
  return fetchWithError(`${API_BASE}/trades?skip=${skip}&limit=${limit}`);
}