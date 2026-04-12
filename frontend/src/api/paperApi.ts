import { ModelStatusData, PaperEquityPoint, PaperFill, PaperPosition, PaperSummary } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export async function getPaperSummary(): Promise<PaperSummary> {
  return fetchWithError(`${API_BASE}/paper/summary`);
}

export async function getPaperPositions(): Promise<PaperPosition[]> {
  return fetchWithError(`${API_BASE}/paper/positions`);
}

export async function getPaperEquity(limit = 250): Promise<PaperEquityPoint[]> {
  return fetchWithError(`${API_BASE}/paper/equity?limit=${limit}`);
}

export async function getPaperFills(limit = 100): Promise<PaperFill[]> {
  return fetchWithError(`${API_BASE}/paper/fills?limit=${limit}`);
}

export async function getPaperConfig(): Promise<{ active_coins: string[]; tier_map: Record<string, string> }> {
  return fetchWithError(`${API_BASE}/paper/config`);
}

export async function getModelStatus(): Promise<ModelStatusData> {
  return fetchWithError(`${API_BASE}/paper/model-status`);
}
