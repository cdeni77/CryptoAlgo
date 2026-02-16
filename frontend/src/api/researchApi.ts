import { ResearchCoinDetail, ResearchFeatures, ResearchRun, ResearchSummary } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getResearchSummary(): Promise<ResearchSummary> {
  return fetchWithError(`${API_BASE}/research/summary`);
}

export async function getResearchCoin(coin: string): Promise<ResearchCoinDetail> {
  return fetchWithError(`${API_BASE}/research/coins/${coin}`);
}

export async function getResearchRuns(limit = 25): Promise<ResearchRun[]> {
  return fetchWithError(`${API_BASE}/research/runs?limit=${limit}`);
}

export async function getResearchFeatures(coin: string): Promise<ResearchFeatures> {
  return fetchWithError(`${API_BASE}/research/features/${coin}`);
}
