import {
  ResearchCoinDetail,
  ResearchFeatures,
  ResearchJobLaunchResponse,
  ResearchJobLogResponse,
  ResearchRun,
  ResearchScriptListResponse,
  ResearchSummary,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
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

export async function getResearchScripts(): Promise<ResearchScriptListResponse> {
  return fetchWithError(`${API_BASE}/research/scripts`);
}


export async function getResearchJobs(limit = 25): Promise<ResearchJobLaunchResponse[]> {
  return fetchWithError(`${API_BASE}/research/jobs?limit=${limit}`);
}

export async function launchResearchJob(job: string, args: string[]): Promise<ResearchJobLaunchResponse> {
  return fetchWithError(`${API_BASE}/research/launch/${job}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ args }),
  });
}


export async function getResearchJobLogs(pid: number, lines = 200): Promise<ResearchJobLogResponse> {
  return fetchWithError(`${API_BASE}/research/jobs/${pid}/logs?lines=${lines}`);
}
