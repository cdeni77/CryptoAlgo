import { OpsActionResponse, OpsLogsResponse, OpsStatus } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithError<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function startPipeline(): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/pipeline/start`, { method: 'POST' });
}

export async function stopPipeline(): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/pipeline/stop`, { method: 'POST' });
}

export async function triggerRetrain(): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/retrain`, { method: 'POST' });
}

export async function getOpsStatus(): Promise<OpsStatus> {
  return fetchWithError(`${API_BASE}/ops/status`);
}

export async function getOpsLogs(limit = 200): Promise<OpsLogsResponse> {
  return fetchWithError(`${API_BASE}/ops/logs?limit=${limit}`);
}
