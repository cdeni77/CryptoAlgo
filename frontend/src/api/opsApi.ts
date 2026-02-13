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

const makeQuery = <T extends object>(params: T) => {
  const q = new URLSearchParams();
  Object.entries(params as { [key: string]: unknown }).forEach(([k, v]) => {
    if (v !== undefined && v !== null) q.set(k, String(v));
  });
  return q.toString();
};

export interface ParallelLaunchOptions {
  trials: number;
  jobs: number;
  coins: string;
  plateau_patience: number;
  plateau_min_delta: number;
  plateau_warmup: number;
}

export interface TrainScratchOptions {
  backfill_days: number;
  include_oi: boolean;
  debug: boolean;
  threshold: number;
  min_auc: number;
  leverage: number;
  exclude_symbols: string;
}

export interface RetrainOptions {
  train_window_days: number;
  retrain_every_days: number;
  debug: boolean;
}

export async function startPipeline(): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/pipeline/start`, { method: 'POST' });
}

export async function stopPipeline(): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/pipeline/stop`, { method: 'POST' });
}

export async function triggerRetrain(options: RetrainOptions): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/retrain?${makeQuery(options)}`, { method: 'POST' });
}

export async function launchParallel(options: ParallelLaunchOptions): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/parallel-launch?${makeQuery(options)}`, { method: 'POST' });
}

export async function trainScratch(options: TrainScratchOptions): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/train-scratch?${makeQuery(options)}`, { method: 'POST' });
}

export async function launchParallel(trials = 200, jobs = 16): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/parallel-launch?trials=${trials}&jobs=${jobs}`, { method: 'POST' });
}

export async function trainScratch(): Promise<OpsActionResponse> {
  return fetchWithError(`${API_BASE}/ops/train-scratch`, { method: 'POST' });
}

export async function getOpsStatus(): Promise<OpsStatus> {
  return fetchWithError(`${API_BASE}/ops/status`);
}

export async function getOpsLogs(limit = 200): Promise<OpsLogsResponse> {
  return fetchWithError(`${API_BASE}/ops/logs?limit=${limit}`);
}
