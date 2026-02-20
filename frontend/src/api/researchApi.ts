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

const normalizeTier = (tier: unknown, robustnessGate: boolean): 'FULL' | 'PILOT' | 'SHADOW' | 'REJECT' | 'UNKNOWN' => {
  if (tier === 'FULL' || tier === 'PILOT' || tier === 'SHADOW' || tier === 'REJECT' || tier === 'UNKNOWN') {
    return tier;
  }
  return robustnessGate ? 'PILOT' : 'REJECT';
};

const normalizeScale = (scale: unknown, tier: 'FULL' | 'PILOT' | 'SHADOW' | 'REJECT' | 'UNKNOWN'): number => {
  if (typeof scale === 'number' && Number.isFinite(scale)) {
    return scale;
  }
  if (tier === 'FULL') return 1.0;
  if (tier === 'PILOT') return 0.5;
  if (tier === 'SHADOW') return 0.2;
  return 0.0;
};


async function fetchWithError<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getResearchSummary(): Promise<ResearchSummary> {
  const data = await fetchWithError<ResearchSummary>(`${API_BASE}/research/summary`);
  const coins = data.coins.map((coin) => {
    const readinessTier = normalizeTier(coin.readiness_tier, coin.robustness_gate);
    return {
      ...coin,
      readiness_tier: readinessTier,
      recommended_position_scale: normalizeScale(coin.recommended_position_scale, readinessTier),
    };
  });

  return {
    ...data,
    coins,
    kpis: {
      ...data.kpis,
      readiness_tier: normalizeTier(data.kpis.readiness_tier, data.kpis.robustness_gate),
      recommended_position_scale: normalizeScale(data.kpis.recommended_position_scale, normalizeTier(data.kpis.readiness_tier, data.kpis.robustness_gate)),
    },
  };
}

export async function getResearchCoin(coin: string): Promise<ResearchCoinDetail> {
  const data = await fetchWithError<ResearchCoinDetail>(`${API_BASE}/research/coins/${coin}`);
  const readinessTier = normalizeTier(data.coin.readiness_tier, data.coin.robustness_gate);
  return {
    ...data,
    coin: {
      ...data.coin,
      readiness_tier: readinessTier,
      recommended_position_scale: normalizeScale(data.coin.recommended_position_scale, readinessTier),
    },
  };
}

export async function getResearchRuns(limit = 25): Promise<ResearchRun[]> {
  const runs = await fetchWithError<ResearchRun[]>(`${API_BASE}/research/runs?limit=${limit}`);
  return runs.map((run) => {
    const readinessTier = normalizeTier(run.readiness_tier, run.robustness_gate);
    return {
      ...run,
      readiness_tier: readinessTier,
      recommended_position_scale: normalizeScale(run.recommended_position_scale, readinessTier),
    };
  });
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
