import { get, post } from './client';
import {
  ResearchFeatures,
  ResearchJobLaunchResponse,
  ResearchJobLogResponse,
  ResearchRun,
  ResearchScriptListResponse,
  ResearchSummary,
} from '../types';

/**
 * The client-side `normalizeTier`/`normalizeScale` helpers this file used to
 * carry are gone, and their absence is the point. They invented a tier when the
 * API returned none — falling back to 'PILOT' or 'REJECT' from a boolean — and
 * invented a position scale from that invented tier. So the dashboard displayed
 * a readiness grade and a recommended size that the backend had never computed.
 *
 * Nothing is normalised here now. A null from the API reaches the component as
 * a null, and the component renders "—".
 */

export async function getResearchSummary(): Promise<ResearchSummary> {
  return get('/research/summary');
}


export async function getResearchRuns(limit = 50): Promise<ResearchRun[]> {
  return get(`/research/runs?limit=${limit}`);
}

export async function getResearchFeatures(coin: string): Promise<ResearchFeatures> {
  return get(`/research/features/${coin}`);
}

export async function getResearchScripts(): Promise<ResearchScriptListResponse> {
  return get('/research/scripts');
}

export async function getResearchJobs(limit = 25): Promise<ResearchJobLaunchResponse[]> {
  return get(`/research/jobs?limit=${limit}`);
}

export async function launchResearchJob(
  job: string,
  args: string[],
): Promise<ResearchJobLaunchResponse> {
  return post(`/research/launch/${job}`, { args });
}

export async function getResearchJobLogs(
  pid: number,
  lines = 200,
): Promise<ResearchJobLogResponse> {
  return get(`/research/jobs/${pid}/logs?lines=${lines}`);
}
