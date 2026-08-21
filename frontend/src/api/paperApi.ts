import { get } from './client';
import {
  ModelStatusData,
  PaperEquityPoint,
  PaperFill,
  PaperPosition,
  PaperSummary,
} from '../types';

export async function getPaperSummary(): Promise<PaperSummary> {
  return get('/paper/summary');
}

export async function getPaperPositions(): Promise<PaperPosition[]> {
  return get('/paper/positions');
}

export async function getPaperEquity(limit = 250): Promise<PaperEquityPoint[]> {
  return get(`/paper/equity?limit=${limit}`);
}

export async function getPaperFills(limit = 100): Promise<PaperFill[]> {
  return get(`/paper/fills?limit=${limit}`);
}

export async function getPaperConfig(): Promise<{
  active_coins: string[];
  tier_map: Record<string, string>;
}> {
  return get('/paper/config');
}

export async function getModelStatus(): Promise<ModelStatusData> {
  return get('/paper/model-status');
}
