import { get } from './client';
import { Signal } from '../types';

export async function getRecentSignals(limit = 50): Promise<Signal[]> {
  return get(`/signals?limit=${limit}`);
}
