import { get } from './client';
import { Trade } from '../types';

export async function getAllTrades(skip = 0, limit = 50): Promise<Trade[]> {
  return get(`/trades/?skip=${skip}&limit=${limit}`);
}
