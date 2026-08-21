import { get } from './client';
import { PriceData, HistoryEntry, CDESpecs, CoinSymbol } from '../types';

export async function getCurrentPrices(): Promise<PriceData> {
  return get('/coins/prices');
}

export async function getCDEPrices(): Promise<PriceData> {
  return get('/coins/cde-prices');
}

export async function getCDESpecs(): Promise<CDESpecs> {
  return get('/coins/cde-specs');
}

const RANGE_PARAM: Record<string, [string, string]> = {
  '1h': ['hours', '1'],
  '1d': ['days', '1'],
  '1w': ['days', '7'],
  '1m': ['days', '30'],
  '1y': ['days', '365'],
};

export async function getCoinHistory(
  symbol: CoinSymbol,
  range: '1h' | '1d' | '1w' | '1m' | '1y' = '1d',
): Promise<HistoryEntry[]> {
  const [key, value] = RANGE_PARAM[range] ?? RANGE_PARAM['1d'];
  return get(`/coins/history/${symbol}?${key}=${value}`);
}
