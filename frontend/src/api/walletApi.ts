import { WalletData } from '../types';  

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

async function fetchWithError<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getWallet(): Promise<WalletData> {
  return fetchWithError(`${API_BASE}/wallet`);
}