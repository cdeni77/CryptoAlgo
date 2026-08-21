import { get } from './client';
import { WalletData } from '../types';

export async function getWallet(): Promise<WalletData> {
  return get('/wallet/');
}
