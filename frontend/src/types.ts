export interface PriceInfo {
  price: number | null;
  change24h: number | null;
}

export type PriceData = {
  BTC: PriceInfo;
  ETH: PriceInfo;
  SOL: PriceInfo;
};

export interface HistoryEntry {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  id: number;
  coin: string;
  datetime_open: string;
  datetime_close: string | null;
  side: 'long' | 'short';
  contracts: number;
  entry_price: number;
  exit_price: number | null;
  fee_open: number | null;
  fee_close: number | null;
  net_pnl: number | null;
  margin_used: number | null;
  leverage: number | null;
  reason_entry: string | null;
  reason_exit: string | null;
  status: 'open' | 'closed';
}

export interface WalletData {
  balance: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
}