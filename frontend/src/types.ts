export interface PriceInfo {
  price: number | null;
  change24h: number | null;
}

export type CoinSymbol = 'BTC' | 'ETH' | 'SOL' | 'XRP' | 'DOGE';

export type PriceData = Record<CoinSymbol, PriceInfo>;

export type DataSource = 'spot' | 'cde';

export interface HistoryEntry {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CDESpec {
  symbol: string;
  code: string;
  units_per_contract: number;
  approx_contract_value: number;
  fee_pct: number;
}

export type CDESpecs = Record<CoinSymbol, CDESpec>;

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

export interface Signal {
  id: number;
  coin: string;
  timestamp: string;
  direction: 'long' | 'short' | 'neutral';
  confidence: number;
  raw_probability: number | null;
  model_auc: number | null;
  price_at_signal: number | null;
  momentum_pass: boolean | null;
  trend_pass: boolean | null;
  regime_pass: boolean | null;
  ml_pass: boolean | null;
  contracts_suggested: number | null;
  notional_usd: number | null;
  acted_on: boolean;
  trade_id: number | null;
  created_at: string | null;
}

export interface WalletData {
  balance: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
}

export interface PaperOrder {
  id: number;
  signal_id: number;
  coin: string;
  side: 'long' | 'short';
  contracts: number;
  target_price: number;
  status: 'new' | 'filled' | 'canceled';
  created_at: string;
}

export interface PaperFill {
  id: number;
  order_id: number;
  signal_id: number;
  coin: string;
  side: 'long' | 'short';
  contracts: number;
  fill_price: number;
  fee: number;
  notional: number;
  slippage_bps: number;
  created_at: string;
}

export interface PaperPosition {
  id: number;
  coin: string;
  side: 'long' | 'short';
  contracts: number;
  entry_price: number;
  mark_price: number;
  notional: number;
  realized_pnl: number;
  unrealized_pnl: number;
  fees_paid: number;
  opened_at: string;
  updated_at: string | null;
  is_open: boolean;
}

export interface PaperEquityPoint {
  id: number;
  timestamp: string;
  equity: number;
  cash_balance: number;
  unrealized_pnl: number;
  realized_pnl: number;
  open_positions: number;
}
