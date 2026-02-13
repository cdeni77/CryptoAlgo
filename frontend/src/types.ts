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

export interface OpsActionResponse {
  action: string;
  status: string;
  detail: string;
  pid?: number;
}

export interface OpsLogEntry {
  raw: string;
  timestamp: string | null;
  level: string | null;
  message: string | null;
}

export interface OpsLogsResponse {
  entries: OpsLogEntry[];
}

export interface OpsStatus {
  pipeline_running: boolean;
  training_running: boolean;
  phase: string;
  symbol: string | null;
  metrics: Record<string, number>;
  last_run_time: string | null;
  next_run_time: string | null;
  log_file: string;
}
