export interface PriceInfo {
  price: number | null;
  change24h: number | null;
}

export type CoinSymbol = 'BTC' | 'ETH' | 'SOL' | 'XRP' | 'DOGE' | 'AVAX' | 'ADA' | 'LINK' | 'LTC';
export const ALL_COINS: CoinSymbol[] = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'ADA', 'LINK', 'LTC'];

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

export type CDESpecs = Record<string, CDESpec>;

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
  passed_gates: boolean;
  gate_failure_reason: string | null;
  created_at: string | null;
}

export interface PaperSummary {
  total_return_pct: number;
  realized_pnl: number;
  unrealized_pnl: number;
  equity: number;
  cash_balance: number;
  max_drawdown_pct: number;
  win_rate: number;
  fill_count: number;
  open_positions: number;
  sharpe_ratio: number | null;
  profit_factor: number | null;
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

export interface ChartMarker {
  coin: string;
  side: 'long' | 'short';
  price: number;
  timestamp: string;
  contracts: number;
  kind: 'entry' | 'exit';
}

export type ReadinessTier = 'FULL' | 'PILOT' | 'SHADOW' | 'REJECT' | 'UNKNOWN';

export interface ReadinessTierDisplayMeta {
  label: string;
  tone: 'emerald' | 'amber' | 'slate' | 'rose';
  description: string;
}

export interface ResearchSummaryKpis {
  holdout_auc: number | null;
  pr_auc: number | null;
  precision_at_threshold: number | null;
  win_rate_realized: number;
  acted_signal_rate: number;
  drift_delta: number;
  readiness_tier?: ReadinessTier;
  recommended_position_scale?: number;
  readiness_tier_meta?: ReadinessTierDisplayMeta;
  robustness_gate: boolean;
}

export interface ResearchCoinHealth {
  coin: string;
  holdout_auc: number | null;
  pr_auc: number | null;
  precision_at_threshold: number | null;
  win_rate_realized: number;
  acted_signal_rate: number;
  drift_delta: number;
  readiness_tier?: ReadinessTier;
  recommended_position_scale?: number;
  robustness_gate: boolean;
  optimization_freshness_hours: number | null;
  last_optimized_at: string | null;
  health: 'healthy' | 'watch' | 'at_risk';
}

export interface ResearchSummary {
  generated_at: string;
  kpis: ResearchSummaryKpis;
  coins: ResearchCoinHealth[];
}

export interface ResearchCoinDetail {
  generated_at: string;
  coin: ResearchCoinHealth;
}

export interface ResearchRun {
  id: string;
  coin: string;
  run_type: 'train' | 'optimize' | 'validate';
  status: string;
  started_at: string;
  finished_at: string;
  duration_seconds: number;
  holdout_auc: number | null;
  readiness_tier?: ReadinessTier;
  recommended_position_scale?: number;
  readiness_tier_meta?: ReadinessTierDisplayMeta;
  robustness_gate: boolean;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface SignalDistributionItem {
  label: string;
  value: number;
}

export interface ResearchFeatures {
  coin: string;
  generated_at: string;
  feature_importance: FeatureImportanceItem[];
  signal_distribution: SignalDistributionItem[];
}

export interface ResearchScriptInfo {
  name: string;
  module: string;
  default_args: string[];
  launch_metadata?: {
    preset_choices?: string[];
    preset_default?: string;
  };
}

export interface ResearchScriptListResponse {
  scripts: ResearchScriptInfo[];
}

export interface ResearchJobLaunchRequest {
  args: string[];
}

export interface ResearchJobLaunchResponse {
  job: string;
  module: string;
  pid: number;
  command: string[];
  cwd: string;
  launched_at: string;
  log_path: string;
}

export interface ResearchJobLogResponse {
  pid: number;
  running: boolean;
  command: string[];
  launched_at: string;
  log_path: string;
  logs: string[];
}

export type ModelCoinStatus = 'active' | 'gate_rejected' | 'stale' | 'auc_rejected';

export interface ModelCoinInfo {
  coin: string;
  last_signal_at: string | null;
  model_auc: number | null;
  gate_failure_reason: string | null;
  passed_gates: boolean;
  status: ModelCoinStatus;
  hours_since_signal: number | null;
}

export interface ModelStatusData {
  coins: ModelCoinInfo[];
  last_retrain: {
    started_at: string | null;
    finished_at: string | null;
    status: string;
    symbols_trained: number;
    symbols_total: number;
    version: string | null;
    error: string | null;
  } | null;
  next_retrain_at: string | null;
  retrain_every_days: number;
}

export interface WalletAsset {
  asset: string;
  amount: number;
  price_usd: number;
  value_usd: number;
}

export interface WalletData {
  balance: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  wallets?: Record<string, { value_usd: number; cash_usd?: number; unrealized_pnl?: number; status: string }>;
  coinbase?: {
    spot?: { value_usd: number | null; status: string; assets?: WalletAsset[] };
    perps?: { value_usd: number | null; status: string; positions?: Array<{ symbol: string; contracts: number | null; mark_price: number | null; notional_usd: number | null; unrealized_pnl_usd: number | null }> };
    total_value_usd: number | null;
  };
  ledger?: {
    status: string;
    assets?: WalletAsset[];
    value_usd: number;
    updated_at?: string;
  };
  portfolio_history_by_range?: Record<string, Array<{ timestamp: string; paper_equity_usd: number; external_usd: number; total_value_usd: number }>>;
}
