export interface PriceInfo {
  price: number | null;
  change24h: number | null;
}

// The universe, once. `CoinSymbol` is derived from the array rather than
// declared alongside it, so a coin added here cannot be missing from the type —
// and two pages had already grown their own private copies, in different
// orders, which is how a coin ends up tradeable on one screen and absent from
// the other.
//
// All sixteen the trader models (`core/profiles.py`), not the nine the API used
// to serve. The nine were mistaken for the traded universe when writing the
// spot scrape, which would have left seven instruments with no cross-venue
// features and no row on any screen.
// `test_orm_parity.py::test_the_api_serves_every_instrument_the_trader_models`
// keeps the three lists in step.
export const ALL_COINS = [
  'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'ADA', 'LINK', 'LTC',
  'BCH', 'DOT', 'NEAR', 'SUI', 'XLM', 'PEPE', 'SHIB',
] as const;
export type CoinSymbol = (typeof ALL_COINS)[number];

export type PriceData = Record<CoinSymbol, PriceInfo>;


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
  /** Stale 10bp/side guess, kept only for compatibility. Coinbase CDE charges a
   *  flat per-contract commission, not a percentage — use `CDESpecs.fees`. */
  fee_pct: number;
}

/** The venue's actual commission schedule, read from the same config file the
 *  research pipeline prices its targets with. */
export interface CDEFeeSchedule {
  version: string | null;
  mode: string | null;
  per_contract_usd_default: number | null;
  per_contract_usd_by_code: Record<string, number>;
  taker_fee_bps: number | null;
  maker_fee_bps: number | null;
}

export interface CDESpecs {
  contracts: Record<string, CDESpec>;
  fees: CDEFeeSchedule | Record<string, never>;
}


export interface Signal {
  id: number;
  coin: string;
  timestamp: string;
  direction: 'long' | 'short' | 'neutral';
  /** Edge-to-risk: expected net return over its own uncertainty. Named
   *  `confidence` because the column is, but it is not a probability. */
  confidence: number;
  price_at_signal: number | null;
  contracts_suggested: number | null;
  notional_usd: number | null;
  acted_on: boolean;
  trade_id: number | null;
  passed_gates: boolean;
  gate_failure_reason: string | null;
  created_at: string | null;

  // The decision, decomposed. This is what `decide()` actually produced.
  expected_net_bps: number | null;
  expected_price_bps: number | null;
  expected_carry_bps: number | null;
  cost_bps: number | null;
  sigma_bps: number | null;
  edge_to_risk: number | null;
  /** Share of the expected edge that is carry rather than direction. */
  carry_share: number | null;
  participation: number | null;
  model_version: string | null;

  // Classifier-era columns, left null by the current signal writer. Historical
  // rows still carry real values, which is why they are not dropped.
  raw_probability: number | null;
  model_auc: number | null;
  momentum_pass: boolean | null;
  trend_pass: boolean | null;
  regime_pass: boolean | null;
  ml_pass: boolean | null;
}

export interface PaperSummary {
  // All nullable: `/paper/summary` returns nulls plus `unavailable_reason` when
  // the account has not traded. Declaring them non-nullable is what let
  // `summary.data.equity - 0` evaluate to 0 and render a fresh install as a
  // $0.00 portfolio that had lost 100%, with tsc unable to see it.
  total_return_pct: number | null;
  realized_pnl: number | null;
  unrealized_pnl: number | null;
  equity: number | null;
  cash_balance: number | null;
  max_drawdown_pct: number | null;
  win_rate: number | null;
  fill_count: number;
  open_positions: number;
  sharpe_ratio: number | null;
  profit_factor: number | null;
  initial_equity?: number;
  total_fees?: number | null;
  total_notional?: number | null;
  closed_positions?: number;
  unavailable_reason?: string | null;
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
  /** Funding accrued so far. On hourly-funding perps a long hold can pay more in
   *  funding than in commission, so it is reported separately from fees. */
  funding_paid: number;
  opened_at: string;
  updated_at: string | null;
  is_open: boolean;
  tp_price?: number | null;
  sl_price?: number | null;
  max_hold_until?: string | null;
  exit_reason?: string | null;
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


/**
 * Research types, rewritten to match what the model produces.
 *
 * The old shapes described a classifier: `holdout_auc`, `pr_auc`,
 * `precision_at_threshold`, `readiness_tier`, `robustness_gate`. The model
 * regresses net return, so AUC is undefined for it — the API now leaves it null
 * — and the readiness tier was read from an artifact of a deleted pipeline, so it
 * was "UNKNOWN" for everything. `pr_auc` and `precision_at_threshold` were the
 * AUC with 0.06 and 0.04 subtracted: one number displayed three times.
 *
 * What replaces them is the comparison the model can be held to — the edge it
 * claimed against the edge it earned — plus the promotion gates, which are the
 * real readiness decision.
 */

export type Health = 'healthy' | 'watch' | 'at_risk' | 'unknown';

export interface EdgeCalibration {
  expected_net_bps: number | null;
  realised_net_bps: number | null;
  /** Realised minus expected. Persistently negative means the model overstates
   *  its edge, which over-sizes every position clearing the conviction floor. */
  delta_bps: number | null;
  sample: number;
}

export interface ResearchCoinHealth {
  coin: string;
  signals_total: number;
  signals_passed_gates: number;
  gate_pass_rate: number | null;
  top_gate_reason: string | null;
  last_signal_at: string | null;
  expected_net_bps: number | null;
  expected_carry_share: number | null;
  mean_cost_bps: number | null;
  trades_closed: number;
  win_rate_realized: number | null;
  net_pnl: number | null;
  realised_net_bps: number | null;
  calibration: EdgeCalibration;
  health: Health;
  health_reason: string | null;
}

export interface ResearchSummaryKpis {
  signals_total: number;
  signals_passed_gates: number;
  gate_pass_rate: number | null;
  trades_closed: number;
  win_rate_realized: number | null;
  net_pnl: number | null;
  expected_net_bps: number | null;
  realised_net_bps: number | null;
  calibration_delta_bps: number | null;
  expected_carry_share: number | null;
  model_version: string | null;
  model_promoted: boolean;
  model_forced: boolean;
  model_age_hours: number | null;
  gates_failed: string[];
  kill_switch_status: string;
  trials_to_date: number;
  health: Health;
}

export interface ResearchSummary {
  generated_at: string;
  kpis: ResearchSummaryKpis;
  coins: ResearchCoinHealth[];
}


export interface ResearchRun {
  id: string;
  run_type: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  duration_seconds: number | null;
  artifacts_version: string | null;
  symbols_trained: number;
  symbols_total: number;
  retrain_window_days: number | null;
  promoted: boolean | null;
  forced: boolean;
  failed_gates: string[];
  sharpe: number | null;
  trades: number | null;
  error: string | null;
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
  /** Set when importances could not be read, so the client can say why rather
   *  than render an empty chart that looks like a zero result. */
  importance_unavailable_reason: string | null;
}

// ---------------------------------------------------------------------------
// Model provenance and promotion gates
// ---------------------------------------------------------------------------

export interface GateResult {
  name: string;
  value: number | null;
  threshold: number;
  /** 'min' — value must be at least the threshold; 'max' — at most. */
  comparison: 'min' | 'max' | string;
  passed: boolean;
  note: string | null;
}

export interface ModelProvenance {
  version: string | null;
  feature_set_hash: string | null;
  n_features: number | null;
  heads: string[];
  uses_symbol_identity: boolean;
  horizon_bars: number | null;
  cost_config_version: string | null;
  trained_at: string | null;
  data_as_of: string | null;
  train_rows: number | null;
  /** Overlapping labels are not independent observations. Any significance claim
   *  belongs to this number, not to train_rows. */
  effective_observations: number | null;
  train_start: string | null;
  train_end: string | null;
  symbols: string[];
}

export interface BacktestSummary {
  trades: number | null;
  net_pnl: number | null;
  price_pnl: number | null;
  funding_pnl: number | null;
  fees: number | null;
  carry_contribution: number | null;
  return_pct: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  liquidations: number | null;
  max_entry_participation: number | null;
  max_exit_participation: number | null;
}

export interface PathDistributionSummary {
  n: number | null;
  median: number | null;
  mean: number | null;
  p05: number | null;
  p95: number | null;
  positive_fraction: number | null;
}

export interface SimulationSummary {
  bootstrap_sharpe: PathDistributionSummary | null;
  bootstrap_max_drawdown: PathDistributionSummary | null;
  probability_positive: number | null;
  risk_of_ruin: number | null;
  block_length: number | null;
  per_period_sharpe: PathDistributionSummary | null;
  synthetic_sharpe: PathDistributionSummary | null;
  stressed_worst_sharpe: number | null;
  parameter_plateau: number | null;
}

export interface PromotionRecord {
  version: string;
  created_at: string | null;
  promoted: boolean;
  forced: boolean;
  force_reason: string | null;
  is_live: boolean;
  failed_gates: string[];
  gates: GateResult[];
  provenance: ModelProvenance;
  backtest: BacktestSummary;
  simulation: SimulationSummary;
  error: string | null;
}

export interface KillSwitchStatus {
  status: string;
  version: string | null;
  evaluated_at: string | null;
  reasons: string[];
  trades: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  drawdown: number | null;
  expectancy: number | null;
  trades_per_week: number | null;
  window_days: number | null;
}

export interface LiveModel {
  generated_at: string;
  has_model: boolean;
  artifact_path: string | null;
  artifact_modified_at: string | null;
  trials_to_date: number;
  live: PromotionRecord | null;
  kill_switch: KillSwitchStatus;
  /** An artifact with no ledger entry: installed outside the gates. */
  unrecorded_artifact: boolean;
}

export interface PromotionHistory {
  generated_at: string;
  trials_to_date: number;
  live_version: string | null;
  records: PromotionRecord[];
}

export interface FeatureImportanceEntry {
  feature: string;
  importance: number;
  head: string;
}

export interface FeatureImportanceResponse {
  generated_at: string;
  version: string | null;
  features: FeatureImportanceEntry[];
  unavailable_reason: string | null;
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

export type ModelCoinStatus = 'active' | 'gate_rejected' | 'stale' | 'no_signal';

export interface ModelCoinInfo {
  coin: string;
  last_signal_at: string | null;
  /** The forecast and the round trip it has to clear. `model_auc` used to be
   *  here and is null on every row the current signal writer creates. */
  expected_net_bps: number | null;
  cost_bps: number | null;
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
  balance: number | null;
  realized_pnl: number | null;
  unrealized_pnl: number | null;
  unrealized_unavailable_reason?: string | null;
  total_pnl: number | null;
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
