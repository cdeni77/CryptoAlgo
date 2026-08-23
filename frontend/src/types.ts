/** Shapes the API actually returns.
 *
 * Two conventions run through all of this and they are the point:
 *
 * `Measured<T>` is a value that may be absent *with a reason*. The API serves
 * measurements and never substitutes, so a missing number arrives as
 * `{ value: null, reason: '...' }` rather than as a zero, and the UI renders an
 * em-dash with the reason on hover. The previous version of this surface
 * reported `pr_auc` as `holdout_auc - 0.06` and a hardcoded table of feature
 * importances, all of which rendered identically to real data.
 *
 * Probabilities are the app's native unit. A binary's price *is* a probability,
 * so price, forecast and break-even are all on one dimensionless scale and are
 * directly comparable — no basis points, no notional.
 */

export interface Measured<T> {
  value: T | null;
  reason: string | null;
}

/** Why a decision was refused, in funnel order. Mirrors core.decide.Reason. */
export type Reason =
  | 'traded'
  | 'not_finite'
  | 'price_out_of_band'
  | 'disagreement_implausible'
  | 'edge_below_gate'
  | 'below_min_contracts'
  | 'fee_ceiling'
  | 'window_exposure'
  | 'position_limit'
  | 'already_entered'
  | 'bankroll_floor';

export type Side = 'up' | 'down';
export type PositionOutcome = 'pending' | 'won' | 'lost';

export interface Prediction {
  symbol: string;
  window_open: string;
  settle_time: string;
  offset_minutes: number;
  decision_time: string;
  strike: number;
  last_price: number;
  displacement: number;
  sigma_remaining: number | null;
  z_score: number | null;
  baseline_probability: number;
  model_probability: number;
  reason: Reason;
  traded: boolean;
  side: Side | null;
  price: number | null;
  effective_cost: number | null;
  edge: number | null;
  contracts: number | null;
  model_version: string | null;
}

export interface Position {
  id: number;
  symbol: string;
  window_open: string;
  settle_time: string;
  offset_minutes: number;
  side: Side;
  contracts: number;
  price: number;
  outlay: number;
  fee: number;
  model_probability: number;
  baseline_probability: number;
  edge: number;
  outcome: PositionOutcome;
  settled_up: boolean | null;
  payout: number | null;
  pnl: number | null;
  settled_at: string | null;
}

export interface AccountState {
  configured: boolean;
  starting_bankroll: number;
  bankroll: Measured<number>;
  /** Bankroll plus open stake carried at COST — never marked to our own
   *  forecast, which would book unrealised belief as profit. */
  equity: Measured<number>;
  staked: Measured<number>;
  open_positions: number;
  realized_pnl: Measured<number>;
  fees_paid: Measured<number>;
  halted: boolean;
  halted_reason: string | null;
  updated_at: string | null;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
  bankroll: number;
  staked: number;
  open_positions: number;
  realized_pnl: number;
}

export interface FunnelStage {
  reason: Reason;
  count: number;
  share: number | null;
}

export interface Gate {
  name: string;
  value: number;
  threshold: number;
  direction?: 'min' | 'max';
  passed: boolean;
}

export interface ModelState {
  present: boolean;
  reason: string | null;
  version?: string;
  created_at?: string;
  installed?: boolean;
  forced?: boolean;
  force_reason?: string | null;
  folds?: number | null;
  windows_evaluated?: number | null;
  log_loss_skill?: number | null;
  log_loss_skill_se?: number | null;
  folds_positive?: number | null;
  calibration_error?: number | null;
  residual_scale?: number | null;
  control_gain_share?: number | null;
  sharpe?: number | null;
  total_return?: number | null;
  gates?: Gate[];
  failed_gates?: string[];
  provenance?: Record<string, unknown>;
}

export interface ModelAttempt {
  version: string;
  created_at: string;
  installed: boolean;
  forced: boolean;
  log_loss_skill: number | null;
  folds_positive: number | null;
  windows_evaluated: number | null;
  failed_gates: string[];
  force_reason: string | null;
}

export interface CalibrationBin {
  source: 'model' | 'baseline';
  bin_low: number;
  bin_high: number;
  predicted: number | null;
  observed: number | null;
  count: number;
}

export interface CalibrationTable {
  version: string | null;
  reason: string | null;
  bins: CalibrationBin[];
}

export interface LiveState {
  windows: Prediction[];
  account: AccountState;
  open_positions: Position[];
}

export interface JobDescriptor {
  module: string;
  description: string;
}
