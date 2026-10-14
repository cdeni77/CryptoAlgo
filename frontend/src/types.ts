/** Shapes the API returns.
 *
 * `Measured<T>` is a value that may be absent with a reason: a missing number
 * arrives as `{ value: null, reason: '...' }`, never as a zero. The UI renders
 * an em-dash with the reason on hover.
 *
 * Probabilities are the native unit. A binary's price is a probability, so
 * price, forecast and break-even share one scale.
 */

export interface Measured<T> {
  value: T | null;
  reason: string | null;
}

/** Why a decision was refused, in funnel order. Mirrors core.decide.Reason. */
export type Reason =
  | 'traded'
  | 'not_finite'
  | 'probability_invalid'
  | 'no_quote'
  | 'no_peer_book'
  | 'price_out_of_band'
  | 'disagreement_implausible'
  | 'edge_below_gate'
  | 'edge_uncertain'
  | 'members_disagree'
  | 'below_min_contracts'
  | 'fee_ceiling'
  | 'window_exposure'
  | 'position_limit'
  | 'already_entered'
  | 'offset_not_traded'
  | 'bankroll_floor'
  | 'halted'
  | 'model_not_pinned';

export type Side = 'up' | 'down';
export type PositionOutcome = 'pending' | 'won' | 'lost';

export interface MinuteBar {
  minute: string;
  open: number;
  high: number | null;
  low: number | null;
  close: number;
}

export interface WindowStrike {
  symbol: string;
  window_open: string;
  settle_time: string;
  strike: number;
}

export interface PriceSeries {
  symbol: string;
  minutes: number;
  bars: MinuteBar[];
  strikes: WindowStrike[];
}

export interface OrderTicket {
  id: number;
  symbol: string;
  window_open: string;
  settle_time: string;
  offset_minutes: number;
  market_ticker: string | null;
  side: Side;
  contracts: number;
  limit_price: number;
  max_price: number;
  expected_cost: number;
  model_probability: number;
  edge: number;
  status: 'new' | 'placed' | 'filled' | 'skipped';
  filled_contracts: number | null;
  filled_price: number | null;
  filled_at: string | null;
  note: string | null;
}

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
  /** The venue's implied probability, when a real book was read. */
  market_probability: number | null;
  /** 'quote' when the venue's ask priced this decision, 'baseline' when the
   *  calibrated barrier stood in for an unobserved market. Render them
   *  differently. */
  price_source: 'quote' | 'baseline';
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
  /** Real money or not. Surfaced everywhere a number from this account appears. */
  mode: 'paper' | 'live';
  starting_bankroll: number;
  bankroll: Measured<number>;
  /** Bankroll plus open stake at cost, never marked to our own forecast. */
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

/* ------------------------------------------------------- the venue's ledger */

/** The account as Kalshi's ledger reports it: the account of record.
 *
 *  `AccountState` is our own arithmetic. Both are rendered; `pnl_gap` is where
 *  they disagree.
 */
export interface VenueAccountState {
  /** False when no venue ledger is stored, as for a paper account. */
  available: boolean;
  reason: string | null;
  mode: 'paper' | 'live';
  settlements: number;
  realized_pnl: Measured<number>;
  fees: Measured<number>;
  /** Cash at the venue, from the last sample. Not equity: an open position has
   *  already left it. */
  balance: Measured<number>;
  win_rate: Measured<number>;
  wins: number;
  losses: number;
  /** Settled markets the venue named no result for, or that we held neither side
   *  of. Not losses. */
  undecided: number;
  /** Settlements excluded from `realized_pnl` because the venue left a field
   *  absent. Non-zero means the total is short by an unknown amount. */
  incomplete: number;
  contracts: number;
  first_settled: string | null;
  last_settled: string | null;
  our_realized_pnl: Measured<number>;
  /** Venue minus ours. */
  pnl_gap: Measured<number>;
  balance_drift: Measured<number>;
  balance_at: string | null;
  exchange_index: number | null;
}

/** One step of the realised-P&L curve: a market that settled. */
export interface VenuePnlPoint {
  timestamp: string;
  ticker: string;
  pnl: number;
  cumulative_pnl: number;
}

/** One sample of the venue's cash balance, taken every live cycle. */
export interface VenueBalancePoint {
  timestamp: string;
  balance: number;
  our_bankroll: number | null;
  drift: number | null;
  exchange_index: number | null;
}

export interface VenueEquityCurve {
  days: number;
  points: VenuePnlPoint[];
  balances: VenueBalancePoint[];
  /** Realised P&L that settled before the window opened. */
  pnl_before_window: number | null;
  realized_pnl_in_window: number | null;
}

export interface VenueSettlement {
  ticker: string;
  event_ticker: string | null;
  /** 'yes' or 'no', as the venue resolved it. */
  market_result: string | null;
  yes_contracts: number;
  no_contracts: number;
  contracts: number;
  cost: number | null;
  revenue: number | null;
  fee_cost: number | null;
  pnl: number | null;
  settled_time: string | null;
  our_pnl: number | null;
  pnl_gap: number | null;
  /** Did we win, per the venue's resolution, not the sign of the P&L. A
   *  favourite can win and still net negative after the fee. */
  won: boolean | null;
  synced_at: string | null;
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
  tickets: OrderTicket[];
}
