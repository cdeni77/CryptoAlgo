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
  /** The window is already held on the opposite side. */
  | 'opposite_side'
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
  /** The same quantity read for the SIDE TAKEN. `market_probability` stays
   *  P(up) because the promotion gate compares in up terms; this one is what
   *  may be plotted against `model_probability`, which is already
   *  side-relative. Mixing them put the bar and the caret on opposite axes for
   *  every down row. */
  market_probability_side: number | null;
  /** 'quote' when the venue's ask priced this decision, 'baseline' when the
   *  calibrated barrier stood in for a market that was not observed. Those are
   *  different claims and the UI must not render them identically. */
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
  /** What the VENUE's bid says one contract is worth now. Null until a quote
   *  has been read for it, and meaningless once settled — `pnl` answers then. */
  mark_price: number | null;
  mark_at: string | null;
  mark_value: number | null;
  /** Marked value minus what we paid. Unrealised: a binary goes to the full
   *  win or the full loss at settlement, so this is a position tracker, not
   *  profit. */
  unrealized_pnl: number | null;
}

export interface AccountState extends AccountMarks {
  configured: boolean;
  /** Real money or not. Surfaced everywhere a number from this account appears. */
  mode: 'paper' | 'live';
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

/* ------------------------------------------------------- the venue's ledger */

/** The account as Kalshi's own ledger reports it.
 *
 *  `AccountState` above is our arithmetic: a bankroll debited at the price
 *  `decide()` sized at, a fee predicted from the published schedule, and a payout
 *  decided by an OHLC mean of Coinbase standing in for sixty seconds of CF
 *  Benchmarks BRTI. This is the account of record — `/portfolio/settlements` and
 *  `/portfolio/balance`, stored per settled market.
 *
 *  Both are rendered, and neither is substituted for the other. `pnl_gap` is
 *  where they disagree, which is a measurement (a mispriced fee, a settlement our
 *  proxy called differently, a fill nobody booked) rather than an error to hide.
 */
/** One Eastern calendar day of the venue's own settlements. */
/** Open-position marking, added to `AccountState`. */
export interface AccountMarks {
  /** Cash plus open contracts at the VENUE's bid. What the account is worth
   *  right now, as opposed to `equity`, which carries them at cost. */
  account_value: Measured<number>;
  /** The open contracts' share of `account_value`. */
  mark_value: Measured<number>;
  /** Marked minus cost — the open P&L, which is what moves between cycles. */
  unrealized_pnl: Measured<number>;
  /** Positions still carried at cost because no quote has been read for them.
   *  Non-zero means part of `mark_value` is not actually live. */
  unmarked_positions: number;
  /** When the OLDEST quote inside `mark_value` was read. */
  marked_at: string | null;
  open_contracts: number;
  /** Cash across EVERY shard — what the venue's own app shows, and what
   *  `account_value` is built on. */
  total_balance: Measured<number>;
  /** The trading shard alone: the only cash a crypto order can draw on, and
   *  what Kelly sizes off. */
  tradeable_balance: Measured<number>;
  /** `total_balance - tradeable_balance`: real money that cannot reach these
   *  markets. */
  stranded: Measured<number>;
}

export interface VenueDay {
  /** ISO date in US Eastern, the boundary Kalshi's own app rolls the day on. */
  date: string;
  pnl: number;
  markets: number;
  contracts: number;
  fees: number;
  wins: number;
  losses: number;
  /** Settlements the venue left a field absent on: the day's total is short. */
  incomplete: number;
  /** P&L per contract, in probability points. Null when nothing settled. */
  pp_per_contract: number | null;
  /** Cash-balance change over the same Eastern day — **the number Kalshi's own
   *  app shows**. Differs from `pnl` because a window bought before midnight
   *  pays out after it, so cost lands in one day and revenue in the next; on a
   *  completed day the two agree within a few cents. It also moves on
   *  deposits, which is why it is not the primary series. */
  cash_change: number | null;
  /** Always 0 now, and kept only so the payload shape is stable. The boundary
   *  correction it used to carry no longer exists: `cash_change` is a change
   *  in ACCOUNT VALUE, which already contains whatever was open at each end. */
  open_value: number | null;
  /** No position mark was recorded that day, so the figure is cash only and is
   *  exact only if the book was flat at midnight. True for every day before
   *  the per-second sampler existed. */
  unmarked?: boolean;
  /** The day is still running, so `cash_change` is partial. */
  partial: boolean;
}

export interface VenueDaily {
  timezone: string;
  /** The dashboard epoch these days are bounded by; null means the full
   *  ledger. Carried so the surface can say what it shows. */
  epoch: string | null;
  days: VenueDay[];
  first: string | null;
  last: string | null;
  total: number;
}

export interface VenueAccountState {
  /** False when no venue ledger has been stored — the normal state of a paper
   *  account, and the reason says so rather than showing $0.00 realised. */
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
  /** Markets the venue settled that no position of ours maps to — a fill
   *  nobody booked. Counted, not inferred from a dollar gap: such a market
   *  can settle at exactly break-even and move no total at all. */
  unbooked: number;
  unbooked_pnl: Measured<number>;
  balance_drift: Measured<number>;
  balance_at: string | null;
  exchange_index: number | null;
  /** Cash across EVERY shard — the figure the venue's own app shows. `balance`
   *  above is the trading shard alone, which is the only money a crypto order
   *  can draw on. Showing one without the other is why this page read $482.46
   *  against an app saying $483.57 with drift at exactly 0.00. */
  total_balance: Measured<number>;
  /** `total_balance` minus `balance`: cash the trading shard cannot reach. */
  stranded: Measured<number>;
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
  /** Realised P&L that settled before the window opened, so the series can be
   *  read on a lifetime scale rather than mistaken for one. */
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
  /** Did we win, per the venue's resolution — not per the sign of the P&L. A
   *  favourite bought at 97c can win and still net negative after the fee. */
  won: boolean | null;
  synced_at: string | null;
}

export interface FunnelStage {
  reason: Reason;
  count: number;
  share: number | null;
  /** False for `traded`, which is in the list so the shares sum to the whole
   *  population — but is not a decline, and should not lead a chart titled
   *  "why it declined". */
  declined: boolean;
}

export interface Funnel {
  stages: FunnelStage[];
  /** What was asked for. */
  requested_days: number;
  /** What the dashboard epoch actually left — the span the counts cover.
   *  These differed by 7 vs 2.2 days with nothing on screen saying so. */
  covered_days: number;
  epoch: string | null;
  total: number;
}

export interface Gate {
  name: string;
  value: number;
  threshold: number;
  direction?: 'min' | 'max';
  passed: boolean;
  /** Whether the gate could be COMPUTED at all. The market comparisons read
   *  NaN until the live loop has recorded enough quotes, and arrived here as
   *  `passed: false` — rendering identically to a gate that was measured and
   *  lost. Those are different claims. */
  measured?: boolean;
  /** One sentence on what this gate means and why it has that threshold.
   *  Written in `core/metrics.py` and, until 2026-10-20, never sent. */
  note?: string;
}

/** Sharpe after accounting for how many times we looked. */
export interface DeflatedSharpe {
  sharpe: number | null;
  n_trials: number | null;
  expected_max_sharpe: number | null;
  probabilistic_sharpe: number | null;
  deflated_sharpe: number | null;
}

/** Statistics the evaluator computes and the ledger keeps. */
export interface ReportStats {
  log_loss_skill_se?: number | null;
  log_loss_skill_t?: number | null;
  log_loss_skill_p?: number | null;
  effective_folds?: number | null;
  sign_agreement_p?: number | null;
  mean_residual_scale?: number | null;
  median_residual_scale?: number | null;
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
  /** True when this is the artifact the live signal writer loads. */
  is_live?: boolean;
  /** A newer attempt that did NOT install, if there is one. */
  latest_candidate?: {
    version: string;
    created_at: string;
    installed: boolean;
    forced: boolean;
    failed_gates: string[];
  } | null;
  /** The multiple-testing denominator, and the figure that applies it. */
  trial_count?: number | null;
  deflated_sharpe?: DeflatedSharpe | null;
  gate_spec_hash?: string | null;
  /** The alpha baked into the ARTIFACT — a different quantity from the
   *  `residual_scale` gate, which measures it across the folds. Measured
   *  2026-10-20 they were 0.0151 and 0.9404: same name, 62x apart. */
  deployed_residual_scale?: number | null;
  report_stats?: ReportStats;
}

export interface ModelAttempt {
  version: string;
  created_at: string;
  installed: boolean;
  forced: boolean;
  log_loss_skill: number | null;
  folds_positive: number | null;
  /** The denominator. A count of agreeing folds is only comparable at a fixed
   *  fold count: "5 of 10" is a 62% event under the null, "5 of 6" a 10.9%
   *  one. Without this the ledger sorted 6/6 below 25/29. */
  folds: number | null;
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

/* ----------------------------------------------------- collection health -- */

/** Is a dataset still receiving rows?
 *
 *  Not the same question as whether the process writing it is alive, and the
 *  difference cost thirty-one days: `venue_ladder` stopped writing on
 *  2026-09-16 and every component heartbeat stayed green the whole time,
 *  because the recorder's loop ran happily and produced nothing.
 *
 *  `stale` is `null`, never `false`, whenever nothing was measured — a dataset
 *  with no rows, or one with no continuous producer and therefore no age to
 *  expect. `reason` says which.
 */
export interface DatasetFreshness {
  dataset: string;
  newest_event_time: string | null;
  age_seconds: number | null;
  limit_seconds: number | null;
  stale: boolean | null;
  reason: string | null;
}

export interface FreshnessReport {
  available: boolean;
  reason: string | null;
  /** When the trader last measured this. The API cannot read a Parquet footer,
   *  so the measurement is made trader-side and published; if it is old, every
   *  age below it is an upper bound rather than a reading. */
  measured_at: string | null;
  measurement_age_seconds: number | null;
  measurement_stale: boolean | null;
  datasets: DatasetFreshness[];
}

/** Did a component's loop finish a pass? Process liveness, not data freshness. */
export interface ComponentLiveness {
  component: string;
  age_seconds: number | null;
  limit_seconds: number | null;
  stale: boolean | null;
  reason: string | null;
}

export interface LivenessReport {
  available: boolean;
  reason: string | null;
  components: ComponentLiveness[];
}

export interface CollectionHealth {
  freshness: FreshnessReport;
  liveness: LivenessReport;
}
