/** Every request the app makes, in one place. */
import { get } from './client';
import type {
  AccountState,
  CalibrationTable,
  EquityPoint,
  FunnelStage,
  LiveState,
  ModelAttempt,
  ModelState,
  Position,
  Prediction,
  PriceSeries,
  VenueAccountState,
  VenueEquityCurve,
  VenueSettlement,
} from '../types';

export const fetchLive = () => get<LiveState>('/live');

export const fetchAccount = () => get<AccountState>('/account');

export const fetchEquity = (days = 30) =>
  get<{ days: number; points: EquityPoint[] }>(`/account/equity?days=${days}`);

/** The account as the venue's ledger reports it: the account of record.
 *  `fetchAccount` is our own arithmetic. See `VenueAccountState`. */
export const fetchVenueAccount = () => get<VenueAccountState>('/account/venue');

export const fetchVenueEquity = (days = 30) =>
  get<VenueEquityCurve>(`/account/venue/equity?days=${days}`);

export const fetchVenueSettlements = (limit = 200, days?: number) =>
  get<{ settlements: VenueSettlement[] }>(
    `/venue/settlements?limit=${limit}${days == null ? '' : `&days=${days}`}`,
  );

export const fetchPredictions = (limit = 100, tradedOnly = false) =>
  get<{ predictions: Prediction[] }>(
    `/predictions?limit=${limit}&traded_only=${tradedOnly}`,
  );

export const fetchFunnel = (days = 7) =>
  get<{ days: number; stages: FunnelStage[] }>(`/funnel?days=${days}`);

export const fetchPositions = (openOnly = false, limit = 100) =>
  get<{ positions: Position[] }>(
    `/positions?open_only=${openOnly}&limit=${limit}`,
  );

export const fetchPrices = (symbol: string, minutes = 240) =>
  get<PriceSeries>(`/prices/${encodeURIComponent(symbol)}?minutes=${minutes}`);

export const fetchModel = () => get<ModelState>('/model');

export const fetchModelHistory = (limit = 50) =>
  get<{ attempts: ModelAttempt[] }>(`/model/history?limit=${limit}`);

export const fetchCalibration = (version?: string) =>
  get<CalibrationTable>(
    version ? `/model/calibration?version=${encodeURIComponent(version)}` : '/model/calibration',
  );
