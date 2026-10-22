/** Every request the app makes, in one place. */
import { get } from './client';
import type {
  AccountState,
  CalibrationTable,
  CollectionHealth,
  EquityPoint,
  Funnel,
  LiveState,
  ModelAttempt,
  ModelDetail,
  ModelPerformance,
  ModelState,
  Position,
  Prediction,
  PriceSeries,
  VenueAccountState,
  VenueDaily,
  VenueEquityCurve,
  VenueSettlement,
} from '../types';

export const fetchLive = () => get<LiveState>('/live');

export const fetchAccount = () => get<AccountState>('/account');

/** Settled P&L per model version. `allTime` ignores the dashboard epoch — a
 *  model that traded before the last reset still did so. */
export const fetchModelPerformance = (allTime = false) =>
  get<ModelPerformance>(`/account/models${allTime ? '?all_time=true' : ''}`);

/** One model's own record: its settled positions and its P&L from zero. */
export const fetchModelDetail = (version: string, allTime = true) =>
  get<ModelDetail>(
    `/account/models/${encodeURIComponent(version)}?all_time=${allTime}`,
  );

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

/** Realised P&L per Eastern day. Not bounded by the dashboard epoch — a
 *  calendar is history, and the headline figures above it stay epoch-scoped. */
export const fetchVenueDaily = (days = 120) =>
  get<VenueDaily>(`/account/venue/daily?days=${days}`);

export const fetchPredictions = (limit = 100, tradedOnly = false) =>
  get<{ predictions: Prediction[] }>(
    `/predictions?limit=${limit}&traded_only=${tradedOnly}`,
  );

export const fetchFunnel = (days = 7) => get<Funnel>(`/funnel?days=${days}`);

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

/** Data-collection health: are rows still arriving, and are the loops still
 *  cycling. Two questions, served side by side and never merged — a recorder
 *  can cycle forever and write nothing, which is exactly what happened to
 *  `venue_ladder` for thirty-one days. */
export const fetchCollectionHealth = () => get<CollectionHealth>('/collection');
