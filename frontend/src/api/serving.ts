/** Every request the app makes, in one place.
 *
 * One module rather than one per page: there are nine read-only routes and they
 * share a client, so splitting them across six files is how five copies of
 * `fetchWithError` drifted in the previous version until a failing request said
 * something different depending on which screen you were on.
 */
import { get, post } from './client';
import type {
  AccountState,
  CalibrationTable,
  EquityPoint,
  FunnelStage,
  JobDescriptor,
  LiveState,
  ModelAttempt,
  ModelState,
  OrderTicket,
  Position,
  Prediction,
  PriceSeries,
} from '../types';

export const fetchLive = () => get<LiveState>('/live');

export const fetchAccount = () => get<AccountState>('/account');

export const fetchEquity = (days = 30) =>
  get<{ days: number; points: EquityPoint[] }>(`/account/equity?days=${days}`);

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

export const fetchTickets = (openOnly = true, limit = 50) =>
  get<{ tickets: OrderTicket[] }>(`/tickets?open_only=${openOnly}&limit=${limit}`);

export const fetchModel = () => get<ModelState>('/model');

export const fetchModelHistory = (limit = 50) =>
  get<{ attempts: ModelAttempt[] }>(`/model/history?limit=${limit}`);

export const fetchCalibration = (version?: string) =>
  get<CalibrationTable>(
    version ? `/model/calibration?version=${encodeURIComponent(version)}` : '/model/calibration',
  );

export const fetchJobs = () => get<{ jobs: JobDescriptor[] }>('/jobs');

/** Authenticated, and fails closed: without API_TOKEN on the backend this is a
 *  503, which the client turns into "set VITE_API_TOKEN" rather than a generic
 *  failure. */
export const launchJob = (module: string, args: string[] = []) =>
  post<{ job: string; pid: number; args: string[]; note: string }>(
    `/jobs/${module}`,
    { args },
  );
