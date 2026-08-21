import { get } from './client';
import { FeatureImportanceResponse, LiveModel, PromotionHistory } from '../types';

/**
 * The promotion surface: what is live, what has been tried, and why the
 * rejections were rejected.
 *
 * Read-only by design. Promotion itself runs the gates, and the only thing that
 * can install a model is the thing that ran them — so the dashboard's promote
 * action goes through the authenticated `POST /research/launch/promote`, not
 * through here.
 */

export async function getLiveModel(): Promise<LiveModel> {
  return get('/model/');
}

export async function getPromotionHistory(limit = 50): Promise<PromotionHistory> {
  return get(`/model/promotions?limit=${limit}`);
}

export async function getFeatureImportance(head = 'price'): Promise<FeatureImportanceResponse> {
  return get(`/model/features?head=${head}`);
}
