/** What counts as an alert, in one place.
 *
 *  Every page used to decide for itself what "stale" or "bad" meant, so the
 *  same condition could read as a warning on one screen and nothing at all on
 *  another. These rules are the single source.
 *
 *  **A successful fetch is never evidence the trader is alive.** The API
 *  answers from Postgres; it will happily serve a decision from an hour ago
 *  with a 200. Liveness and freshness come from the timestamps INSIDE the
 *  payload, never from the fact that one arrived.
 */
import type { AccountState, CollectionHealth } from '../types';

export type Severity = 'critical' | 'warning' | 'info';

export interface Alert {
  id: string;
  severity: Severity;
  /** Short enough for a strip. No punctuation at the end. */
  title: string;
  /** One sentence of why it matters, shown when there is room. */
  detail?: string;
}

/** Seconds after which a decision is old enough to say so.
 *
 *  Decisions run at +3/+6/+9/+12 of a 15-minute window, so the longest
 *  legitimate gap is from +12 of one window to +3 of the next: six minutes.
 *  Ten gives a margin for a slow cycle without hiding a stopped loop.
 */
export const DECISION_STALE_SECONDS = 10 * 60;

export function secondsSince(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  return (Date.now() - then) / 1000;
}

/** Alerts worth interrupting someone for, most severe first. */
export function deriveAlerts(
  account: AccountState | null,
  health: CollectionHealth | null,
  lastDecisionAt: string | null,
): Alert[] {
  const out: Alert[] = [];

  if (account?.halted) {
    out.push({
      id: 'halted',
      severity: 'critical',
      title: 'Trading halted',
      detail: account.halted_reason ?? 'The account is halted and will not open new positions.',
    });
  }

  const decisionAge = secondsSince(lastDecisionAt);
  if (decisionAge != null && decisionAge > DECISION_STALE_SECONDS) {
    out.push({
      id: 'no-decisions',
      severity: 'critical',
      title: `No decision for ${Math.floor(decisionAge / 60)} minutes`,
      detail:
        'Decisions run at +3/+6/+9/+12 of every window, so a gap this long ' +
        'means the loop is not deciding — not that nothing qualified.',
    });
  }

  // A stale MEASUREMENT makes every dataset age below it an upper bound, so
  // it is reported before the individual datasets rather than beside them.
  if (health?.freshness?.measurement_stale) {
    out.push({
      id: 'freshness-measurement',
      severity: 'warning',
      title: 'Data freshness is itself stale',
      detail:
        'The trader publishes these ages; this reading is old, so every ' +
        'dataset age is an upper bound rather than a measurement.',
    });
  }

  // **A limit tighter than the measurement interval cannot be evaluated.**
  // The trader publishes these ages periodically; measured 2026-10-22, the
  // snapshot was 1,242s old while three datasets were judged against a 900s
  // limit. All three were writing rows that minute. Their reported age was
  // snapshot lag, not a stopped producer, and reporting it would light this
  // strip permanently -- and a strip that is always lit is one nobody reads.
  const measurementAge = health?.freshness?.measurement_age_seconds ?? null;
  const unresolvable: string[] = [];
  const stale = (health?.freshness?.datasets ?? []).filter((d) => {
    if (d.stale !== true) return false;
    if (measurementAge != null && d.limit_seconds != null
        && measurementAge > d.limit_seconds) {
      unresolvable.push(d.dataset);
      return false;
    }
    return true;
  });
  if (stale.length) {
    out.push({
      id: 'stale-datasets',
      severity: 'warning',
      title: `${stale.length} dataset${stale.length > 1 ? 's' : ''} not receiving rows`,
      detail: stale.map((d) => d.dataset).join(', '),
    });
  }
  if (unresolvable.length) {
    out.push({
      id: 'freshness-unresolvable',
      severity: 'info',
      title: `${unresolvable.length} dataset${unresolvable.length > 1 ? 's' : ''} cannot be judged`,
      detail:
        `The freshness snapshot is older than their limit, so their age is ` +
        `snapshot lag rather than a reading: ${unresolvable.join(', ')}.`,
    });
  }

  const dead = (health?.liveness?.components ?? []).filter((c) => c.stale === true);
  if (dead.length) {
    out.push({
      id: 'dead-components',
      severity: 'warning',
      title: `${dead.length} component${dead.length > 1 ? 's' : ''} not completing a pass`,
      detail: dead.map((c) => c.component).join(', '),
    });
  }

  return out;
}

/** The loudest severity present, or null. Drives the strip's colour. */
export function worstSeverity(alerts: Alert[]): Severity | null {
  if (alerts.some((a) => a.severity === 'critical')) return 'critical';
  if (alerts.some((a) => a.severity === 'warning')) return 'warning';
  return alerts.length ? 'info' : null;
}

/** How a freshness age should read. `null` means nothing was measured, which
 *  is NOT the same as fresh, and must never be shown as a pass. */
export function freshnessTone(stale: boolean | null): 'pass' | 'fail' | 'unknown' {
  if (stale === null) return 'unknown';
  return stale ? 'fail' : 'pass';
}
