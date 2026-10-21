/** Number and time formatting. Not components, so not in a component module —
 *  a file that exports both cannot hot-reload cleanly, and eslint is right about
 *  it.
 *
 *  Everything here degrades to an em-dash rather than to `NaN` or `0`. A missing
 *  measurement must never render as a number, which is the same rule the API
 *  follows on the other side of the wire.
 */

export function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e4) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(digits);
}


export function pct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}


export function cents(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(0)}¢`;
}


export function signedPp(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '—';
  const v = value * 100;
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}pp`;
}


/** **Every time on this dashboard is US Eastern.**
 *
 *  It used to be three zones at once, which is worse than any one of them:
 *  `stamp` formatted through `toISOString()` and was therefore always UTC,
 *  `clock` passed `undefined` to `toLocaleTimeString` and was therefore
 *  whatever zone the viewing device happened to be in, and the daily calendar
 *  buckets by Eastern. So a settlement row could read 14:00 while the calendar
 *  counted it under a different day, and the same page read differently on a
 *  laptop and on a phone in another timezone.
 *
 *  Eastern rather than UTC or local because it is the venue's own day
 *  boundary: Kalshi rolls its trading day at midnight ET, and these figures
 *  are reconciled against that account. Measured 2026-10-19, settlements since
 *  00:00 UTC came to $10.08 against the app's $7.07 — the difference was two
 *  markets from the previous evening ET.
 *
 *  Named explicitly rather than taken from the browser, so the page says the
 *  same thing wherever it is opened.
 */
export const DISPLAY_TIMEZONE = 'America/New_York';

const TIME_FMT = new Intl.DateTimeFormat('en-US', {
  timeZone: DISPLAY_TIMEZONE,
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});

const STAMP_FMT = new Intl.DateTimeFormat('en-US', {
  timeZone: DISPLAY_TIMEZONE,
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});


export function clock(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return TIME_FMT.format(d);
}


export function stamp(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  // `MM-DD HH:MM`, the shape the charts' axes were built around. The parts are
  // assembled by hand because `formatToParts` is the only way to get this
  // ordering out of `Intl` without a locale that happens to agree.
  const parts = Object.fromEntries(
    STAMP_FMT.formatToParts(d).map((p) => [p.type, p.value]),
  );
  return `${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`;
}


/** An age in seconds, as the coarsest unit that still says something.
 *
 *  `31d` and `2m 54s` answer the same question at the two scales this
 *  dashboard needs: a feed that died a month ago, and one that is a couple of
 *  minutes behind. Seconds are dropped above an hour because at that range
 *  they are noise, and the em-dash is for the case the whole surface exists to
 *  make visible — an age nobody measured, which must never render as `0s`.
 */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}
