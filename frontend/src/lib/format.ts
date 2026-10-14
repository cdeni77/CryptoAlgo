/** Number and time formatting, kept out of component modules for hot reload.
 *
 *  A missing value renders as an em-dash, never as `NaN` or `0`.
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


export function clock(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}


export function stamp(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return `${d.toISOString().slice(5, 10)} ${d.toISOString().slice(11, 16)}`;
}
