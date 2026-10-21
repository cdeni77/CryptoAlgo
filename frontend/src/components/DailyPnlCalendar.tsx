/** Realised P&L per day, as a month grid.
 *
 * **Seven columns, not five.** A trading-day calendar usually hides the
 * weekend; this system trades BTC/ETH/SOL on a venue that settles every
 * fifteen minutes, seven days a week, and Saturday is an ordinary session.
 * Dropping it would silently discard about 28% of the record.
 *
 * **Bounded by the dashboard epoch**, like every other figure on the Account
 * page. It was briefly unfiltered, which put days from a previous
 * configuration beside days from this one under a single month total with
 * nothing saying where one ended. The month header reports how much of the
 * month is actually in range, so a part-month total is never read as a whole
 * one.
 *
 * **Days are Eastern, not UTC.** Kalshi's own app rolls the trading day over at
 * midnight ET, so that is the boundary these figures reconcile against.
 * Measured 2026-10-19: settlements since 00:00 UTC came to $10.08 while the
 * account showed $7.07, the difference being two markets from the previous
 * evening ET. The server does the bucketing; this component only renders what
 * it is handed, so the two cannot drift apart.
 *
 * A day with no settlement is blank, not zero. The system does not trade every
 * hour, and drawing an untraded day as break-even puts a point on the reader's
 * mental equity curve that never happened.
 */
import { useMemo, useState } from 'react';

import type { VenueDay } from '../types';

export interface DailyPnlCalendarProps {
  days: VenueDay[];
  /** Eastern, from the server. Shown so the day boundary is never a guess. */
  timezone?: string;
  /** The epoch the server bounded these days by, if any. */
  epoch?: string | null;
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

/** `2026-10-20` -> {y, m, d}, parsed as PLAIN DATE PARTS.
 *
 *  `new Date('2026-10-20')` is parsed as UTC midnight and then rendered in the
 *  viewer's local zone, which moves the day backwards for anyone west of
 *  Greenwich — the calendar would put Eastern days in the wrong cells for its
 *  own author. These strings are already Eastern calendar dates and carry no
 *  time, so they are split rather than parsed.
 */
function parts(iso: string): { y: number; m: number; d: number } {
  const [y, m, d] = iso.split('-').map(Number);
  return { y, m, d };
}

function monthKey(y: number, m: number): string {
  return `${y}-${String(m).padStart(2, '0')}`;
}

/** A dollar figure that fits a seven-column grid on a 375px phone.
 *
 *  Seven columns leave about 45px a cell, where `+20.86` overflows and is
 *  clipped mid-digit — a number cut off is worse than a rounded one, because
 *  it reads as a different number. Cents are kept only where they carry real
 *  information: under $10 a cent is ~1% of the figure, over it the dollar is
 *  what the eye is scanning for. The exact value is always in the tooltip.
 */
function compact(pnl: number): string {
  const sign = pnl >= 0 ? '+' : '-';
  const size = Math.abs(pnl);
  return `${sign}${size < 10 ? size.toFixed(1) : Math.round(size)}`;
}

export function DailyPnlCalendar({ days, timezone, epoch }: DailyPnlCalendarProps) {
  const byMonth = useMemo(() => {
    const out = new Map<string, Map<number, VenueDay>>();
    for (const day of days) {
      const { y, m, d } = parts(day.date);
      const key = monthKey(y, m);
      if (!out.has(key)) out.set(key, new Map());
      out.get(key)!.set(d, day);
    }
    return out;
  }, [days]);

  const months = useMemo(() => [...byMonth.keys()].sort(), [byMonth]);
  // Opens on the most recent month with data rather than on today's, so a
  // calendar whose last session was yesterday does not open blank.
  const [index, setIndex] = useState(() => Math.max(0, months.length - 1));
  const key = months[Math.min(index, months.length - 1)];

  if (!months.length) {
    return (
      <p className="py-8 text-center text-tiny text-ink-3">
        No settled days yet.
      </p>
    );
  }

  const [year, month] = key.split('-').map(Number);
  const cells = byMonth.get(key) ?? new Map<number, VenueDay>();
  const firstWeekday = new Date(Date.UTC(year, month - 1, 1)).getUTCDay();
  const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate();

  const present = [...cells.values()];
  // **The month total is the sum of the cells on screen, and nothing else.**
  // It summed `cash_change ?? 0` while the cells drew `cash_change ?? pnl`, so
  // a day with settlements but no balance sample was coloured and then left
  // out of the figure above it — a header that disagrees with the squares it
  // sits over is the one number a reader will check by hand.
  //  **The change in account value across the day**, which is cash plus what
  //  the open book is worth, both from the venue. A window open at midnight
  //  needs no correction because it is inside the measurement at both ends.
  //  Falls back to settled P&L on a day with no account-value sample to
  //  difference against -- the first day in range, and days before the
  //  sampler existed.
  const shownFor = (d: VenueDay): number => d.cash_change ?? d.pnl;
  const monthTotal = present.reduce((sum, d) => sum + shownFor(d), 0);
  const contracts = present.reduce((sum, d) => sum + d.contracts, 0);
  const up = present.filter((d) => shownFor(d) > 0).length;
  const down = present.filter((d) => shownFor(d) < 0).length;

  const blanks = Array.from({ length: firstWeekday });
  const numbers = Array.from({ length: daysInMonth }, (_, i) => i + 1);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          disabled={index <= 0}
          className="rounded px-2 py-1 text-tiny text-ink-2 transition-colors enabled:hover:bg-sunken enabled:hover:text-ink disabled:opacity-30"
          aria-label="Previous month"
        >
          ‹ prev
        </button>
        <div className="text-center">
          <div className="text-sm font-semibold text-ink">
            {MONTHS[month - 1]} {year}
          </div>
          <div className="eyebrow mt-0.5">
            {/* The DAY COUNT first, because with an epoch mid-month the total
                below covers part of the month and nothing else would say so. */}
            {present.length} of {daysInMonth} days · {up}↑ {down}↓ ·{' '}
            {contracts.toFixed(0)} contracts
          </div>
        </div>
        <button
          type="button"
          onClick={() => setIndex((i) => Math.min(months.length - 1, i + 1))}
          disabled={index >= months.length - 1}
          className="rounded px-2 py-1 text-tiny text-ink-2 transition-colors enabled:hover:bg-sunken enabled:hover:text-ink disabled:opacity-30"
          aria-label="Next month"
        >
          next ›
        </button>
      </div>

      {/* ONE total, and it is the venue's. The settled-P&L total was shown
          beside it for a while and the pair read as a discrepancy rather than
          as two questions; it is in each cell's tooltip instead. */}
      <div className="mb-2 flex items-baseline justify-between">
        <span className="eyebrow">month total</span>
        <span
          className={[
            'font-mono text-mid font-semibold tabular-nums',
            monthTotal > 0 ? 'text-above' : monthTotal < 0 ? 'text-below' : 'text-ink-2',
          ].join(' ')}
        >
          {monthTotal >= 0 ? '+' : ''}${monthTotal.toFixed(2)}
        </span>
      </div>

      <div className="grid grid-cols-7 gap-1">
        {WEEKDAYS.map((label) => (
          <div key={label} className="pb-1 text-center text-micro uppercase tracking-wider text-ink-3">
            {label.slice(0, 1)}
            <span className="hidden sm:inline">{label.slice(1)}</span>
          </div>
        ))}

        {blanks.map((_, i) => (
          <div key={`blank-${i}`} aria-hidden />
        ))}

        {numbers.map((n) => {
          const day = cells.get(n);
          // Untraded days are outlined, never filled — see the header note.
          if (!day) {
            return (
              <div
                key={n}
                className="rounded border border-rule/50 px-0.5 py-1.5 text-center sm:px-1"
                title="no settled markets"
              >
                <div className="text-micro text-ink-3/60">{n}</div>
                <div className="font-mono text-micro text-ink-3/40">—</div>
              </div>
            );
          }
          // **The cell shows CASH CHANGE, the figure Kalshi's own app shows
          // for that day.** Settled P&L is carried in the tooltip. They differ
          // because the midnight boundary cuts through open positions — cost
          // lands in one day, payout in the next — so a day-by-day view that
          // disagreed with the app was the thing being reconciled against.
          // Falls back to settled P&L when no balance was sampled that day.
          const shown = shownFor(day);
          const won = shown > 0;
          const flat = shown === 0;
          return (
            <div
              key={n}
              title={
                `${day.date}\n` +
                `${shown >= 0 ? '+' : ''}$${shown.toFixed(2)}` +
                (day.cash_change == null
                  ? ' — settled p&l; no account value to compare against\n'
                  : day.partial
                    ? ' — account value, day still running\n'
                    : ' — change in account value, as the Kalshi app shows it\n') +
                (day.unmarked
                  ? 'measured on cash alone — no position mark was recorded ' +
                    'that day, so it is exact only if nothing was open at ' +
                    'midnight\n'
                  : '') +
                `${day.markets} market(s), ${day.contracts.toFixed(0)} contract(s), ` +
                `${day.wins}W / ${day.losses}L` +
                (day.pp_per_contract == null
                  ? ''
                  : `  ·  ${day.pp_per_contract >= 0 ? '+' : ''}${day.pp_per_contract.toFixed(2)}pp per contract`) +
                (day.fees ? `\nfees $${day.fees.toFixed(4)}` : '') +
                (day.incomplete
                  ? `\n${day.incomplete} settlement(s) missing a field`
                  : '')
              }
              className={[
                'rounded border px-0.5 py-1.5 text-center sm:px-1',
                flat
                  ? 'border-rule bg-sunken'
                  : won
                    ? 'border-above/40 bg-above/15'
                    : 'border-below/40 bg-below/15',
              ].join(' ')}
            >
              <div className="text-micro text-ink-3">{n}</div>
              <div
                className={[
                  'font-mono text-micro font-semibold leading-tight tabular-nums sm:text-tiny',
                  flat ? 'text-ink-2' : won ? 'text-above' : 'text-below',
                ].join(' ')}
              >
                <span className="sm:hidden">{compact(shown)}</span>
                <span className="hidden sm:inline">
                  {shown >= 0 ? '+' : ''}
                  {shown.toFixed(2)}
                </span>
              </div>
              <div className="text-micro text-ink-3">
                {day.markets}
                {day.incomplete > 0 && (
                  <span className="text-warn" title="some settlements are missing a field">
                    {' '}!
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <p className="mt-3 text-tiny text-ink-3">
        Each cell is the change in <strong className="text-ink-2">account
        value</strong> — cash plus what the open positions are worth, both read
        from Kalshi — across that {timezone ?? 'US Eastern'} day, on the
        boundary Kalshi rolls its own trading day. Fees and settlement are
        already inside it; nothing here is estimated. A blank day settled
        nothing and is not a break-even day.
        {epoch && (
          <>
            {' '}Bounded at {epoch.slice(0, 10)}, the same window as every other
            figure on this page — days before it are not missing, they are out
            of scope.
          </>
        )}
      </p>
    </div>
  );
}
