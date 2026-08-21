import { PaperPosition, PriceData } from '../types';

/**
 * Open positions, marked to the live price where there is one.
 *
 * Two things changed. The contract-size table that used to live here carried a
 * comment reading "Must match trading_costs.py" — a file that has since been
 * deleted, so nothing was checking the claim, and contract size multiplies
 * straight into unrealised PnL. It now comes from the caller, which reads
 * `/coins/cde-specs`.
 *
 * And the layout is a compact row rather than an eight-column table. This panel
 * sits in a third-width card, where eight columns silently clipped the unrealised
 * PnL — the one number the panel exists to show.
 */

const money = (v: number, digits = 2) =>
  v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });

interface Props {
  positions: PaperPosition[];
  prices?: PriceData | null;
  /** Contract size per instrument, or null for an instrument the specs do not
   *  cover. Either way the row falls back to the stored mark rather than
   *  marking against an invented size — `?? 1` here would be a 100x error on
   *  XRP and a 5000x one on DOGE. */
  unitsFor?: (coin: string) => number | null;
}

export default function PaperPositionsTable({ positions, prices, unitsFor }: Props) {
  const open = positions.filter((p) => p.is_open);

  if (!open.length) {
    return <div className="px-4 py-8 text-center text-sm text-tx-muted">No open positions</div>;
  }

  return (
    <div className="space-y-2">
      {open.map((p) => {
        const live = prices?.[p.coin as keyof typeof prices]?.price ?? null;
        const units = unitsFor?.(p.coin) ?? null;
        const sign = p.side === 'long' ? 1 : -1;
        const unrealised =
          live != null && units !== null
            ? p.contracts * units * (live - p.entry_price) * sign
            : p.unrealized_pnl;
        const mark = live ?? p.mark_price;

        return (
          <div
            key={p.id}
            className="rounded-lg border border-[rgba(56,189,248,0.08)] bg-surface-2/40 px-3 py-2.5"
          >
            <div className="flex items-baseline justify-between gap-2">
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-semibold text-tx-primary">{p.coin}</span>
                <span
                  className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
                    p.side === 'long'
                      ? 'bg-accent-emerald/10 text-accent-emerald'
                      : 'bg-accent-rose/10 text-accent-rose'
                  }`}
                >
                  {p.side.toUpperCase()}
                </span>
                <span className="font-mono text-[11px] text-tx-muted">{p.contracts}c</span>
              </div>
              <span
                className={`font-mono text-sm font-semibold tabular-nums ${
                  unrealised >= 0 ? 'text-accent-emerald' : 'text-accent-rose'
                }`}
              >
                {unrealised >= 0 ? '+' : '-'}${money(Math.abs(unrealised))}
              </span>
            </div>

            <div className="mt-1.5 flex items-baseline justify-between gap-2 font-mono text-[10px] text-tx-muted tabular-nums">
              <span>
                {money(p.entry_price, p.entry_price > 100 ? 2 : 4)}
                <span className="mx-1 opacity-50">→</span>
                {money(mark, mark > 100 ? 2 : 4)}
                {live != null && <span className="ml-1 text-accent-cyan opacity-70">live</span>}
              </span>
              {/* Fees and funding kept apart: on hourly-funding perps, a long
                  hold can pay more in funding than in commission. */}
              <span>
                fee ${money(p.fees_paid)} · fund ${money(p.funding_paid ?? 0)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
