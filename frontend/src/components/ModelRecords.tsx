/** Settled P&L per model version, so a model's record can be read on its own.
 *
 *  The account total answers "how are we doing"; this answers "which model
 *  did that". They are different questions and the second was previously only
 *  answerable by joining positions to predictions by hand.
 */
import { useState } from 'react';

import { fetchModelPerformance } from '../api/serving';
import { usePolling } from '../hooks/usePolling';
import { Failed, Loading, Panel, SectionHead } from './Primitives';

const money = (v: number) => `${v >= 0 ? '+' : '−'}$${Math.abs(v).toFixed(2)}`;

function tone(v: number): string {
  if (v > 0) return 'text-up';
  if (v < 0) return 'text-down';
  return 'text-ink-3';
}

function day(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

export function ModelRecords({ pollMs }: { pollMs: number }) {
  const [allTime, setAllTime] = useState(true);
  const perf = usePolling(() => fetchModelPerformance(allTime), pollMs, [allTime]);

  return (
    <Panel>
      <SectionHead
        title="By model"
        note="Settled P&L for every model that has traded, newest first."
        right={
          <div className="flex rounded-md border border-line text-tiny">
            {([true, false] as const).map((value) => (
              <button
                key={String(value)}
                type="button"
                onClick={() => setAllTime(value)}
                className={`px-2 py-1 ${
                  allTime === value ? 'bg-line text-ink-1' : 'text-ink-3'
                }`}
              >
                {value ? 'All time' : 'Since reset'}
              </button>
            ))}
          </div>
        }
      />

      {perf.loading ? (
        <Loading what="the model records" />
      ) : perf.error ? (
        <Failed error={perf.error} what="the model records" />
      ) : (
        <>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-tiny">
              <thead className="text-ink-3">
                <tr className="text-left">
                  <th className="py-1 pr-3 font-normal">Model</th>
                  <th className="py-1 pr-3 text-right font-normal">Traded</th>
                  <th className="py-1 pr-3 text-right font-normal">P&L</th>
                  <th className="py-1 pr-3 text-right font-normal">Per pos.</th>
                  <th className="py-1 pr-3 text-right font-normal">Won</th>
                  <th className="py-1 text-right font-normal">Window</th>
                </tr>
              </thead>
              <tbody>
                {(perf.data?.models ?? []).map((m, i) => (
                  <tr key={m.model_version} className="border-t border-line">
                    <td className="py-1.5 pr-3 font-mono">
                      {m.model_version}
                      {i === 0 && (
                        <span className="ml-2 rounded bg-line px-1 text-ink-3">
                          current
                        </span>
                      )}
                    </td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {m.positions}
                    </td>
                    <td
                      className={`py-1.5 pr-3 text-right tabular-nums ${tone(m.pnl)}`}
                    >
                      {money(m.pnl)}
                    </td>
                    <td
                      className={`py-1.5 pr-3 text-right tabular-nums ${tone(
                        m.pnl_per_position ?? 0,
                      )}`}
                    >
                      {m.pnl_per_position == null
                        ? '—'
                        : m.pnl_per_position.toFixed(3)}
                    </td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {m.win_rate == null
                        ? '—'
                        : `${(100 * m.win_rate).toFixed(0)}%`}
                    </td>
                    <td className="py-1.5 text-right text-ink-3">
                      {day(m.first_window)} – {day(m.last_window)}
                    </td>
                  </tr>
                ))}

                {perf.data && perf.data.unattributed.positions > 0 && (
                  <tr className="border-t border-line text-ink-3">
                    <td className="py-1.5 pr-3 italic">no model recorded</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {perf.data.unattributed.positions}
                    </td>
                    <td
                      className={`py-1.5 pr-3 text-right tabular-nums ${tone(
                        perf.data.unattributed.pnl,
                      )}`}
                    >
                      {money(perf.data.unattributed.pnl)}
                    </td>
                    <td colSpan={3} />
                  </tr>
                )}

                {perf.data && (
                  <tr className="border-t border-line font-medium">
                    <td className="py-1.5 pr-3">Total</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {perf.data.total_positions}
                    </td>
                    <td
                      className={`py-1.5 pr-3 text-right tabular-nums ${tone(
                        perf.data.total_pnl,
                      )}`}
                    >
                      {money(perf.data.total_pnl)}
                    </td>
                    <td colSpan={3} />
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <p className="mt-2 text-tiny text-ink-3">
            Settled positions only. <strong>Won</strong> is the share that made
            money, not the share where the market rose — a down position
            settling down is a win. Read it beside per-position: a model can win
            most of its positions and still lose, by losing bigger. Rows with no
            model recorded predate the model being stamped on a prediction and
            are kept so the table reconciles with the account.
          </p>
        </>
      )}
    </Panel>
  );
}
