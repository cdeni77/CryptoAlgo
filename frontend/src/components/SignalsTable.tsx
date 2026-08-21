import { Signal } from '../types';

/**
 * What the model proposed, and what it expected to earn for it.
 *
 * The columns changed with the model. This table showed `Mom`, `Trend`, `ML` and
 * `AUC` — the gates and score of a classifier that no longer exists, so all four
 * now come back null and rendered as four columns of dashes. What the model
 * produces instead is a net-return forecast split into price and carry, against
 * a round-trip cost it has to clear, so those are the columns.
 *
 * Cost is shown beside the expected edge on purpose: a +12bp forecast is a trade
 * on DOGE at 5bp round trip and a loss on ETH at 54bp, and the two are
 * indistinguishable without the cost next to them.
 */

interface Props {
  signals: Signal[];
  limit?: number;
}

const bps = (v: number | null) => (v === null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}`);

export default function SignalsTable({ signals, limit = 20 }: Props) {
  const rows = signals.slice(0, limit);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[rgba(56,189,248,0.08)]">
            {['Time', 'Coin', 'Dir', 'Net', 'Price', 'Carry', 'Cost', 'E/R', 'Result'].map((h) => (
              <th
                key={h}
                className="px-2 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-tx-muted"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={9} className="px-2 py-6 text-center text-tx-muted">
                No signals yet
              </td>
            </tr>
          )}
          {rows.map((s) => (
            <tr
              key={s.id}
              className="border-b border-[rgba(56,189,248,0.04)] transition-colors hover:bg-[rgba(56,189,248,0.03)]"
            >
              <td className="whitespace-nowrap px-2 py-1.5 font-mono text-tx-muted">
                {new Date(s.timestamp).toLocaleTimeString('en-GB', {
                  hour12: false,
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </td>
              <td className="px-2 py-1.5 font-medium text-tx-secondary">{s.coin}</td>
              <td className="px-2 py-1.5">
                <span
                  className={`font-mono font-semibold ${
                    s.direction === 'long'
                      ? 'text-accent-emerald'
                      : s.direction === 'short'
                        ? 'text-accent-rose'
                        : 'text-tx-muted'
                  }`}
                >
                  {s.direction === 'long' ? '▲' : s.direction === 'short' ? '▼' : '—'}
                </span>
              </td>
              <td
                className={`px-2 py-1.5 font-mono tabular-nums ${
                  (s.expected_net_bps ?? 0) > 0 ? 'text-accent-emerald' : 'text-tx-secondary'
                }`}
              >
                {bps(s.expected_net_bps)}
              </td>
              <td className="px-2 py-1.5 font-mono text-tx-muted tabular-nums">
                {bps(s.expected_price_bps)}
              </td>
              <td className="px-2 py-1.5 font-mono text-tx-muted tabular-nums">
                {bps(s.expected_carry_bps)}
              </td>
              <td className="px-2 py-1.5 font-mono text-tx-muted tabular-nums">
                {s.cost_bps === null ? '—' : s.cost_bps.toFixed(1)}
              </td>
              <td className="px-2 py-1.5 font-mono text-tx-secondary tabular-nums">
                {s.edge_to_risk === null ? '—' : s.edge_to_risk.toFixed(2)}
              </td>
              <td className="px-2 py-1.5">
                {s.passed_gates ? (
                  <span className="text-[10px] text-accent-cyan">
                    {s.contracts_suggested ? `${s.contracts_suggested}c` : 'pass'}
                  </span>
                ) : (
                  <span
                    className="text-[10px] text-tx-muted"
                    title={s.gate_failure_reason ?? undefined}
                  >
                    {s.gate_failure_reason ?? 'blocked'}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
