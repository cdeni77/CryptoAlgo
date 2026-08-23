/** The account over time, and every settled position.
 *
 * The one number worth reading twice is the gap between predicted and realised
 * edge. Predicted is what the model claimed at entry; realised is the win rate
 * minus what was actually paid. A large gap in either direction is the winner's
 * curse, and it comes before any Sharpe ratio in the order of things to trust.
 */
import { usePolling } from '../hooks/usePolling';
import { fetchAccount, fetchEquity, fetchPositions } from '../api/serving';
import { EquityChart } from '../components/Charts';
import {
  Chip,
  Column,
  DataTable,
  Empty,
  Failed,
  Loading,
  Metric,
  Panel,
  SectionHead,
} from '../components/Primitives';
import { cents, pct, signedPp, stamp } from '../lib/format';
import type { Position } from '../types';

export function AccountPage() {
  const account = usePolling(fetchAccount, 30_000);
  const equity = usePolling(() => fetchEquity(90), 60_000);
  const positions = usePolling(() => fetchPositions(false, 200), 60_000);

  if (account.loading) return <Loading what="the account" />;
  if (account.error) return <Failed error={account.error} what="the account" />;
  const state = account.data;
  if (!state) return null;

  const settled = (positions.data?.positions ?? []).filter((p) => p.outcome !== 'pending');
  const won = settled.filter((p) => p.outcome === 'won').length;
  const paid = settled.reduce((sum, p) => sum + p.outlay, 0);
  const meanCost = settled.length ? paid / settled.reduce((s, p) => s + p.contracts, 0) : null;
  const winRate = settled.length ? won / settled.length : null;
  const realisedEdge = winRate != null && meanCost != null ? winRate - meanCost : null;
  const predictedEdge = settled.length
    ? settled.reduce((s, p) => s + p.edge, 0) / settled.length
    : null;

  return (
    <div className="space-y-8">
      <SectionHead
        eyebrow={`started at $${state.starting_bankroll.toFixed(2)}`}
        title="Account"
        note="Equity is cash plus open stake at cost. Marking an open binary at our own forecast would book belief as profit, which is how a losing system draws a rising curve."
        right={state.halted ? <Chip tone="fail">halted</Chip> : undefined}
      />

      <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
        <Panel>
          {equity.loading ? (
            <Loading what="the curve" />
          ) : equity.error ? (
            <Failed error={equity.error} what="the equity curve" />
          ) : (
            <EquityChart points={equity.data?.points ?? []} />
          )}
        </Panel>
        <Panel>
          <div className="grid grid-cols-2 gap-x-4 gap-y-4">
            <Metric label="equity" value={state.equity} unit="$" size="lg" />
            <Metric label="cash" value={state.bankroll} unit="$" size="lg" />
            <Metric label="at risk" value={state.staked} unit="$" />
            <Metric label="open" value={state.open_positions} digits={0} />
            <Metric
              label="realised p&l"
              value={state.realized_pnl}
              unit="$"
              tone={(state.realized_pnl.value ?? 0) >= 0 ? 'above' : 'below'}
            />
            <Metric label="fees" value={state.fees_paid} unit="$" tone="muted" />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-4 border-t border-rule pt-4">
            <Metric
              label="win rate"
              value={winRate == null ? null : winRate * 100}
              unit="%"
              digits={2}
              hint="a high win rate is expected: the system buys favourites, where the fee is cheapest"
            />
            <Metric label="settled" value={settled.length} digits={0} />
            <Metric
              label="edge predicted"
              value={predictedEdge == null ? null : predictedEdge * 100}
              unit="pp"
              digits={2}
              tone="muted"
              hint="what the model claimed at entry"
            />
            <Metric
              label="edge realised"
              value={realisedEdge == null ? null : realisedEdge * 100}
              unit="pp"
              digits={2}
              tone={(realisedEdge ?? 0) >= 0 ? 'above' : 'below'}
              hint="win rate minus mean cost paid. The gap against predicted is the winner's curse."
            />
          </div>
        </Panel>
      </div>

      <section>
        <SectionHead
          eyebrow={`${settled.length} settled`}
          title="Settled positions"
          note="One fee, at entry. Settlement is free, which is why nothing here exits early."
        />
        <Panel flush>
          {positions.loading ? (
            <Loading what="positions" />
          ) : (
            <DataTable
              columns={settledColumns}
              rows={settled}
              keyOf={(p) => String(p.id)}
              empty={
                <Empty
                  what="Nothing has settled yet."
                  next="python -m scripts.paper"
                />
              }
            />
          )}
        </Panel>
      </section>
    </div>
  );
}

const settledColumns: Column<Position>[] = [
  { key: 'when', head: 'window', render: (p) => <span className="font-mono">{stamp(p.window_open)}</span> },
  { key: 'symbol', head: 'symbol', render: (p) => <span className="font-mono">{p.symbol}</span> },
  {
    key: 'side',
    head: 'side',
    render: (p) => (
      <Chip tone={p.side === 'up' ? 'above' : 'below'}>{p.side}</Chip>
    ),
  },
  { key: 'offset', head: 'at', numeric: true, render: (p) => `+${p.offset_minutes}m` },
  { key: 'contracts', head: 'qty', numeric: true, render: (p) => p.contracts },
  { key: 'price', head: 'price', numeric: true, render: (p) => cents(p.price) },
  { key: 'q', head: 'forecast', numeric: true, render: (p) => pct(p.model_probability, 1) },
  { key: 'edge', head: 'edge', numeric: true, render: (p) => signedPp(p.edge) },
  { key: 'fee', head: 'fee', numeric: true, render: (p) => `$${p.fee.toFixed(2)}` },
  {
    key: 'result',
    head: 'result',
    render: (p) => (
      <Chip tone={p.outcome === 'won' ? 'pass' : 'fail'}>{p.outcome}</Chip>
    ),
  },
  {
    key: 'pnl',
    head: 'p&l',
    numeric: true,
    render: (p) =>
      p.pnl == null ? '—' : (
        <span className={p.pnl >= 0 ? 'text-above' : 'text-below'}>
          {p.pnl >= 0 ? '+' : ''}
          {p.pnl.toFixed(2)}
        </span>
      ),
  },
];
