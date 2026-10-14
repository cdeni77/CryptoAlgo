/** The account, as the venue's ledger reports it.
 *
 * Headline numbers come from `/portfolio/settlements` and `/portfolio/balance`,
 * the account of record. Our own figures sit beside them; the reconciliation
 * panel shows the gap, and a growing drift is the alarm.
 *
 * Open positions are never marked: a binary held to expiry has one price that
 * matters, the settlement. The public trade tape is not used: it has no
 * account attribution.
 */
import { usePolling } from '../hooks/usePolling';
import {
  fetchAccount,
  fetchEquity,
  fetchPositions,
  fetchVenueAccount,
  fetchVenueEquity,
  fetchVenueSettlements,
} from '../api/serving';
import { EquityChart, VenueAccountChart } from '../components/Charts';
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
import type { Position, VenueSettlement } from '../types';

export function AccountPage() {
  const account = usePolling(fetchAccount, 30_000);
  // Polled faster: the live loop samples the venue's balance every cycle.
  const venue = usePolling(fetchVenueAccount, 20_000);
  const venueCurve = usePolling(() => fetchVenueEquity(30), 30_000);
  const equity = usePolling(() => fetchEquity(90), 60_000);
  const settlements = usePolling(() => fetchVenueSettlements(200), 60_000);
  const positions = usePolling(() => fetchPositions(false, 200), 60_000);

  if (account.loading) return <Loading what="the account" />;
  if (account.error) return <Failed error={account.error} what="the account" />;
  const state = account.data;
  if (!state) return null;

  const ledger = venue.data;
  // The venue has a ledger only once real orders have filled. Until then this
  // page shows our arithmetic and says so.
  const onVenue = Boolean(ledger?.available);

  const settled = (positions.data?.positions ?? []).filter((p) => p.outcome !== 'pending');
  const settledContracts = settled.reduce((s, p) => s + p.contracts, 0);

  // Per trade: the venue resolves markets, not contracts. Never subtract a
  // per-contract cost from it.
  const winRate = onVenue
    ? ledger!.win_rate.value
    : settled.length
      ? settled.filter((p) => p.outcome === 'won').length / settled.length
      : null;

  // Realised P&L per contract, not `win_rate - mean_cost`, which mixes a
  // per-trade rate with a per-contract cost. This is the quantity the
  // `realised_edge_pp` gate uses.
  const realisedEdge = onVenue
    ? ledger!.contracts > 0 && ledger!.realized_pnl.value != null
      ? ledger!.realized_pnl.value / ledger!.contracts
      : null
    : settledContracts > 0
      ? settled.reduce((s, p) => s + (p.pnl ?? 0), 0) / settledContracts
      : null;

  // Contract-weighted, to compare like with like against `realisedEdge`.
  const predictedEdge = settledContracts
    ? settled.reduce((s, p) => s + p.edge * p.contracts, 0) / settledContracts
    : null;

  const venueRows = settlements.data?.settlements ?? [];
  const beforeWindow = venueCurve.data?.pnl_before_window ?? null;

  return (
    <div className="space-y-8">
      <SectionHead
        eyebrow={`${state.mode === 'live' ? 'live · real money' : 'paper'} · started at $${state.starting_bankroll.toFixed(2)}`}
        title="Account"
        note={
          onVenue
            ? 'Every figure below the chart is the venue’s: cost, revenue and fee per settled market, from /portfolio/settlements. Ours is kept beside it, and the gap is the measurement.'
            : 'No venue ledger yet, so these are our own figures — a bankroll debited at the price we sized at and a payout decided by our own bars. The venue has a ledger once real orders fill.'
        }
        right={
          <div className="flex items-center gap-4">
            <Chip
              tone={onVenue ? 'pass' : 'neutral'}
              title={
                onVenue
                  ? 'P&L from the venue’s own settlements — the account of record'
                  : (ledger?.reason ?? 'the venue ledger has not been read yet')
              }
            >
              {onVenue ? 'venue ledger' : 'our books'}
            </Chip>
            {state.mode === 'live' && <Chip tone="below">live</Chip>}
            {state.halted && <Chip tone="fail">halted</Chip>}
          </div>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
        <Panel>
          {onVenue ? (
            venueCurve.loading ? (
              <Loading what="the curve" />
            ) : venueCurve.error ? (
              <Failed error={venueCurve.error} what="the venue curve" />
            ) : (
              <>
                <VenueAccountChart points={venueCurve.data?.points ?? []} />
                <p className="mt-2 text-tiny text-ink-3">
                  Realised P&amp;L, stepped once per settled market. The venue’s cash
                  is the figure beside this chart, not a line on it: it moves on
                  deposits and withdrawals too, and those are not the strategy.{' '}
                  {beforeWindow != null && (
                    <>
                      Cumulative within the last {venueCurve.data?.days ?? 30} days;{' '}
                      {beforeWindow >= 0 ? '+' : ''}${beforeWindow.toFixed(2)} settled before
                      that and is not in the series.
                    </>
                  )}
                </p>
              </>
            )
          ) : equity.loading ? (
            <Loading what="the curve" />
          ) : equity.error ? (
            <Failed error={equity.error} what="the equity curve" />
          ) : (
            <EquityChart points={equity.data?.points ?? []} />
          )}
        </Panel>
        <Panel>
          <div className="grid grid-cols-2 gap-x-4 gap-y-4">
            <Metric
              label={onVenue ? 'realised p&l · venue' : 'realised p&l'}
              value={onVenue ? ledger!.realized_pnl : state.realized_pnl}
              unit="$"
              size="lg"
              tone={
                ((onVenue ? ledger!.realized_pnl.value : state.realized_pnl.value) ?? 0) >= 0
                  ? 'above'
                  : 'below'
              }
              hint={
                onVenue
                  ? 'revenue minus cost minus fee, summed over every market the venue settled'
                  : 'our own arithmetic — the venue has not been read'
              }
            />
            <Metric
              label={onVenue ? 'cash · venue' : 'cash'}
              value={onVenue ? ledger!.balance : state.bankroll}
              unit="$"
              size="lg"
              hint={
                onVenue
                  ? `from /portfolio/balance${
                      ledger!.exchange_index == null
                        ? ''
                        : `, shard ${ledger!.exchange_index}`
                    }. Cash, not equity: an open position has already left it.`
                  : undefined
              }
            />
            <Metric
              label="at risk"
              value={state.staked}
              unit="$"
              hint="open stake at cost. Never marked to our own forecast, which would book conviction as profit."
            />
            <Metric label="open" value={state.open_positions} digits={0} />
            <Metric
              label={onVenue ? 'fees · venue' : 'fees'}
              value={onVenue ? ledger!.fees : state.fees_paid}
              unit="$"
              tone="muted"
              hint={
                onVenue
                  ? 'as charged, not as predicted from the published schedule'
                  : 'predicted from the published fee schedule'
              }
            />
            <Metric
              label="settled"
              value={onVenue ? ledger!.settlements : settled.length}
              digits={0}
              hint={onVenue ? 'markets the venue has resolved' : undefined}
            />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-4 border-t border-rule pt-4">
            <Metric
              label="win rate"
              value={winRate == null ? null : winRate * 100}
              unit="%"
              digits={2}
              hint={
                onVenue
                  ? 'from the venue’s own market_result against the side we held — not from the sign of the P&L, because a favourite bought at 97c can win and still net negative after the fee'
                  : 'a high win rate is expected: the system buys favourites, where the fee is cheapest'
              }
            />
            <Metric
              label="contracts"
              value={onVenue ? ledger!.contracts : null}
              digits={0}
              tone="muted"
              hint={onVenue ? undefined : 'the venue ledger has not been read'}
            />
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
          {onVenue && ledger!.incomplete > 0 && (
            <p className="mt-4 border-t border-rule pt-4 text-tiny text-warn">
              {ledger!.incomplete} settlement
              {ledger!.incomplete === 1 ? '' : 's'} had a field the venue did not serve and
              are excluded from the P&amp;L above. The total is short by an unknown amount,
              not by zero.
            </p>
          )}
        </Panel>
      </div>

      {onVenue && (
        <section>
          <SectionHead
            eyebrow="ours against theirs"
            title="Reconciliation"
            note="The venue is the account of record, so a gap is our error. Its size matters less than whether it grows: a stable gap is usually a starting-balance mismatch, a growing one is an unrecorded fill, a partial, or a fee we mispriced."
          />
          <Panel>
            <div className="grid grid-cols-2 gap-x-4 gap-y-4 lg:grid-cols-4">
              <Metric
                label="p&l · venue"
                value={ledger!.realized_pnl}
                unit="$"
                hint="revenue minus cost minus fee, per settled market"
              />
              <Metric
                label="p&l · ours"
                value={ledger!.our_realized_pnl}
                unit="$"
                tone="muted"
                hint="settled from our own bars, with the fee predicted from the schedule"
              />
              <Metric
                label="p&l gap"
                value={ledger!.pnl_gap}
                unit="$"
                tone={Math.abs(ledger!.pnl_gap.value ?? 0) > 0.01 ? 'warn' : 'muted'}
                hint="venue minus ours"
              />
              <Metric
                label="balance drift"
                value={ledger!.balance_drift}
                unit="$"
                tone={Math.abs(ledger!.balance_drift.value ?? 0) > 0.01 ? 'warn' : 'muted'}
                hint="the venue's balance minus ours, at the last sample"
              />
            </div>
            <p className="mt-4 text-tiny text-ink-3">
              Last balance sample {stamp(ledger!.balance_at)}
              {ledger!.last_settled && <> · last settlement {stamp(ledger!.last_settled)}</>}
              {ledger!.undecided > 0 && (
                <>
                  {' '}
                  · {ledger!.undecided} settled market
                  {ledger!.undecided === 1 ? '' : 's'} the venue named no result for
                </>
              )}
            </p>
          </Panel>
        </section>
      )}

      <section>
        <SectionHead
          eyebrow={onVenue ? `${venueRows.length} settled · venue` : `${settled.length} settled`}
          title={onVenue ? 'Settled markets' : 'Settled positions'}
          note={
            onVenue
              ? 'What the venue paid, per market: cost basis, payout, and the fee it actually charged. One fee, at entry — settlement is free, which is why nothing here exits early.'
              : 'One fee, at entry. Settlement is free, which is why nothing here exits early.'
          }
        />
        <Panel flush>
          {onVenue ? (
            settlements.loading ? (
              <Loading what="settlements" />
            ) : settlements.error ? (
              <Failed error={settlements.error} what="the venue settlements" />
            ) : (
              <DataTable
                columns={venueColumns}
                rows={venueRows}
                keyOf={(r) => r.ticker}
                empty={
                  <Empty
                    what="The venue has settled nothing yet."
                    next="python -m scripts.sync_venue"
                  />
                }
              />
            )
          ) : positions.loading ? (
            <Loading what="positions" />
          ) : (
            <DataTable
              columns={settledColumns}
              rows={settled}
              keyOf={(p) => String(p.id)}
              empty={
                <Empty
                  what="Nothing has settled yet."
                  next="python -m scripts.live"
                />
              }
            />
          )}
        </Panel>
      </section>

      {/* Kept beside the venue table: only our own rows hold what the model
          claimed at entry. */}
      {onVenue && settled.length > 0 && (
        <section>
          <SectionHead
            eyebrow={`${settled.length} recorded`}
            title="What we thought at entry"
            note="Our own rows, for the forecast provenance the venue's ledger does not carry: the model's probability, the edge it claimed, and the fee we predicted. The money numbers above are the venue's."
          />
          <Panel flush>
            <DataTable
              columns={settledColumns}
              rows={settled}
              keyOf={(p) => String(p.id)}
              empty={<Empty what="Nothing has settled yet." />}
            />
          </Panel>
        </section>
      )}
    </div>
  );
}

/** The venue's settlement rows. Every column is a number Kalshi served, except
 *  `vs ours`, the gap against our books. */
const venueColumns: Column<VenueSettlement>[] = [
  {
    key: 'settled',
    head: 'settled',
    render: (r) => <span className="font-mono">{stamp(r.settled_time)}</span>,
  },
  {
    key: 'ticker',
    head: 'market',
    render: (r) => <span className="font-mono text-tiny">{r.ticker}</span>,
  },
  {
    key: 'side',
    head: 'side',
    render: (r) =>
      r.yes_contracts > 0 && r.no_contracts > 0 ? (
        <span className="text-ink-3">both</span>
      ) : r.yes_contracts > 0 ? (
        <Chip tone="above">up</Chip>
      ) : r.no_contracts > 0 ? (
        <Chip tone="below">down</Chip>
      ) : (
        <span className="text-ink-3">—</span>
      ),
  },
  { key: 'qty', head: 'qty', numeric: true, render: (r) => r.contracts.toFixed(0) },
  {
    key: 'cost',
    head: 'cost',
    numeric: true,
    render: (r) => (r.cost == null ? '—' : `$${r.cost.toFixed(2)}`),
  },
  {
    key: 'revenue',
    head: 'revenue',
    numeric: true,
    render: (r) => (r.revenue == null ? '—' : `$${r.revenue.toFixed(2)}`),
  },
  {
    key: 'fee',
    head: 'fee',
    numeric: true,
    render: (r) => (r.fee_cost == null ? '—' : `$${r.fee_cost.toFixed(2)}`),
  },
  {
    key: 'result',
    head: 'result',
    render: (r) =>
      r.won == null ? (
        <span className="text-ink-3" title="the venue named no result for this market">
          —
        </span>
      ) : (
        <Chip tone={r.won ? 'pass' : 'fail'}>{r.won ? 'won' : 'lost'}</Chip>
      ),
  },
  {
    key: 'pnl',
    head: 'p&l',
    numeric: true,
    render: (r) =>
      r.pnl == null ? (
        <span className="text-ink-3" title="the venue left a field absent — this is not zero">
          —
        </span>
      ) : (
        <span className={r.pnl >= 0 ? 'text-above' : 'text-below'}>
          {r.pnl >= 0 ? '+' : ''}
          {r.pnl.toFixed(2)}
        </span>
      ),
  },
  {
    key: 'gap',
    head: 'vs ours',
    numeric: true,
    render: (r) =>
      r.pnl_gap == null ? (
        <span
          className="text-ink-3"
          title={
            r.our_pnl == null
              ? 'we have no record of buying this market'
              : 'the venue left a field absent'
          }
        >
          —
        </span>
      ) : (
        <span className={Math.abs(r.pnl_gap) > 0.01 ? 'text-warn' : 'text-ink-3'}>
          {r.pnl_gap >= 0 ? '+' : ''}
          {r.pnl_gap.toFixed(2)}
        </span>
      ),
  },
];

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
