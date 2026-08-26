/** The account, as the venue's ledger reports it.
 *
 * **This page used to report our own arithmetic and call it P&L.** The bankroll
 * was debited at the price `decide()` sized at, the fee was predicted from the
 * published schedule, and the payout was decided by an OHLC mean of Coinbase
 * standing in for sixty seconds of CF Benchmarks BRTI. Live, every one of those
 * three is an estimate of a number Kalshi already holds, and the audit found the
 * paper bankroll had never once been credited a win.
 *
 * So the headline numbers here come from `/portfolio/settlements` and
 * `/portfolio/balance` — what the account of record shows. Our own figures are
 * kept beside them rather than thrown away, because the gap is a measurement in
 * its own right: a mispriced fee, a settlement our proxy called differently, or a
 * fill nobody booked. The reconciliation panel is that gap, and a drift that
 * grows is the alarm.
 *
 * Two things are deliberately not here. Open positions are not marked to
 * anything — not to our forecast, which would book conviction as profit, and not
 * to the tape either, because a 15-minute binary held to expiry has one price
 * that matters and it is the settlement. And the public trade tape
 * (`/historical/trades`) contributes nothing: it carries every print in a market
 * by anyone, with no account attribution, so summing it sums the exchange rather
 * than the portfolio.
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
  // Faster than the internal figures, because this is the series that moves: the
  // live loop samples the venue's balance every cycle.
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
  // page shows our arithmetic and says so, rather than drawing a $0.00 P&L that
  // looks like a measurement.
  const onVenue = Boolean(ledger?.available);

  const settled = (positions.data?.positions ?? []).filter((p) => p.outcome !== 'pending');
  const paid = settled.reduce((sum, p) => sum + p.outlay, 0);
  const meanCost = settled.length ? paid / settled.reduce((s, p) => s + p.contracts, 0) : null;
  // The win rate comes from the venue where there is one — it resolved the
  // markets, we only estimated them from Coinbase bars.
  const winRate = onVenue
    ? ledger!.win_rate.value
    : settled.length
      ? settled.filter((p) => p.outcome === 'won').length / settled.length
      : null;
  const realisedEdge = winRate != null && meanCost != null ? winRate - meanCost : null;
  const predictedEdge = settled.length
    ? settled.reduce((s, p) => s + p.edge, 0) / settled.length
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
                <VenueAccountChart
                  points={venueCurve.data?.points ?? []}
                  balances={venueCurve.data?.balances ?? []}
                />
                <p className="mt-2 text-tiny text-ink-3">
                  Realised P&amp;L, stepped once per settled market, against the venue’s
                  cash on the right axis.{' '}
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

      {/* Kept alongside the venue table rather than replaced by it. The venue
          knows what a position paid; only our own row knows what the model
          claimed at entry, and the forecast against the outcome is the whole
          research question. */}
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

/** The venue's own settlement rows. Money only — every column is a number Kalshi
 *  served, and `ours` is the one comparison against our books. */
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
