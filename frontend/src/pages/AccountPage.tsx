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
  fetchVenueDaily,
  fetchVenueEquity,
  fetchVenueSettlements,
} from '../api/serving';
import { EquityChart, VenueAccountChart } from '../components/Charts';
import {
  Chip,
  Failed,
  Loading,
  Metric,
  Panel,
  PageLink,
  SectionHead,
} from '../components/Primitives';
import { DailyPnlCalendar } from '../components/DailyPnlCalendar';
import { go } from '../lib/navigate';
import { cents } from '../lib/format';

export function AccountPage() {
  // **Everything on this page polls at one second.**
  //
  // The cadences here used to be tuned per endpoint, from 10s to 120s, and the
  // result was a page whose panels disagreed with each other: the account value
  // had moved, the day's total had not, and the two were describing the book at
  // different instants. A single interval means every number on screen is read
  // from the same second.
  //
  // It is affordable because these are small reads against a local backend --
  // measured 2026-10-20: /account 4ms, the day totals 46ms and 737 bytes.
  // `Cache-Control: no-store` means each one is a real query rather than a
  // revalidation, and `usePolling` stops the timer entirely when the tab is
  // hidden, so an unwatched dashboard costs nothing.
  //
  // The one to watch is the venue curve: 475 KB a response, so a second is
  // ~475 KB/s on the wire for a 30-day series that gains one step per settled
  // market. It is here for consistency rather than need.
  const LIVE_MS = 1_000;

  const account = usePolling(fetchAccount, LIVE_MS);
  const venue = usePolling(fetchVenueAccount, LIVE_MS);
  const venueCurve = usePolling(() => fetchVenueEquity(30), LIVE_MS);
  const equity = usePolling(() => fetchEquity(90), LIVE_MS);
  const settlements = usePolling(() => fetchVenueSettlements(200), LIVE_MS);
  const positions = usePolling(() => fetchPositions(false, 200), LIVE_MS);
  const daily = usePolling(() => fetchVenueDaily(180), LIVE_MS);
  const openPositions = usePolling(() => fetchPositions(true, 50), LIVE_MS);

  if (account.loading) return <Loading what="the account" />;
  if (account.error) return <Failed error={account.error} what="the account" />;
  const state = account.data;
  if (!state) return null;

  const ledger = venue.data;
  // Seconds since the venue was last read. `lastUpdated` is when the RESPONSE
  // landed; `balance_at` is when the sample inside it was taken, and it is the
  // second that matters — a fresh response carrying an hour-old sample is
  // exactly the failure being watched for.
  const staleSeconds = venue.data?.balance_at
    ? (Date.now() - new Date(venue.data.balance_at).getTime()) / 1000
    : null;
  // The venue has a ledger only once real orders have filled. Until then this
  // page shows our arithmetic and says so, rather than drawing a $0.00 P&L that
  // looks like a measurement.
  const onVenue = Boolean(ledger?.available);

  const settled = (positions.data?.positions ?? []).filter((p) => p.outcome !== 'pending');
  // Newest first, so the entry that just filled is at the top rather than
  // buried under whatever has been running longest.
  const openRows = [...(openPositions.data?.positions ?? [])].sort((a, b) =>
    b.window_open.localeCompare(a.window_open) || a.offset_minutes - b.offset_minutes,
  );
  const settledContracts = settled.reduce((s, p) => s + p.contracts, 0);

  // Per TRADE, and displayed as such. The venue resolves markets, not
  // contracts, so this is the natural unit for "how often were we right" —
  // it is only wrong when subtracted from a per-contract cost, which is what
  // `realisedEdge` used to do.
  const winRate = onVenue
    ? ledger!.win_rate.value
    : settled.length
      ? settled.filter((p) => p.outcome === 'won').length / settled.length
      : null;

  // **P&L per contract, not `win_rate - mean_cost`.** That subtraction mixed
  // denominators: the win rate is per TRADE (the venue resolves markets, not
  // contracts) while the mean cost is per CONTRACT. They only agree if every
  // trade is the same size, and they are not — measured 2026-10-03, the win
  // rate was 0.5714 per trade against 0.5853 per contract, because winners
  // carry more contracts than losers. The gap read as 1.4pp of edge that was
  // never lost: -2.71pp displayed against a true -1.32pp.
  //
  // Dividing realised P&L by contracts has no such ambiguity, and it is the
  // same quantity `realised_edge_pp` gates on, so the dashboard and the
  // promotion gate finally mean one thing.
  const realisedEdge = onVenue
    ? ledger!.contracts > 0 && ledger!.realized_pnl.value != null
      ? ledger!.realized_pnl.value / ledger!.contracts
      : null
    : settledContracts > 0
      ? settled.reduce((s, p) => s + (p.pnl ?? 0), 0) / settledContracts
      : null;

  // Contract-weighted too, or the comparison against realised is
  // apples-to-oranges: a per-trade mean over-weights the small trades the
  // whole-contract floor barely admitted.
  const predictedEdge = settledContracts
    ? settled.reduce((s, p) => s + p.edge * p.contracts, 0) / settledContracts
    : null;

  const venueRows = settlements.data?.settlements ?? [];
  const beforeWindow = venueCurve.data?.pnl_before_window ?? null;

  return (
    <div className="space-y-8">
      {/* **No eyebrow.** It read "live · real money · started at $462.10",
          and both halves had to go. The mode is already on screen as a `live`
          chip two lines below — the invariant is that `account.mode` reaches
          every surface showing one of its numbers, not that it appears twice.
          And `starting_bankroll` is a CONFIG value (`--bankroll`), not the
          balance this dashboard epoch actually began at, so presenting it as
          where the account "started" was simply wrong. */}
      <SectionHead
        title="Account"
        note={
          onVenue
            ? 'Every figure here is the venue’s — cash and the mark on what is open, both read from Kalshi, so fees and settlement are already inside them. Nothing on this page is our own arithmetic about money.'
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

      {state.open_positions > 0 && (
        <Panel>
          {/* **Marked to the venue's bid, not to our forecast.** `at risk`
              below is cost and stays cost; this strip is what the book says
              the open contracts would fetch right now, which is safe to show
              moving because it is not an opinion. Only rendered while
              something is open — an empty live strip reads as a broken feed. */}
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3">
            {/* **Every shard, not just the trading one.** This read $494.27
                against a real $495.37 because it was built on
                `account.bankroll`, which holds the crypto shard alone. Sizing
                still uses that — it is the only cash an order can reach — but
                the figure a person reads as "my account" has to be the whole
                account. The stranded part is named below rather than dropped. */}
            <Metric
              label="account value"
              value={state.account_value}
              unit="$"
              size="lg"
              hint="every shard's cash plus open contracts at the venue’s bid — what someone would pay for the position right now, not what we paid"
            />
            <Metric
              label="open p&l"
              value={state.unrealized_pnl}
              unit="$"
              size="lg"
              tone={(state.unrealized_pnl.value ?? 0) >= 0 ? 'above' : 'below'}
              hint="the venue's own mark, minus what the fills cost and the fee it charged to open them. Unrealised: it moves every second and is worth nothing until the market settles."
            />
            <Metric
              label="open"
              value={state.open_positions}
              digits={0}
              hint={`${state.open_contracts} contract(s) across ${state.open_positions} position(s)`}
            />
            {/* **Only the warning survives here.** The quote timestamp and
                the stranded-cash figure were steady-state facts dressed as
                news: the mark is rewritten every second, so its time is always
                now, and cash sits off the trading shard permanently without
                anything being wrong. A position still carried at COST is
                different — it means no book was readable for it, so its P&L
                above is a placeholder rather than a measurement. */}
            {state.unmarked_positions > 0 && (
              <div className="text-tiny text-warn">
                {state.unmarked_positions} still at cost — no book for{' '}
                {state.unmarked_positions === 1 ? 'it' : 'them'} yet
              </div>
            )}
          </div>

          {/* One row per open entry.
              **Four columns, not six.** `symbol | side | ct | bought | now |
              p&l` overflowed 375px and the horizontal scroll clipped P&L — the
              one column worth looking at. Side is folded into the symbol as a
              colour and an arrow, and size is folded into the entry price, so
              nothing is lost and nothing is cut off.

              The offset is kept because multi-entry puts several rows on the
              same symbol and window; without it two rows look like a duplicate
              rather than two bets. */}
          {openRows.length > 0 && (
            <div className="mt-4 border-t border-rule pt-3">
              <table className="w-full text-tiny">
                <thead>
                  <tr className="text-micro uppercase tracking-wider text-ink-3">
                    <th className="pb-1 text-left font-normal">position</th>
                    <th className="pb-1 text-right font-normal">bought</th>
                    <th className="pb-1 text-right font-normal">now</th>
                    <th className="pb-1 text-right font-normal">p&amp;l</th>
                  </tr>
                </thead>
                <tbody>
                  {openRows.map((p) => {
                    const live = p.unrealized_pnl;
                    const up = p.side === 'up';
                    return (
                      <tr key={p.id} className="border-t border-rule/40">
                        <td className="py-1.5 whitespace-nowrap">
                          <span className={up ? 'text-above' : 'text-below'}>
                            {up ? '▲' : '▼'}
                          </span>{' '}
                          <span className="font-medium text-ink">
                            {p.symbol.replace('-USD', '')}
                          </span>
                          <span className="ml-1 text-micro text-ink-3">
                            +{p.offset_minutes}m
                          </span>
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-ink-2">
                          {p.contracts}&thinsp;@&thinsp;{cents(p.price)}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-ink">
                          {p.mark_price == null ? (
                            <span className="text-ink-3">—</span>
                          ) : (
                            cents(p.mark_price)
                          )}
                        </td>
                        <td
                          className={[
                            'py-1.5 text-right font-semibold tabular-nums',
                            live == null
                              ? 'text-ink-3'
                              : live >= 0
                                ? 'text-above'
                                : 'text-below',
                          ].join(' ')}
                        >
                          {live == null
                            ? '—'
                            : `${live >= 0 ? '+' : ''}${live.toFixed(2)}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}

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
                  Trading profit only, stepped once per market that resolved.
                  It is not the balance: the balance also moves when you deposit
                  or withdraw, and that is not the strategy. Over any stretch
                  with nothing open at either end the two agree.{' '}
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
            {/* **The same quantity the calendar below sums to, and read from
                the same response.** It used to be realised P&L on settled
                markets while the calendar showed the change in account value,
                so the headline and the squares under it disagreed by whatever
                was open and by the midnight boundary — which simply reads as
                one of them being broken. One number, one definition. */}
            <Metric
              label={onVenue ? 'p&l · since reset' : 'realised p&l'}
              value={
                onVenue
                  ? (daily.data?.total ?? null)
                  : state.realized_pnl
              }
              unit="$"
              size="lg"
              tone={
                ((onVenue ? daily.data?.total : state.realized_pnl.value) ?? 0) >= 0
                  ? 'above'
                  : 'below'
              }
              hint={
                onVenue
                  ? 'the change in account value — cash plus what is open, both from Kalshi — summed over the days below. Fees and settlement are already inside it.'
                  : 'our own arithmetic — the venue has not been read'
              }
            />
            {/* **The total, because that is the number the venue's own app
                shows.** This used to render the trading shard alone, so the
                page read $482.46 against an app saying $483.57 while
                `balance_drift` sat at exactly 0.00 — two correct figures and
                no way to tell that from a sync fault. The shard is the one an
                order can actually spend and is carried directly beneath. */}
            <Metric
              label={onVenue ? 'cash · venue' : 'cash'}
              value={onVenue ? ledger!.total_balance : state.bankroll}
              unit="$"
              size="lg"
              hint={
                onVenue
                  ? 'every shard, from /portfolio/balance — the figure the Kalshi app shows. Cash, not equity: an open position has already left it.'
                  : undefined
              }
            />
            {onVenue && (
              <Metric
                label={
                  ledger!.exchange_index == null
                    ? 'tradeable'
                    : `tradeable · shard ${ledger!.exchange_index}`
                }
                value={ledger!.balance}
                unit="$"
                size="lg"
                tone={
                  (ledger!.stranded.value ?? 0) > 0.01 ? 'muted' : undefined
                }
                hint={
                  (ledger!.stranded.value ?? 0) > 0.01
                    ? `what a crypto order can draw on. $${ledger!.stranded.value!.toFixed(
                        2,
                      )} sits on other shards and cannot reach these markets — balances are local to a shard, which is what made every order come back insufficient_balance against an apparently healthy total.`
                    : 'what a crypto order can draw on. Balances are local to a shard; nothing is stranded elsewhere.'
                }
              />
            )}
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
              hint="settled P&L per contract, fees included. The gap against predicted is the winner's curse."
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

      <section>
        <SectionHead
          eyebrow="as the Kalshi app shows it"
          title="Day by day"
          note="Each cell is how much the Kalshi balance moved on that Eastern calendar day — the same number, on the same day boundary, as the app. It will not tie to the chart above: that one counts a trade on the day it resolves, this one on the day the cash moved, and a window bought before midnight pays out after it."
        />
        <Panel>
          {daily.loading ? (
            <Loading what="the daily record" />
          ) : daily.error ? (
            <Failed error={daily.error} what="the daily record" />
          ) : (
            <DailyPnlCalendar
              days={daily.data?.days ?? []}
              timezone={daily.data?.timezone}
              epoch={daily.data?.epoch}
            />
          )}
        </Panel>
      </section>

      {/* **No reconciliation section, and no completeness section either.**

          The first compared `p&l · venue` against `p&l · ours` — two figures
          now both SET from the venue's own `realized_pnl`, so it was Kalshi
          against Kalshi and the gap read -2e-14.

          The second asked whether every settled market had a position of ours
          behind it. That is a real question, but not one for this page: no
          money figure here reads our `positions` table any more. A trade with
          no decision record of ours is a hole in the RESEARCH sample — it has
          no offset, no forecast, no edge — and it costs nothing. The
          reconciler reports it, which is where it belongs.

          What can still go wrong is the mirror going stale. Everything here
          is Kalshi's, read once a second by `scripts/mark_positions.py`; if
          that stops, the page freezes on numbers that look perfectly healthy.
          So the only thing left is a staleness warning, and it shows up only
          when there is something to say. */}
      {onVenue && staleSeconds != null && staleSeconds > 90 && (
        <p className="rounded border border-warn/40 bg-warn/10 px-3 py-2 text-tiny text-warn">
          The venue was last read {Math.round(staleSeconds)}s ago. Every figure
          on this page is that old — the per-second sampler has stopped, so
          nothing here is moving even if the market is.
        </p>
      )}

      {/* **Linked, not duplicated.** Both of these tables were full copies of
          pages that already exist: the venue's settled markets and our own
          entry rows are both in Decisions, which has the filters and the room
          for them. Rendering them twice meant two places to keep correct and a
          page you had to scroll past to reach anything. */}
      <section>
        <SectionHead
          eyebrow="the detail"
          title="Every trade, one by one"
          note="This page is the totals. The row-level record — what was bought, what it cost and what the venue paid — is the Decisions page with its traded filter on."
        />
        {/* **One link, and it lands filtered.** Decisions already holds this
            table; the account does not need a second copy of it, only a way
            in. Calibration is not here because it is a question about the
            MODEL — whether a forecast was honest about its own confidence —
            and it has its own tab. */}
        <PageLink
          href="/decisions?traded=1"
          title="Trades"
          note="every position taken, with the forecast and the edge that justified it"
          count={onVenue ? `${venueRows.length.toLocaleString()} settled` : undefined}
          onNavigate={go}
        />
      </section>
    </div>
  );
}
