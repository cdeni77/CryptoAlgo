/** The barrier, now. One card per symbol, the account beside it.
 *
 * The layout answers the question the system actually asks, in the order it asks
 * it: how far has price moved from the strike, how much volatility is left, what
 * probability does that imply, what does the market charge, and is the gap
 * between those two big enough to pay for. The quarter-hour track and the
 * probability scale carry the first and the last of those; the numbers in
 * between are supporting detail.
 */
import { usePolling } from '../hooks/usePolling';
import { fetchFunnel, fetchLive, fetchPrices } from '../api/serving';
import type { OrderTicket, Prediction } from '../types';
import { FunnelChart } from '../components/Charts';
import { ProbabilityScale } from '../components/ProbabilityScale';
import { QuarterTrack } from '../components/QuarterTrack';
import { WindowChart } from '../components/WindowChart';
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
  SideChip,
} from '../components/Primitives';
import { cents, clock, pct, signedPp } from '../lib/format';

const WINDOW_MINUTES = 15;
const OFFSETS = [3, 6, 9, 12];

export function LivePage() {
  const live = usePolling(fetchLive, 15_000);
  const funnel = usePolling(() => fetchFunnel(7), 60_000);

  if (live.loading) return <Loading what="the current window" />;
  if (live.error) return <Failed error={live.error} what="the current window" />;
  const state = live.data;
  if (!state) return null;

  const account = state.account;

  return (
    <div className="space-y-8">
      {account.mode === 'live' && (
        <div className="flex items-center gap-3 border-l-2 border-below bg-below-wash px-4 py-2">
          <Chip tone="below">live</Chip>
          <span className="text-tiny text-ink">
            This account holds real money. Positions below are actual exposure,
            not a simulation.
          </span>
        </div>
      )}

      <section>
        <SectionHead
          eyebrow={`${state.windows.length} symbols · ${WINDOW_MINUTES}-minute windows`}
          title="Barrier state"
          note="Displacement from the strike is known exactly. Only the volatility over the minutes that remain has to be forecast — that is the entire estimand, and it is the one quantity this project ever measured as forecastable."
        />
        {state.windows.length === 0 ? (
          <Empty
            what="No decision has been recorded yet."
            next="python -m scripts.live"
          />
        ) : (
          <div className="grid gap-4 lg:grid-cols-3">
            {state.windows.map((w) => (
              <WindowCard key={w.symbol} window={w} />
            ))}
          </div>
        )}
      </section>

      <section className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <Panel>
          <SectionHead
            eyebrow={
              account.halted
                ? 'halted'
                : account.mode === 'live'
                  ? 'live account · real money'
                  : 'paper account'
            }
            title="Account"
            note="Open positions are carried at cost, never marked to our own forecast."
          />
          <div className="grid grid-cols-2 gap-x-4 gap-y-4">
            <Metric label="equity" value={account.equity} unit="$" digits={2} size="lg" />
            <Metric label="cash" value={account.bankroll} unit="$" digits={2} size="lg" />
            <Metric label="at risk" value={account.staked} unit="$" digits={2} />
            <Metric
              label="realised p&l"
              value={account.realized_pnl}
              unit="$"
              digits={2}
              tone={
                (account.realized_pnl.value ?? 0) >= 0 ? 'above' : 'below'
              }
            />
            <Metric label="fees paid" value={account.fees_paid} unit="$" digits={2} tone="muted" />
            <Metric label="open" value={account.open_positions} digits={0} />
          </div>
          {account.halted && (
            <p className="mt-4 border-l-2 border-fail bg-below-wash px-3 py-2 text-tiny text-ink">
              Trading halted: {account.halted_reason ?? 'bankroll floor breached'}
            </p>
          )}
        </Panel>

        <Panel>
          <SectionHead
            eyebrow={`${funnel.data?.days ?? 7} days`}
            title="Why it declined"
            note="Abstention is the default action. This funnel dominated by edge-below-gate is the system working: the forecast does not cover the fee, so it declines."
          />
          {funnel.loading ? (
            <Loading what="the funnel" />
          ) : funnel.error ? (
            <Failed error={funnel.error} what="the funnel" />
          ) : (
            <FunnelChart stages={funnel.data?.stages ?? []} />
          )}
        </Panel>
      </section>

      {state.tickets.length > 0 && (
        <section>
          <SectionHead
            eyebrow={`${state.tickets.length} awaiting`}
            title="Order tickets"
            note="Written for every live decision. A ticket that was never placed and a fill are different things, and the status column is the difference."
          />
          <Panel flush>
            <DataTable
              columns={ticketColumns}
              rows={state.tickets}
              keyOf={(t) => String(t.id)}
            />
          </Panel>
        </section>
      )}

      <section>
        <SectionHead
          eyebrow={`${state.open_positions.length} open`}
          title="Open positions"
          note="Held to settlement. Settlement is free; an exit pays a second fee and crosses the spread again, and a binary's loss is already capped at the stake."
        />
        <Panel flush>
          <DataTable
            columns={openColumns}
            rows={state.open_positions}
            keyOf={(p) => String(p.id)}
            empty={<Empty what="Nothing open. The next window opens on the quarter hour." />}
          />
        </Panel>
      </section>
    </div>
  );
}

/** The last two hours of price for one symbol, with each window's strike.
 *
 * Polled separately from the decision state and at a slower cadence: the path is
 * two hours of history that changes one minute at a time, while the decision
 * changes every cycle. One request for both would refetch 120 bars to learn a
 * probability moved.
 */
function SymbolChart({ symbol }: { symbol: string }) {
  const series = usePolling(() => fetchPrices(symbol, 120), 60_000, [symbol]);
  if (series.loading) return <Loading what="prices" />;
  if (series.error) return <Failed error={series.error} what={`${symbol} prices`} />;
  return (
    <WindowChart
      bars={series.data?.bars ?? []}
      strikes={series.data?.strikes ?? []}
      height={140}
    />
  );
}

function WindowCard({ window: w }: { window: Prediction }) {
  const secondsToSettle = Math.max(
    0,
    (new Date(w.settle_time).getTime() - Date.now()) / 1000,
  );
  const displacementBps = w.displacement * 10_000;
  const above = w.model_probability >= 0.5;

  return (
    <Panel>
      <div className="flex items-baseline justify-between">
        <h3 className="font-mono text-mid font-medium text-ink">{w.symbol}</h3>
        {w.traded ? (
          <SideChip side={w.side} />
        ) : (
          <Chip tone="neutral" title={`gate: ${w.reason}`}>
            {w.reason.replace(/_/g, ' ')}
          </Chip>
        )}
      </div>

      <div className="mt-3">
        <QuarterTrack
          windowMinutes={WINDOW_MINUTES}
          elapsed={w.offset_minutes}
          offsets={OFFSETS}
          secondsToSettle={secondsToSettle}
        />
      </div>

      <div className="mt-4 border-t border-rule pt-3">
        <div className="eyebrow mb-1">price against strike</div>
        <SymbolChart symbol={w.symbol} />
      </div>

      <div className="mt-4">
        {/* market_probability is the real quote, null when none was read.
            price is what WE paid and is null on every refused window — using
            it as a market fallback drew our own baseline and labelled it
            "market" on the majority of rows. */}
        <ProbabilityScale
          probability={w.model_probability}
          price={w.market_probability}
          breakEven={w.effective_cost}
        />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-rule pt-3">
        <Metric
          label="displacement"
          value={displacementBps}
          unit="bp"
          digits={1}
          tone={displacementBps >= 0 ? 'above' : 'below'}
          hint="last price against the strike the window opened on"
        />
        <Metric
          label="vol remaining"
          value={w.sigma_remaining == null ? null : w.sigma_remaining * 10_000}
          unit="bp"
          digits={1}
          hint="sigma over the minutes still to come — the only quantity being forecast. Labelled in Latin because the eyebrow style uppercases, and CSS turns a lowercase sigma into a capital one."
        />
        <Metric
          label="forecast"
          value={w.model_probability * 100}
          unit="%"
          digits={1}
          tone={above ? 'above' : 'below'}
          hint="probability the window settles above its strike"
        />
        <Metric
          label="baseline"
          value={w.baseline_probability * 100}
          unit="%"
          digits={1}
          tone="muted"
          hint="F(x/σ) — what a clock and a volatility estimate alone imply. This is the benchmark, not 50%."
        />
        <div>
          <div className="eyebrow" title="model probability minus quote, half-spread and fee">
            edge
          </div>
          <div
            className={`mt-0.5 font-mono text-mid font-medium ${
              (w.edge ?? 0) > 0 ? 'text-above' : 'text-ink-3'
            }`}
          >
            {signedPp(w.edge)}
          </div>
        </div>
        <div>
          <div
            className="eyebrow"
            title={
              w.price_source === 'quote'
                ? "the venue's own ask priced this decision"
                : 'no book was read; the calibrated barrier stood in for the market'
            }
          >
            priced by
          </div>
          <div className="mt-0.5 font-mono text-mid font-medium text-ink-2">
            {w.price_source === 'quote' ? 'quote' : 'baseline'}
          </div>
        </div>
      </dl>
    </Panel>
  );
}

const ticketColumns: Column<OrderTicket>[] = [
  { key: 'symbol', head: 'symbol', render: (t) => <span className="font-mono">{t.symbol}</span> },
  {
    key: 'market',
    head: 'market',
    render: (t) => (
      <span className="font-mono text-micro text-ink-3">{t.market_ticker ?? '—'}</span>
    ),
  },
  { key: 'side', head: 'side', render: (t) => <SideChip side={t.side} /> },
  { key: 'contracts', head: 'qty', numeric: true, render: (t) => t.contracts },
  { key: 'limit', head: 'limit', numeric: true, render: (t) => cents(t.limit_price) },
  { key: 'max', head: 'max', numeric: true, render: (t) => cents(t.max_price) },
  { key: 'cost', head: 'cost', numeric: true, render: (t) => `$${t.expected_cost.toFixed(2)}` },
  { key: 'edge', head: 'edge', numeric: true, render: (t) => signedPp(t.edge) },
  {
    key: 'status',
    head: 'status',
    render: (t) => (
      <Chip
        tone={
          t.status === 'filled' ? 'pass'
            : t.status === 'placed' ? 'accent'
            : t.status === 'skipped' ? 'fail'
            : 'warn'
        }
      >
        {t.status}
      </Chip>
    ),
  },
];

const openColumns: Column<import('../types').Position>[] = [
  { key: 'symbol', head: 'symbol', render: (p) => <span className="font-mono">{p.symbol}</span> },
  { key: 'side', head: 'side', render: (p) => <SideChip side={p.side} /> },
  { key: 'settle', head: 'settles', render: (p) => <span className="font-mono">{clock(p.settle_time)}</span> },
  { key: 'offset', head: 'entered', numeric: true, render: (p) => `+${p.offset_minutes}m` },
  { key: 'contracts', head: 'contracts', numeric: true, render: (p) => p.contracts },
  { key: 'price', head: 'price', numeric: true, render: (p) => cents(p.price) },
  { key: 'outlay', head: 'at risk', numeric: true, render: (p) => `$${p.outlay.toFixed(2)}` },
  { key: 'q', head: 'forecast', numeric: true, render: (p) => pct(p.model_probability, 1) },
  { key: 'edge', head: 'edge', numeric: true, render: (p) => signedPp(p.edge) },
];
