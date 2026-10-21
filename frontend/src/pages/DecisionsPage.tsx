/** Every decision point, traded or refused.
 *
 * The refusals are the point. A screen showing only trades cannot show that the
 * system declined because the forecast did not cover the fee, which is the most
 * informative thing it has to say — and on the predecessor of this system,
 * `edge_below_cost rejecting 82% of candidates` said what no Sharpe ratio said.
 */
import { useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchPredictions } from '../api/serving';
import { ProbabilityScale } from '../components/ProbabilityScale';
import {
  Chip,
  Column,
  DataTable,
  Empty,
  Failed,
  Loading,
  Panel,
  SectionHead,
  SideChip,
} from '../components/Primitives';
import { cents, clock, pct, signedPp, stamp } from '../lib/format';
import type { Prediction } from '../types';

/** Did we arrive here asking for trades only?
 *
 *  The Account page links in as "Trades", which is this page with the filter
 *  already on -- it is the same table and does not deserve a second one. The
 *  state lives in the URL rather than in a prop because `PAGES` is a
 *  `Record<RoutePath, ComponentType>`, and because it then survives a reload
 *  and the back button.
 */
function tradedFromUrl(): boolean {
  try {
    return new URLSearchParams(window.location.search).get('traded') === '1';
  } catch {
    return false;
  }
}

export function DecisionsPage() {
  const [tradedOnly, setTradedOnly] = useState(tradedFromUrl);

  // **Kept in step with the toggle.** Without this the URL still said
  // `?traded=1` after switching to ALL, so a reload silently put the filter
  // back on and the page disagreed with its own address bar. `replaceState`
  // rather than `pushState`: flipping a filter is not a navigation, and
  // making it one would turn the back button into an undo for it.
  const show = (next: boolean) => {
    setTradedOnly(next);
    try {
      const url = new URL(window.location.href);
      if (next) url.searchParams.set('traded', '1');
      else url.searchParams.delete('traded');
      window.history.replaceState({}, '', url);
    } catch {
      /* A browser that will not parse its own URL still gets the filter. */
    }
  };
  const state = usePolling(() => fetchPredictions(250, tradedOnly), 30_000, [tradedOnly]);

  const rows = state.data?.predictions ?? [];
  const traded = rows.filter((r) => r.traded).length;
  // **A page, not a total.** `rows` is at most `limit` and is bounded by the
  // dashboard epoch, so "250 decisions" read as the whole population of a
  // table holding 250 of 60,873. And with the TRADED filter on every fetched
  // row is traded, so it printed "250 decisions · 250 traded" — a 100% trade
  // rate on a page whose whole thesis is that abstention is the default.
  const eyebrow = tradedOnly
    ? `${rows.length} traded · newest first`
    : `newest ${rows.length} · ${traded} traded`;

  return (
    <div className="space-y-6">
      <SectionHead
        eyebrow={eyebrow}
        title="Decisions"
        note="Abstention is the default action and every refusal is named. The scale shows the forecast as a bar from the 50% pivot, the market as a caret, and break-even as a hairline — a trade exists only when the bar clears the hairline."
        right={
          <div className="flex border border-rule">
            {[
              { label: 'all', value: false },
              { label: 'traded', value: true },
            ].map((option) => (
              <button
                key={option.label}
                type="button"
                onClick={() => show(option.value)}
                className={[
                  'px-3 py-1 font-mono text-micro uppercase',
                  tradedOnly === option.value
                    ? 'bg-ink text-paper'
                    : 'text-ink-2 hover:bg-sunken',
                ].join(' ')}
              >
                {option.label}
              </button>
            ))}
          </div>
        }
      />

      <Panel flush>
        {state.loading ? (
          <Loading what="decisions" />
        ) : state.error ? (
          <Failed error={state.error} what="decisions" />
        ) : (
          <DataTable
            columns={columns}
            rows={rows}
            keyOf={(p) => `${p.symbol}-${p.window_open}-${p.offset_minutes}`}
            empty={<Empty what="No decisions recorded." next="python -m scripts.live" />}
          />
        )}
      </Panel>
    </div>
  );
}

const columns: Column<Prediction>[] = [
  // **Time only on a phone, date too from `sm` up.** Every row on screen is
  // from the same day in practice, so the date cost a line-wrap per row and
  // told the reader nothing.
  { key: 'when', head: 'window', render: (p) => (
      <span className="whitespace-nowrap font-mono">
        <span className="hidden sm:inline">{stamp(p.window_open)}</span>
        <span className="sm:hidden">{clock(p.window_open)}</span>
      </span>
    ) },
  // `BTC-USD` wrapped to two lines in a 60px column; the `-USD` is the same
  // for all three symbols and carries nothing.
  { key: 'symbol', head: 'symbol', render: (p) => (
      <span className="whitespace-nowrap font-mono">
        <span className="hidden sm:inline">{p.symbol}</span>
        <span className="sm:hidden">{p.symbol.replace('-USD', '')}</span>
      </span>
    ) },
  { key: 'offset', head: 'at', numeric: true, render: (p) => `+${p.offset_minutes}m` },
  {
    key: 'displacement',
    head: 'disp.',
    secondary: true,
    numeric: true,
    render: (p) => (
      <span className={p.displacement >= 0 ? 'text-above' : 'text-below'}>
        {(p.displacement * 10_000).toFixed(1)}bp
      </span>
    ),
  },
  {
    key: 'sigma',
    head: 'vol rem.',
    secondary: true,
    numeric: true,
    render: (p) => (p.sigma_remaining == null ? '—' : `${(p.sigma_remaining * 10_000).toFixed(1)}bp`),
  },
  {
    key: 'scale',
    head: 'forecast vs market',
    secondary: true,
    width: '13rem',
    render: (p) => (
      <div className="py-0.5">
        {/* **Side-relative, not P(up).** `probability` below is already the
            probability of the side TAKEN, so a raw P(up) market put the bar
            and the caret on OPPOSITE axes for every down row — they summed to
            1.0, and a +0.51pp edge drew as a 59-point gap with the caret on
            the wrong side of break-even. A third of the table is down rows. */}
        <ProbabilityScale
          probability={p.model_probability}
          price={p.market_probability_side}
          breakEven={p.effective_cost}
          height={12}
          showAxis={false}
        />
      </div>
    ),
  },
  { key: 'q', head: 'model', numeric: true, secondary: true, render: (p) => pct(p.model_probability, 1) },
  { key: 'base', head: 'baseline', numeric: true, secondary: true, render: (p) => <span className="text-ink-3">{pct(p.baseline_probability, 1)}</span> },
  { key: 'price', head: 'price', numeric: true, secondary: true, render: (p) => cents(p.price) },
  { key: 'edge', head: 'edge', numeric: true, render: (p) => (
    <span className={(p.edge ?? 0) > 0 ? 'text-above' : 'text-ink-3'}>{signedPp(p.edge)}</span>
  ) },
  {
    key: 'outcome',
    head: 'action',
    render: (p) =>
      p.traded ? (
        <span className="flex items-center gap-2">
          <SideChip side={p.side} />
          <span className="font-mono text-micro text-ink-3">{p.contracts}x</span>
        </span>
      ) : (
        <Chip tone="neutral">{p.reason.replace(/_/g, ' ')}</Chip>
      ),
  },
];
