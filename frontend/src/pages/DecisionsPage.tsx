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
import { cents, pct, signedPp, stamp } from '../lib/format';
import type { Prediction } from '../types';

export function DecisionsPage() {
  const [tradedOnly, setTradedOnly] = useState(false);
  const state = usePolling(() => fetchPredictions(250, tradedOnly), 30_000, [tradedOnly]);

  const rows = state.data?.predictions ?? [];
  const traded = rows.filter((r) => r.traded).length;

  return (
    <div className="space-y-6">
      <SectionHead
        eyebrow={`${rows.length} decisions · ${traded} traded`}
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
                onClick={() => setTradedOnly(option.value)}
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
            empty={<Empty what="No decisions recorded." next="python -m scripts.paper" />}
          />
        )}
      </Panel>
    </div>
  );
}

const columns: Column<Prediction>[] = [
  { key: 'when', head: 'window', render: (p) => <span className="font-mono">{stamp(p.window_open)}</span> },
  { key: 'symbol', head: 'symbol', render: (p) => <span className="font-mono">{p.symbol}</span> },
  { key: 'offset', head: 'at', numeric: true, render: (p) => `+${p.offset_minutes}m` },
  {
    key: 'displacement',
    head: 'disp.',
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
    numeric: true,
    render: (p) => (p.sigma_remaining == null ? '—' : `${(p.sigma_remaining * 10_000).toFixed(1)}bp`),
  },
  {
    key: 'scale',
    head: 'forecast vs market',
    width: '13rem',
    render: (p) => (
      <div className="py-0.5">
        <ProbabilityScale
          probability={p.model_probability}
          price={p.price ?? p.baseline_probability}
          breakEven={p.effective_cost}
          height={12}
          showAxis={false}
        />
      </div>
    ),
  },
  { key: 'q', head: 'model', numeric: true, render: (p) => pct(p.model_probability, 1) },
  { key: 'base', head: 'baseline', numeric: true, render: (p) => <span className="text-ink-3">{pct(p.baseline_probability, 1)}</span> },
  { key: 'price', head: 'price', numeric: true, render: (p) => cents(p.price) },
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
