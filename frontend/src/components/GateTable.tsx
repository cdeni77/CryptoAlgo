import { GateResult } from '../types';

/**
 * The promotion gates, failures first.
 *
 * This is the screen the whole research pipeline exists to produce, so it shows
 * the measured value beside its threshold rather than a pass/fail badge alone: a
 * gate that missed by 0.01 and one that missed by an order of magnitude need
 * different responses, and a red dot cannot tell them apart.
 *
 * A gate with no measurement reads "not measured" and counts as a failure, which
 * is the intended direction — "we did not run that test" is not evidence of
 * safety.
 */

/** Plain-language notes on what each gate is protecting against. */
const EXPLANATIONS: Record<string, string> = {
  walk_forward_median_sharpe:
    'Median Sharpe across the out-of-sample paths. The middle of the distribution, not the best path.',
  walk_forward_p05_sharpe:
    'Fifth-percentile path. A strategy whose bad paths lose money is a strategy that will, eventually.',
  pbo: 'Probability of backtest overfitting. How often the configuration that won in-sample loses out-of-sample.',
  deflated_sharpe:
    'Sharpe discounted for how many configurations were tried. The best of fifty random strategies looks good.',
  bootstrap_positive_fraction:
    'Share of resampled trade orderings that made money. The same trades in a different order.',
  synthetic_positive_fraction:
    'Share of simulated price paths that made money. Measures robustness and sizing, never edge.',
  stressed_median_sharpe: 'Sharpe when costs are worse than assumed.',
  parameter_plateau:
    'Share of neighbouring parameter settings that still work. A spike is a fit to noise; a plateau is a mechanism.',
  oos_trades: 'Out-of-sample trades. Below a hundred, every statistic is noise.',
  max_exit_participation:
    'Largest share of a bar an exit took. Above this, the fills are fiction at the size claimed.',
};

function formatValue(gate: GateResult): string {
  if (gate.value === null || gate.value === undefined) return 'not measured';
  const magnitude = Math.abs(gate.value);
  if (magnitude >= 1000) return gate.value.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (magnitude >= 10) return gate.value.toFixed(1);
  return gate.value.toFixed(3);
}

function formatThreshold(gate: GateResult): string {
  const symbol = gate.comparison === 'min' ? '≥' : '≤';
  const magnitude = Math.abs(gate.threshold);
  const shown =
    magnitude >= 1000
      ? gate.threshold.toLocaleString('en-US', { maximumFractionDigits: 0 })
      : magnitude >= 10
        ? gate.threshold.toFixed(0)
        : gate.threshold.toFixed(2);
  return `${symbol} ${shown}`;
}

/** How far a gate is from its threshold, as a fraction, for the bar width. */
function progress(gate: GateResult): number {
  if (gate.value === null || gate.value === undefined) return 0;
  if (gate.threshold === 0) return gate.passed ? 1 : 0;
  const ratio =
    gate.comparison === 'min'
      ? gate.value / gate.threshold
      : gate.threshold / Math.max(gate.value, 1e-9);
  return Math.max(0, Math.min(1, ratio));
}

export default function GateTable({ gates }: { gates: GateResult[] }) {
  if (!gates.length) {
    return (
      <div className="py-8 text-center text-sm text-tx-muted">
        No gate results recorded for this candidate.
      </div>
    );
  }

  // Failures first: they are the reason nothing was promoted.
  const ordered = [...gates].sort((a, b) => {
    if (a.passed !== b.passed) return a.passed ? 1 : -1;
    return a.name.localeCompare(b.name);
  });
  const failed = gates.filter((g) => !g.passed).length;

  return (
    <div>
      <div className="mb-3 flex items-baseline gap-2">
        <span
          className={`text-sm font-semibold ${failed ? 'text-accent-rose' : 'text-accent-emerald'}`}
        >
          {failed ? `Blocked by ${failed} gate${failed === 1 ? '' : 's'}` : 'All gates passed'}
        </span>
        <span className="font-mono text-[11px] text-tx-muted">
          {gates.length - failed}/{gates.length}
        </span>
      </div>

      <div className="space-y-2">
        {ordered.map((gate) => (
          <div
            key={gate.name}
            className={`rounded-lg border px-3 py-2.5 ${
              gate.passed
                ? 'border-[rgba(56,189,248,0.08)] bg-surface-2/40'
                : 'border-accent-rose/25 bg-accent-rose/5'
            }`}
          >
            <div className="flex items-baseline justify-between gap-3">
              <span className="font-mono text-xs text-tx-primary">{gate.name}</span>
              <span className="flex-shrink-0 font-mono text-xs tabular-nums">
                <span className={gate.passed ? 'text-accent-emerald' : 'text-accent-rose'}>
                  {formatValue(gate)}
                </span>
                <span className="ml-1.5 text-tx-muted">{formatThreshold(gate)}</span>
              </span>
            </div>

            <div className="mt-2 h-0.5 overflow-hidden rounded-full bg-surface-3">
              <div
                className={`h-full ${gate.passed ? 'bg-accent-emerald/70' : 'bg-accent-rose/70'}`}
                style={{ width: `${progress(gate) * 100}%` }}
              />
            </div>

            {(gate.note || EXPLANATIONS[gate.name]) && (
              <div className="mt-1.5 text-[11px] leading-snug text-tx-muted">
                {gate.note ?? EXPLANATIONS[gate.name]}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
