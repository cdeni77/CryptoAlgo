/** What is at stake right now, above everything else on the page.
 *
 *  The third question a trader asks on landing — after "is it running" and
 *  "what is it doing" — is "what am I exposed to". That answer used to be
 *  most of a page away, under the window cards and a panel heading.
 *
 *  Every figure is `Measured`, so a value the account could not supply reads
 *  as unavailable with its reason rather than as zero. A missing number must
 *  never be presented as a safe one.
 */
import type { AccountState } from '../types';
import { Metric } from './Primitives';

export function ExposureStrip({ account }: { account: AccountState }) {
  const live = account.mode === 'live';
  return (
    <section
      className={`rounded border px-4 py-3 ${
        account.halted
          ? 'border-fail/40 bg-below-wash'
          : live
            ? 'border-below/30 bg-surface'
            : 'border-rule bg-surface'
      }`}
    >
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span
          className={`font-mono text-micro uppercase tracking-wider ${
            live ? 'text-below' : 'text-ink-3'
          }`}
        >
          {live ? 'live · real money' : 'paper'}
        </span>
        {account.halted && (
          <span className="font-mono text-micro uppercase tracking-wider text-fail">
            halted
          </span>
        )}
        {account.halted && account.halted_reason && (
          <span className="text-tiny text-ink-2">{account.halted_reason}</span>
        )}
        {!account.halted && (
          <span className="text-tiny text-ink-3">
            {live
              ? 'Positions below are actual exposure, not a simulation.'
              : 'Simulated account. No order reaches the venue.'}
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-6">
        <Metric
          label="equity"
          value={account.equity}
          unit="$"
          size="lg"
          hint="bankroll plus open stake at COST — never marked to our own forecast"
        />
        <Metric label="cash" value={account.bankroll} unit="$" />
        <Metric
          label="at risk"
          value={account.staked}
          unit="$"
          hint="what the open positions cost, which is the most they can lose"
        />
        <Metric
          label="open"
          value={account.open_positions}
          digits={0}
          hint="positions not yet settled"
        />
        <Metric
          label="realised"
          value={account.realized_pnl}
          unit="$"
          tone={(account.realized_pnl?.value ?? 0) >= 0 ? 'above' : 'below'}
          hint="settled positions only, since the dashboard epoch"
        />
        <Metric label="fees" value={account.fees_paid} unit="$" />
      </div>
    </section>
  );
}
