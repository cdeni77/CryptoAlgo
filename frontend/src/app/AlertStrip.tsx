/** The one place a severe condition is allowed to interrupt.
 *
 *  Reserved for halted trading, a loop that has stopped deciding, and
 *  collectors that are not receiving rows. Routine notifications do not
 *  belong here: a strip that is always lit is a strip nobody reads, which is
 *  the same failure as a gate that always fails.
 */
import { useState } from 'react';

import type { Alert, Severity } from '../lib/status';

const TONE: Record<Severity, { box: string; dot: string; label: string }> = {
  critical: {
    box: 'border-fail/40 bg-below-wash',
    dot: 'bg-fail',
    label: 'text-fail',
  },
  warning: {
    box: 'border-warn/40 bg-sunken',
    dot: 'bg-warn',
    label: 'text-warn',
  },
  info: {
    box: 'border-rule bg-sunken',
    dot: 'bg-ink-3',
    label: 'text-ink-2',
  },
};

export function AlertStrip({ alerts }: { alerts: Alert[] }) {
  const [dismissed, setDismissed] = useState<string[]>([]);
  const showing = alerts.filter((a) => !dismissed.includes(a.id));
  if (!showing.length) return null;

  return (
    <div className="flex flex-col gap-2 px-4 pt-3 md:px-8">
      {showing.map((alert) => {
        const tone = TONE[alert.severity];
        return (
          <div
            key={alert.id}
            role={alert.severity === 'critical' ? 'alert' : 'status'}
            className={`flex flex-wrap items-start gap-x-3 gap-y-1 rounded border px-3 py-2 ${tone.box}`}
          >
            {/* A dot AND a word: colour alone cannot carry a state. */}
            <span className="flex shrink-0 items-center gap-2 pt-0.5">
              <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} aria-hidden />
              <span className={`font-mono text-micro uppercase tracking-wider ${tone.label}`}>
                {alert.severity}
              </span>
            </span>
            <span className="min-w-0 text-tiny font-medium text-ink">{alert.title}</span>
            {alert.detail && (
              <span className="min-w-0 flex-1 text-tiny text-ink-3">{alert.detail}</span>
            )}
            {/* Warnings can be set aside for this session; a critical one
                cannot, because the condition is still true. */}
            {alert.severity !== 'critical' && (
              <button
                type="button"
                onClick={() => setDismissed((d) => [...d, alert.id])}
                className="ml-auto shrink-0 rounded px-1 font-mono text-micro uppercase tracking-wider text-ink-3 hover:text-ink"
              >
                hide
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
