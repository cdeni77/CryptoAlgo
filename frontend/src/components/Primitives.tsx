/** The small pieces: panels, section headers, numbers, states, tables.
 *
 * Collected in one file because each is a dozen lines and splitting them across
 * eight modules makes the design system harder to read than the components are.
 */
import type { ReactNode } from 'react';
import type { Measured } from '../types';
import { formatNumber } from '../lib/format';

/* ---------------------------------------------------------------- structure */

export function Panel({
  children,
  className = '',
  flush = false,
}: {
  children: ReactNode;
  className?: string;
  flush?: boolean;
}) {
  return (
    <section className={`panel ${flush ? '' : 'p-4'} ${className}`}>{children}</section>
  );
}

/** A section header. The eyebrow carries a count or a span — real information —
 *  and the note carries the one sentence that says how to read what follows. */
export function SectionHead({
  eyebrow,
  title,
  note,
  right,
}: {
  eyebrow?: string;
  title: string;
  note?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-3 flex items-start justify-between gap-6">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
        <h2 className="text-lg font-semibold text-ink">{title}</h2>
        {note && <p className="mt-1 max-w-[62ch] text-tiny text-ink-2">{note}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ numbers */

/** A number with its label and unit, tabular, with an explicit missing state.
 *
 * `Measured<number>` from the API distinguishes "not measured, and here is why"
 * from zero. Rendering an em-dash with the reason in the tooltip is the whole
 * reason that distinction exists — the previous surface substituted plausible
 * numbers for absent ones and they rendered identically to real data.
 */
export function Metric({
  label,
  value,
  unit,
  digits = 2,
  tone = 'ink',
  hint,
  size = 'base',
}: {
  label: string;
  value: number | null | Measured<number>;
  unit?: string;
  digits?: number;
  // `warn` is for a number that is neither good nor bad but wants looking at:
  // a reconciliation gap against the venue is the case it was added for. It
  // borrows the gate palette rather than the directional one, because a drift of
  // -$0.40 is not "price down".
  tone?: 'ink' | 'above' | 'below' | 'accent' | 'muted' | 'warn';
  hint?: string;
  size?: 'base' | 'lg' | 'xl';
}) {
  const resolved = typeof value === 'object' && value !== null && 'value' in value
    ? value
    : ({ value: value as number | null, reason: null } as Measured<number>);
  const toneClass = {
    ink: 'text-ink',
    above: 'text-above',
    below: 'text-below',
    accent: 'text-accent',
    muted: 'text-ink-3',
    warn: 'text-warn',
  }[tone];
  const sizeClass = { base: 'text-mid', lg: 'text-xl', xl: 'text-2xl' }[size];

  return (
    <div>
      <div className="eyebrow" title={hint}>
        {label}
      </div>
      <div className={`mt-0.5 font-mono font-medium ${sizeClass} ${toneClass}`}>
        {resolved.value == null ? (
          <span className="text-ink-3" title={resolved.reason ?? 'not measured'}>
            —
          </span>
        ) : (
          <>
            {formatNumber(resolved.value, digits)}
            {unit && <span className="ml-0.5 text-tiny font-normal text-ink-3">{unit}</span>}
          </>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------- states */

/** A state chip. A 2px square of colour and a word — not a rounded pill, and
 *  never a gradient. Gate semantics use their own colours so "failed" cannot be
 *  mistaken for "price down". */
export function Chip({
  tone,
  children,
  title,
}: {
  tone: 'pass' | 'fail' | 'warn' | 'above' | 'below' | 'neutral' | 'accent';
  children: ReactNode;
  title?: string;
}) {
  const swatch = {
    pass: 'bg-pass',
    fail: 'bg-fail',
    warn: 'bg-warn',
    above: 'bg-above',
    below: 'bg-below',
    neutral: 'bg-ink-3',
    accent: 'bg-accent',
  }[tone];
  const text = {
    pass: 'text-pass',
    fail: 'text-fail',
    warn: 'text-warn',
    above: 'text-above',
    below: 'text-below',
    neutral: 'text-ink-3',
    accent: 'text-accent',
  }[tone];
  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono text-micro uppercase ${text}`}
      title={title}
    >
      <span className={`h-2 w-2 ${swatch}`} aria-hidden />
      {children}
    </span>
  );
}

export function SideChip({ side }: { side: 'up' | 'down' | null }) {
  if (!side) return <span className="text-ink-3">—</span>;
  return <Chip tone={side === 'up' ? 'above' : 'below'}>{side}</Chip>;
}

/* -------------------------------------------------------------------- empty */

/** What to show when there is nothing yet, with the command that fixes it.
 *  A blank panel says "broken"; a blank panel with the next step says "waiting". */
export function Empty({ what, next }: { what: string; next?: string }) {
  return (
    <div className="border border-dashed border-rule px-4 py-8 text-center">
      <p className="text-tiny text-ink-2">{what}</p>
      {next && (
        <code className="mt-2 inline-block bg-sunken px-2 py-1 font-mono text-micro text-ink-2">
          {next}
        </code>
      )}
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return (
    <div className="px-4 py-8 text-center font-mono text-micro uppercase text-ink-3">
      loading {what}
    </div>
  );
}

export function Failed({ error, what }: { error: Error; what: string }) {
  const auth = 'isAuthProblem' in error && (error as { isAuthProblem: boolean }).isAuthProblem;
  return (
    <div className="border border-fail/40 bg-below-wash px-4 py-3">
      <div className="eyebrow text-fail">{auth ? 'not authorised' : 'request failed'}</div>
      <p className="mt-1 text-tiny text-ink">
        Could not load {what}. {error.message}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------- tables */

export interface Column<T> {
  key: string;
  head: string;
  numeric?: boolean;
  width?: string;
  render: (row: T) => ReactNode;
}

export function DataTable<T>({
  columns,
  rows,
  keyOf,
  empty,
  maxRows = 60,
}: {
  columns: Column<T>[];
  rows: T[];
  keyOf: (row: T, index: number) => string;
  empty?: ReactNode;
  /** Rows rendered. An uncapped table of a few hundred settlements produces a
   *  page fourteen thousand pixels tall, which is not a table anyone reads —
   *  and the count of what is hidden is itself information, so it is stated. */
  maxRows?: number;
}) {
  if (rows.length === 0) return <>{empty ?? <Empty what="Nothing recorded yet." />}</>;
  const shown = rows.slice(0, maxRows);
  const hidden = rows.length - shown.length;
  return (
    <div className="overflow-x-auto">
      <table className="grid-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                style={c.width ? { width: c.width } : undefined}
                className={c.numeric ? 'text-right' : undefined}
              >
                {c.head}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, index) => (
            <tr key={keyOf(row, index)} className="hover:bg-sunken">
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? 'num' : undefined}>
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {hidden > 0 && (
        <p className="border-t border-rule bg-sunken px-3 py-2 font-mono text-micro uppercase text-ink-3">
          showing {shown.length.toLocaleString()} of {rows.length.toLocaleString()} ·{' '}
          {hidden.toLocaleString()} older not shown
        </p>
      )}
    </div>
  );
}

/* --------------------------------------------------------------------- time */

