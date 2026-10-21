/** The small pieces: panels, section headers, numbers, states, tables.
 *
 * Collected in one file because each is a dozen lines and splitting them across
 * eight modules makes the design system harder to read than the components are.
 */
import { useState } from 'react';
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
    // Stacked on phones. `right` is `shrink-0`, so side by side at 375px it
    // squeezed the note into a ~20ch column and still pushed the last chip off
    // the right edge.
    <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
        <h2 className="text-lg font-semibold text-ink">{title}</h2>
        {note && <p className="mt-1 max-w-[62ch] text-tiny text-ink-2">{note}</p>}
      </div>
      {right && <div className="shrink-0 sm:shrink-0">{right}</div>}
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
    // **`break-words` on the label, but NO `min-w-0` on the container.**
    //
    // The label must wrap: a long reason (`price out of band`) was being
    // CLIPPED mid-word in a narrow table cell — `PRICE OU` — which reads as a
    // different reason rather than a truncated one.
    //
    // But `min-w-0` here was worse. It lets the chip shrink below its own
    // content inside a flex row, and the Live page's banner is exactly that:
    // `<Chip>live</Chip>` beside a long sentence. The chip collapsed to one
    // character wide and rendered L / I / V / E stacked vertically. Default
    // `min-width: auto` keeps the chip at its content size in a flex row,
    // while a table cell still constrains it and the label still wraps.
    <span
      className={`inline-flex items-center gap-1.5 font-mono text-micro uppercase ${text}`}
      title={title}
    >
      <span className={`h-2 w-2 shrink-0 ${swatch}`} aria-hidden />
      <span className="break-words">{children}</span>
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
  return (
    <div className="border border-fail/40 bg-below-wash px-4 py-3">
      <div className="eyebrow text-fail">request failed</div>
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
  /** Hide this column below `sm`.
   *
   *  **A dense research table does not fit a phone.** `DecisionsPage` carries
   *  eleven columns, which measured 802px inside a 375px viewport: the row
   *  wrapped to 51px tall and reading one decision meant scrolling sideways.
   *  Density is right on a desktop and wrong on a phone, so the column says
   *  which it is rather than the page being built twice.
   *
   *  Mark the columns that support a reading, not the ones that ARE the
   *  reading — hiding `edge` to save width would defeat the purpose. */
  secondary?: boolean;
  render: (row: T) => ReactNode;
}

export function DataTable<T>({
  columns,
  rows,
  keyOf,
  empty,
  pageSize = 25,
}: {
  columns: Column<T>[];
  rows: T[];
  keyOf: (row: T, index: number) => string;
  empty?: ReactNode;
  /** Rows per page.
   *
   *  **Paged rather than truncated.** This used to render the first 60 rows
   *  and print "138 older not shown", which is the one behaviour a ledger must
   *  not have: the rows it silently dropped were the ones you would go looking
   *  for. Everything is reachable now, and a page of 25 keeps the table about
   *  a screen tall on a laptop and scrollable rather than endless on a phone.
   *
   *  Every table on the dashboard takes the same default, so paging works the
   *  same way on the Account page and on Decisions. */
  pageSize?: number;
}) {
  // **Not reset when `rows` changes.** These pages poll every second, so the
  // array is a new object on every tick; resetting on identity would snap the
  // reader back to page one once a second. The index is only clamped, so a
  // page that stops existing falls back to the last one that does.
  const [requested, setRequested] = useState(0);

  if (rows.length === 0) return <>{empty ?? <Empty what="Nothing recorded yet." />}</>;

  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const page = Math.min(requested, pages - 1);
  const from = page * pageSize;
  const shown = rows.slice(from, from + pageSize);

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="grid-table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  style={c.width ? { width: c.width } : undefined}
                  className={[
                    // `num`, not `text-right` — see the note in index.css.
                    c.numeric ? 'num' : '',
                    c.secondary ? 'hidden sm:table-cell' : '',
                  ].filter(Boolean).join(' ') || undefined}
                >
                  {c.head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, index) => (
              <tr key={keyOf(row, from + index)} className="hover:bg-sunken">
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={[
                      c.numeric ? 'num' : '',
                      c.secondary ? 'hidden sm:table-cell' : '',
                    ].filter(Boolean).join(' ') || undefined}
                  >
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div className="flex items-center justify-between gap-2 border-t border-rule bg-sunken px-2 py-1.5 sm:px-3">
          <button
            type="button"
            onClick={() => setRequested(Math.max(0, page - 1))}
            disabled={page <= 0}
            className="rounded px-2 py-1 font-mono text-micro uppercase tracking-wider text-ink-2 transition-colors enabled:hover:bg-raised enabled:hover:text-ink disabled:opacity-30"
          >
            ‹ <span className="hidden sm:inline">prev</span>
          </button>
          {/* The ROW RANGE, not just the page number: "151–175 of 198" says
              where you are in the record; "page 7 of 8" only says where you are
              in the pagination. */}
          <span className="font-mono text-micro uppercase tracking-wider text-ink-3">
            {(from + 1).toLocaleString()}–{(from + shown.length).toLocaleString()}
            {' '}of {rows.length.toLocaleString()}
          </span>
          <button
            type="button"
            onClick={() => setRequested(Math.min(pages - 1, page + 1))}
            disabled={page >= pages - 1}
            className="rounded px-2 py-1 font-mono text-micro uppercase tracking-wider text-ink-2 transition-colors enabled:hover:bg-raised enabled:hover:text-ink disabled:opacity-30"
          >
            <span className="hidden sm:inline">next</span> ›
          </button>
        </div>
      )}
    </div>
  );
}

/** A link to another page of the dashboard, as a card.
 *
 *  **An anchor, not a button.** `href` is what makes middle-click, cmd-click
 *  and "copy link address" work; the click handler only stops the full page
 *  reload. A div with an onClick looks identical and does none of that.
 */
export function PageLink({
  href,
  title,
  note,
  count,
  onNavigate,
}: {
  href: string;
  title: string;
  note: string;
  count?: ReactNode;
  onNavigate: (path: string) => void;
}) {
  return (
    <a
      href={href}
      onClick={(event) => {
        // Leave the modified clicks to the browser: cmd/ctrl/shift/middle all
        // mean "open this somewhere else", which this component must not eat.
        if (event.defaultPrevented || event.metaKey || event.ctrlKey ||
            event.shiftKey || event.altKey || event.button !== 0) return;
        event.preventDefault();
        onNavigate(href);
      }}
      className="group flex items-center justify-between gap-3 rounded border border-rule bg-raised px-3 py-3 transition-colors hover:border-ink-3 hover:bg-sunken"
    >
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-ink">{title}</span>
        <span className="mt-0.5 block text-tiny text-ink-3">{note}</span>
      </span>
      <span className="flex shrink-0 items-center gap-2">
        {count != null && (
          <span className="font-mono text-tiny tabular-nums text-ink-2">{count}</span>
        )}
        <span className="text-ink-3 transition-transform group-hover:translate-x-0.5">›</span>
      </span>
    </a>
  );
}

/* --------------------------------------------------------------------- time */

