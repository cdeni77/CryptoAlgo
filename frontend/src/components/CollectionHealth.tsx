/** Is the data still arriving, and is the loop still cycling?
 *
 * **Two questions, two lists, never one verdict.** `venue_ladder` stopped
 * writing on 2026-09-16 and nobody noticed until 2026-10-17, because every
 * component heartbeat stayed green for all thirty-one days: the recorder's
 * loop ran happily and produced nothing. A panel that showed only liveness
 * would have rendered that month as healthy, and a panel that merged the two
 * into one light would have shown green for the same reason. So freshness
 * (rows landing) and liveness (loops cycling) are side by side and labelled,
 * and the pair being different IS the finding.
 *
 * Nothing here invents a state. A dataset whose age was not measured renders
 * an em-dash and a reason, not `0s` and not "ok" — which is the whole point,
 * because the failure being designed against is a dead feed that looks alive.
 */
import { fetchCollectionHealth } from '../api/serving';
import { usePolling } from '../hooks/usePolling';
import { duration, stamp } from '../lib/format';
import { Chip, Empty, Failed, Loading, Panel, SectionHead } from './Primitives';
import type { ComponentLiveness, DatasetFreshness } from '../types';

/** One row: a name, an age, and a state that is allowed to be "unknown". */
function Row({
  name,
  age,
  limit,
  stale,
  reason,
}: {
  name: string;
  age: number | null;
  limit: number | null;
  stale: boolean | null;
  reason: string | null;
}) {
  // Three states, not two. `null` is "nothing was measured", and it takes the
  // neutral chip and the reason on hover rather than being folded into either
  // "ok" or "stale".
  const tone = stale == null ? 'neutral' : stale ? 'fail' : 'pass';
  const label = stale == null ? 'unknown' : stale ? 'stale' : 'ok';
  const title =
    reason ??
    (limit == null ? undefined : `expected within ${duration(limit)}`);

  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-rule py-1.5 last:border-b-0">
      <span className="min-w-0 truncate font-mono text-tiny text-ink-2" title={name}>
        {name}
      </span>
      <span className="flex shrink-0 items-baseline gap-3">
        <span
          className={`font-mono text-tiny tabular-nums ${
            stale ? 'text-fail' : 'text-ink'
          }`}
          title={title}
        >
          {duration(age)}
        </span>
        <span className="w-[4.5rem] text-right">
          <Chip tone={tone} title={title}>
            {label}
          </Chip>
        </span>
      </span>
    </div>
  );
}

export function CollectionHealthPanel() {
  // A minute. These are hourly and minute-grid cadences; polling them every
  // few seconds would measure the poller rather than the feeds.
  const health = usePolling(fetchCollectionHealth, 60_000);

  if (health.loading) return <Loading what="collection health" />;
  if (health.error) return <Failed error={health.error} what="collection health" />;
  const data = health.data;
  if (!data) return null;

  const { freshness, liveness } = data;
  // Three populations, because there are three states. Counting "unknown" as
  // either late or current would put the dead feed back behind a word.
  const rows = [
    ...freshness.datasets.map((d) => d.stale),
    ...liveness.components.map((c) => c.stale),
  ];
  const late = rows.filter((s) => s === true).length;
  const unknown = rows.filter((s) => s == null).length;
  const measured = freshness.available && liveness.available;

  return (
    <section>
      <SectionHead
        eyebrow={
          freshness.available
            ? `${freshness.datasets.length} datasets · ${liveness.components.length} components`
            : 'not measured'
        }
        title="Collection"
        note="Two different questions. Freshness asks whether rows are still landing; liveness asks whether each loop finished a pass. venue_ladder was dead for thirty-one days with every heartbeat green — a loop can cycle forever and write nothing."
        right={
          <Chip
            tone={late ? 'fail' : !measured || unknown ? 'neutral' : 'pass'}
            title={
              unknown
                ? `${unknown} of ${rows.length} were not measured — hover each row for why`
                : undefined
            }
          >
            {late
              ? `${late} late`
              : !measured
                ? 'unmeasured'
                : unknown
                  ? `${unknown} unmeasured`
                  : 'all current'}
          </Chip>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel>
          <div className="eyebrow mb-2">freshness · rows arriving</div>
          {freshness.available ? (
            <>
              <div>
                {freshness.datasets.map((d: DatasetFreshness) => (
                  <Row
                    key={d.dataset}
                    name={d.dataset}
                    age={d.age_seconds}
                    limit={d.limit_seconds}
                    stale={d.stale}
                    reason={d.reason}
                  />
                ))}
              </div>
              {/* How old the MEASUREMENT is, which bounds what any age above
                  it can be read as. Rows may have landed since it was taken. */}
              <p
                className={`mt-3 border-t border-rule pt-2 font-mono text-micro ${
                  freshness.measurement_stale ? 'text-warn' : 'text-ink-3'
                }`}
              >
                measured {stamp(freshness.measured_at)} ET ·{' '}
                {duration(freshness.measurement_age_seconds)} ago
                {freshness.measurement_stale
                  ? ' — every age above is an upper bound'
                  : ''}
              </p>
            </>
          ) : (
            <Empty
              what={freshness.reason ?? 'The trader has published no freshness measurement.'}
              next="python -m scripts.check_freshness --publish"
            />
          )}
        </Panel>

        <Panel>
          <div className="eyebrow mb-2">liveness · loops cycling</div>
          {liveness.available && liveness.components.length > 0 ? (
            <>
              <div>
                {liveness.components.map((c: ComponentLiveness) => (
                  <Row
                    key={c.component}
                    name={c.component}
                    age={c.age_seconds}
                    limit={c.limit_seconds}
                    stale={c.stale}
                    reason={c.reason}
                  />
                ))}
              </div>
              {liveness.reason && (
                <p className="mt-3 border-t border-rule pt-2 text-micro text-ink-3">
                  {liveness.reason}
                </p>
              )}
            </>
          ) : (
            <Empty
              what={
                liveness.reason ??
                'No component has reported a completed cycle. That is "not started yet" and "died before its first cycle" at once, and the second is the one worth catching.'
              }
              next="python -m scripts.healthcheck"
            />
          )}
        </Panel>
      </div>
    </section>
  );
}
