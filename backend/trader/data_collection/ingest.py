"""The single door into storage.

Every record that reaches the database goes through here, and here is the only
place that sets `quality`. `DataQuality.UNVALIDATED` is the default on every
record, so anything that still finds its way around this module is visible in
the data rather than indistinguishable from a verified row:

    SELECT COUNT(*) FROM ohlcv WHERE quality = 'unvalidated'

Rejection policy matches what the pipeline already did for bars: VALID and
SUSPICIOUS are stored, INVALID is dropped. Suspicious data is usable with a
caveat; invalid data is not usable at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .models import DataQuality, OHLCVBar
from .validator import DataQualityTracker, DataValidator, ValidationConfig

logger = logging.getLogger(__name__)

# Quality levels that are worth keeping. SUSPICIOUS is stored so a downstream
# consumer can decide, rather than being silently discarded here.
STORABLE = (DataQuality.VALID, DataQuality.SUSPICIOUS)


@dataclass
class IngestResult:
    """What happened to a batch."""

    received: int = 0
    inserted: int = 0
    rejected: int = 0
    suspicious: int = 0
    issues: list[str] = field(default_factory=list)
    # The records that passed and were written, in order. Callers use these to
    # advance watermarks; a rejected record must not move a watermark.
    stored: list = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return self.received - self.rejected

    def __str__(self) -> str:
        parts = [f"{self.inserted}/{self.received} inserted"]
        if self.suspicious:
            parts.append(f"{self.suspicious} suspicious")
        if self.rejected:
            parts.append(f"{self.rejected} rejected")
        return ", ".join(parts)


class Ingestor:
    """Validates and stores market data.

    Hold one per run so the quality tracker accumulates across batches; its
    summary is the answer to "how good was today's data".
    """

    def __init__(
        self,
        database,
        *,
        validator: Optional[DataValidator] = None,
        tracker: Optional[DataQualityTracker] = None,
        validation_config: Optional[ValidationConfig] = None,
    ):
        self.database = database
        self.validator = validator or DataValidator(validation_config or ValidationConfig())
        self.tracker = tracker or DataQualityTracker()

    # -- bars ---------------------------------------------------------------

    def ingest_bars(
        self,
        bars: Sequence[OHLCVBar],
        *,
        venue: str,
        previous_bar: Optional[OHLCVBar] = None,
    ) -> IngestResult:
        """Validate and store bars, stamping the venue they came from.

        `venue` is required rather than defaulted: a bar whose origin is unknown
        is the problem this parameter exists to solve.
        """
        result = IngestResult(received=len(bars))
        if not bars:
            return result

        keep: list[OHLCVBar] = []
        prior = previous_bar
        for bar in bars:
            bar.venue = venue
            outcome = self.validator.validate_ohlcv(bar, prior)
            self.tracker.record_validation(outcome)
            bar.quality = outcome.quality
            bar.quality_notes = '; '.join(outcome.issues) or None

            if outcome.quality in STORABLE:
                keep.append(bar)
                if outcome.quality == DataQuality.SUSPICIOUS:
                    result.suspicious += 1
                # Only a stored bar becomes the reference for the next gap and
                # price-jump check; comparing against a rejected bar would
                # cascade one bad record into rejecting the good ones after it.
                prior = bar
            else:
                result.rejected += 1
                result.issues.extend(outcome.issues)

        result.stored = keep
        if keep:
            result.inserted = self.database.insert_ohlcv_batch(keep)
        return self._log('bars', venue, result)

    # -- reporting ----------------------------------------------------------

    def quality_summary(self) -> dict:
        return self.tracker.get_summary()

    @staticmethod
    def _log(dataset: str, venue: str, result: IngestResult) -> IngestResult:
        if result.rejected:
            sample = '; '.join(dict.fromkeys(result.issues))[:200]
            logger.warning(
                "%s/%s: %s — rejected reasons: %s", venue, dataset, result, sample
            )
        elif result.received:
            logger.info("%s/%s: %s", venue, dataset, result)
        return result


def unvalidated_row_counts(database) -> dict[str, int]:
    """Rows that reached storage without passing through `Ingestor`.

    Should be zero on a database written entirely by the current code. A non-zero
    count is either legacy data or a new bypass; either way it is a fact about
    the data rather than a hidden assumption.
    """
    counts: dict[str, int] = {}
    connection = getattr(database, '_connection', None) or getattr(database, 'conn', None)
    if connection is None:
        return counts
    for table in ('ohlcv',):
        try:
            cursor = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE quality = ?",
                (DataQuality.UNVALIDATED.value,),
            )
            counts[table] = int(cursor.fetchone()[0])
        except Exception:            # table absent on a fresh database
            continue
    return counts
