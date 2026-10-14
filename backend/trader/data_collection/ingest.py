"""The single path into storage, and the only place that sets `quality`.

Every record defaults to `DataQuality.UNVALIDATED`, so a record that bypasses
this module stays visible in the data. VALID and SUSPICIOUS records are stored;
INVALID records are dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .models import DataQuality, OHLCVBar
from .validator import DataQualityTracker, DataValidator, ValidationConfig

logger = logging.getLogger(__name__)

# SUSPICIOUS is stored so that a downstream consumer can decide.
STORABLE = (DataQuality.VALID, DataQuality.SUSPICIOUS)


@dataclass
class IngestResult:
    """What happened to a batch."""

    received: int = 0
    inserted: int = 0
    rejected: int = 0
    suspicious: int = 0
    issues: list[str] = field(default_factory=list)
    # The records written, in order. Callers advance watermarks from these only;
    # a rejected record must not move a watermark.
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

    Hold one per run, so that the quality tracker accumulates across batches.
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

        `venue` has no default, because every bar must record its origin.
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
                # Only a stored bar is the reference for the next gap and jump
                # check, so one bad record cannot reject the good ones after it.
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

    A non-zero count is legacy data or a bypass of this module.
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
