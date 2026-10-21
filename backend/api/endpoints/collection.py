"""Data-collection health: are rows arriving, and are the loops cycling.

Separate from `endpoints/serving.py` because it reads neither the database nor
the serving store — it reads what the trader publishes on the shared data
volume. No `Depends(get_db)` anywhere below, which is the point: this surface
must still answer when the thing it reports on is broken.
"""

from __future__ import annotations

from fastapi import APIRouter

from controllers import collection

router = APIRouter(tags=['collection'])


@router.get('/collection')
def get_collection_health():
    """Dataset freshness and component liveness, side by side and labelled.

    They are different questions. `venue_ladder` stopped writing on 2026-09-16
    and was dead for thirty-one days with every heartbeat green: the recorder
    cycled happily and produced nothing. A surface that showed only liveness
    would have reported that month as healthy.

    Neither half is ever defaulted. An absent measurement is `available: false`
    with a reason, a dataset with no rows is a null age with a reason, and a
    component with no published limit gets a measured age and a null verdict.
    """
    return collection.collection_health()


@router.get('/collection/freshness')
def get_dataset_freshness():
    """Per dataset: newest row, its age now, the limit, and whether it is late.

    `measurement_age_seconds` is how old the trader's own measurement is. It
    matters: rows may have landed since it was taken, so a stale measurement
    makes every age here an upper bound rather than a reading.
    """
    return collection.dataset_freshness()


@router.get('/collection/liveness')
def get_component_liveness():
    """Per component: seconds since its loop last completed a pass.

    Process liveness, not data freshness — a loop can cycle forever and write
    nothing, which is exactly what happened. Read `/collection/freshness` for
    the other question.
    """
    return collection.component_liveness()
