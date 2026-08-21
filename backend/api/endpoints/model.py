"""Model provenance, promotion gates, and the promotion ledger.

Read-only. Promotion itself happens in the trader, through `scripts.promote`, and
the dashboard's promote action goes through `POST /research/launch/promote`, which
is authenticated. Keeping the decision out of this router is deliberate: there
must be exactly one thing that can install a model, and it is the one that runs
the gates.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from controllers.model import get_feature_importance, get_live_model, get_promotion_history
from models.model import (
    FeatureImportanceResponse,
    LiveModelResponse,
    PromotionHistoryResponse,
)

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/", response_model=LiveModelResponse)
def live_model() -> LiveModelResponse:
    """What is trading right now: provenance, gates, and the kill-switch state.

    `has_model=false` means nothing is promoted, which is a real state and not an
    error — a fresh install has no model until a candidate clears the gates.
    `unrecorded_artifact=true` means an artifact exists with no ledger entry: it
    was installed outside the gates and should be treated with suspicion.
    """
    return get_live_model()


@router.get("/promotions", response_model=PromotionHistoryResponse)
def promotions(limit: int = Query(50, ge=1, le=200)) -> PromotionHistoryResponse:
    """Every candidate evaluation, newest first, rejections included.

    `trials_to_date` is the figure the deflated Sharpe ratio discounts by, which
    is why the rejections are served rather than filtered out.
    """
    return get_promotion_history(limit=limit)


@router.get("/features", response_model=FeatureImportanceResponse)
def feature_importance(
    head: str = Query('price', pattern=r'^[a-z_]{1,32}$'),
) -> FeatureImportanceResponse:
    """Split gains from the promoted model's booster, normalised to sum to one.

    Returns an empty list with `unavailable_reason` when there is no model or it
    cannot be loaded — never a substitute table. Invented explainability looks
    exactly like the real thing, which makes it worse than none.
    """
    return get_feature_importance(head=head)
