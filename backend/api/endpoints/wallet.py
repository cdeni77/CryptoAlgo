"""Wallet routes. The logic lives in `controllers.wallet`."""

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from controllers.wallet import build_wallet
from database import get_db

router = APIRouter(prefix="/wallet", tags=["wallet"])


@router.get("/")
def get_wallet(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Paper balance, realised and unrealised PnL, and every external holding.

    Each external section carries its own `status`, because a wallet provider
    being unreachable should degrade that section rather than fail the request.
    """
    return build_wallet(db)
