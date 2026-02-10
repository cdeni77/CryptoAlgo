from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from models.trade import TradeResponse
from controllers.trade import (
    get_trade, get_all_trades, get_open_trades, get_closed_trades,
    get_trades_by_coin, get_recent_trades
)
from database import get_db 

router = APIRouter(prefix="/trades", tags=["trades"])

@router.get("/", response_model=List[TradeResponse])
def list_trades(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return get_all_trades(db, skip=skip, limit=limit)

@router.get("/recent", response_model=List[TradeResponse])
def recent_trades(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    return get_recent_trades(db, limit=limit)

@router.get("/open", response_model=List[TradeResponse])
def open_trades(db: Session = Depends(get_db)):
    return get_open_trades(db)

@router.get("/closed", response_model=List[TradeResponse])
def closed_trades(db: Session = Depends(get_db)):
    return get_closed_trades(db)

@router.get("/coin/{coin}", response_model=List[TradeResponse])
def trades_by_coin(coin: str, db: Session = Depends(get_db)):
    trades = get_trades_by_coin(db, coin)
    if not trades:
        raise HTTPException(status_code=404, detail=f"No trades found for coin: {coin}")
    return trades

@router.get("/{trade_id}", response_model=TradeResponse)
def get_single_trade(trade_id: int, db: Session = Depends(get_db)):
    trade = get_trade(db, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade