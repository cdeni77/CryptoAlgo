import sqlalchemy as sa
import os

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.trade import Trade, TradeStatus, TradeSide
from models.wallet import Wallet
from database import get_db 
from coinbase.rest import RESTClient

router = APIRouter(prefix="/wallet", tags=["wallet"])

# Coinbase client for current prices (assume env vars set)
api_key = os.getenv("COINBASE_API_KEY")
api_secret = os.getenv("COINBASE_API_SECRET")
client = RESTClient(api_key=api_key, api_secret=api_secret)

def get_current_price(spot_id):
    try:
        product = client.get_product(product_id=spot_id)
        return float(product['price'])
    except Exception:
        return None

@router.get("/")
def get_wallet(db: Session = Depends(get_db)):
    # Get current balance from latest wallet entry
    wallet = db.query(Wallet).order_by(Wallet.id.desc()).first()
    balance = wallet.balance if wallet else 10000.0

    # Realized PNL: sum net_pnl from closed trades
    realized_pnl = db.query(sa.func.sum(Trade.net_pnl)).filter(Trade.status == TradeStatus.CLOSED).scalar() or 0.0

    # Unrealized PNL: calculate from open trades
    open_trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()
    unrealized_pnl = 0.0
    for trade in open_trades:
        current_price = get_current_price(trade.coin)
        if current_price is None:
            continue  # Skip if price fetch fails
        multiplier = 1 if trade.side == TradeSide.LONG else -1
        pnl = (current_price - trade.entry_price) * multiplier * trade.contracts  # Adjust with CONTRACT_SIZES if needed
        unrealized_pnl += pnl

    total_pnl = realized_pnl + unrealized_pnl

    return {
        "balance": balance,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl
    }