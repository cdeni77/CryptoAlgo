import os
from typing import Any, Dict, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from coinbase.rest import RESTClient
from database import get_db
from models.trade import Trade, TradeSide, TradeStatus
from models.wallet import Wallet

router = APIRouter(prefix="/wallet", tags=["wallet"])

# Coinbase client (credentials optional; endpoint degrades gracefully without them)
api_key = os.getenv("COINBASE_API_KEY")
api_secret = os.getenv("COINBASE_API_SECRET")
client = RESTClient(api_key=api_key, api_secret=api_secret)


def _to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "to_dict"):
        try:
            return payload.to_dict()
        except Exception:
            return {}
    if hasattr(payload, "__dict__"):
        data = dict(payload.__dict__)
        data.pop("raw_response", None)
        return data
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_current_price(spot_id: str) -> Optional[float]:
    try:
        product = client.get_product(product_id=spot_id)
        product_dict = _to_dict(product)
        return _safe_float(product_dict.get("price") or product_dict.get("mid_market_price"))
    except Exception:
        return None


def get_coinbase_spot_portfolio() -> Dict[str, Any]:
    """Aggregate spot holdings from Advanced Trade accounts."""
    try:
        accounts_response = client.get_accounts(limit=250)
        accounts = _to_dict(accounts_response).get("accounts", [])
        total_usd = 0.0

        for account in accounts:
            account_d = _to_dict(account)
            currency = account_d.get("currency")

            available_balance = _to_dict(account_d.get("available_balance")).get("value")
            hold_balance = _to_dict(account_d.get("hold")).get("value")
            if available_balance is None:
                available_balance = account_d.get("available_balance")
            if hold_balance is None:
                hold_balance = account_d.get("hold")

            amount = (_safe_float(available_balance) or 0.0) + (_safe_float(hold_balance) or 0.0)
            if amount <= 0:
                continue

            if currency in ("USD", "USDC", "USDT"):
                total_usd += amount
                continue

            if not currency:
                continue

            price = get_current_price(f"{currency}-USD")
            if price is not None:
                total_usd += amount * price

        return {
            "value_usd": round(total_usd, 2),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "value_usd": None,
            "status": "error",
            "error": str(exc),
        }


def get_coinbase_perps_portfolio() -> Dict[str, Any]:
    """Fetch perps/INTX portfolio valuation from Coinbase Advanced Trade."""
    try:
        balances_response = client.get_perps_portfolio_balances()
        balances = _to_dict(balances_response)

        candidates = [
            balances.get("total_balance_usd"),
            balances.get("portfolio_balance_usd"),
            balances.get("equity_usd"),
            balances.get("cash_equity_usd"),
            _to_dict(balances.get("portfolio_balance")).get("value"),
            _to_dict(balances.get("equity")).get("value"),
        ]

        value_usd = next((v for v in (_safe_float(c) for c in candidates) if v is not None), None)

        if value_usd is None:
            summary_response = client.get_perps_portfolio_summary()
            summary = _to_dict(summary_response)
            value_usd = _safe_float(summary.get("equity_usd") or summary.get("portfolio_value_usd"))

        return {
            "value_usd": round(value_usd, 2) if value_usd is not None else None,
            "status": "ok" if value_usd is not None else "unavailable",
        }
    except Exception as exc:
        return {
            "value_usd": None,
            "status": "error",
            "error": str(exc),
        }


def _upsert_wallet_balance(db: Session, balance: float) -> None:
    wallet = db.query(Wallet).order_by(Wallet.id.desc()).first()
    if wallet:
        wallet.balance = balance
    else:
        wallet = Wallet(balance=balance)
        db.add(wallet)
    db.commit()


@router.get("/")
def get_wallet(db: Session = Depends(get_db)):
    # Realized PNL: sum net_pnl from closed trades
    realized_pnl = db.query(sa.func.sum(Trade.net_pnl)).filter(Trade.status == TradeStatus.CLOSED).scalar() or 0.0

    # Unrealized PNL: calculate from open trades
    open_trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()
    unrealized_pnl = 0.0
    for trade in open_trades:
        current_price = get_current_price(trade.coin)
        if current_price is None:
            continue
        multiplier = 1 if trade.side == TradeSide.LONG else -1
        pnl = (current_price - trade.entry_price) * multiplier * trade.contracts
        unrealized_pnl += pnl

    # Pull live Coinbase portfolios (spot + advanced perps) and persist aggregate
    spot = get_coinbase_spot_portfolio()
    perps = get_coinbase_perps_portfolio()
    spot_usd = _safe_float(spot.get("value_usd"))
    perps_usd = _safe_float(perps.get("value_usd"))
    coinbase_total = (spot_usd or 0.0) + (perps_usd or 0.0)

    if spot_usd is not None or perps_usd is not None:
        _upsert_wallet_balance(db, coinbase_total)

    # Current balance from latest wallet entry (DB fallback if Coinbase unavailable)
    wallet = db.query(Wallet).order_by(Wallet.id.desc()).first()
    balance = wallet.balance if wallet else 10000.0

    total_pnl = realized_pnl + unrealized_pnl

    return {
        "balance": balance,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "coinbase": {
            "spot": spot,
            "perps": perps,
            "total_value_usd": round(coinbase_total, 2) if (spot_usd is not None or perps_usd is not None) else None,
        },
    }
