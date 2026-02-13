import os
from typing import Any, Dict, List, Optional

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
    """Aggregate spot holdings from Advanced Trade portfolios/accounts.

    Prefers portfolio breakdown data (captures staked assets in `total_balance_crypto`
    and `total_balance_fiat` where available). Falls back to account balances.
    """
    try:
        asset_map: Dict[str, Dict[str, Any]] = {}

        portfolios_response = client.get_portfolios()
        portfolios = _to_dict(portfolios_response).get("portfolios", [])
        for raw_portfolio in portfolios:
            portfolio = _to_dict(raw_portfolio)
            portfolio_uuid = portfolio.get("uuid")
            if not portfolio_uuid:
                continue
            breakdown = _to_dict(client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)).get("breakdown", {})
            for raw_position in breakdown.get("spot_positions", []):
                position = _to_dict(raw_position)
                asset = position.get("asset")
                if not asset:
                    continue

                amount = _safe_float(position.get("total_balance_crypto")) or 0.0
                value_usd = _safe_float(position.get("total_balance_fiat"))
                if value_usd is None and amount > 0:
                    if asset in ("USD", "USDC", "USDT"):
                        value_usd = amount
                    else:
                        px = get_current_price(f"{asset}-USD")
                        value_usd = amount * px if px is not None else None

                existing = asset_map.get(asset)
                if not existing:
                    existing = {"asset": asset, "amount": 0.0, "value_usd": 0.0, "price_usd": None}
                    asset_map[asset] = existing

                existing["amount"] += amount
                if value_usd is not None:
                    existing["value_usd"] += value_usd

        assets: List[Dict[str, Any]] = []
        for item in asset_map.values():
            amount = item["amount"]
            value_usd = item["value_usd"]
            price_usd = (value_usd / amount) if amount > 0 else None
            assets.append(
                {
                    "asset": item["asset"],
                    "amount": round(amount, 8),
                    "price_usd": round(price_usd, 6) if price_usd is not None else None,
                    "value_usd": round(value_usd, 2),
                }
            )

        # Fallback to account balances when no portfolio breakdown positions are available.
        if not assets:
            accounts_response = client.get_accounts(limit=250)
            accounts = _to_dict(accounts_response).get("accounts", [])
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
                if amount <= 0 or not currency:
                    continue

                if currency in ("USD", "USDC", "USDT"):
                    value_usd = amount
                    price_usd = 1.0
                else:
                    price_usd = get_current_price(f"{currency}-USD")
                    if price_usd is None:
                        continue
                    value_usd = amount * price_usd

                assets.append(
                    {
                        "asset": currency,
                        "amount": round(amount, 8),
                        "price_usd": round(price_usd, 6) if price_usd is not None else None,
                        "value_usd": round(value_usd, 2),
                    }
                )

        total_usd = sum((_safe_float(asset.get("value_usd")) or 0.0) for asset in assets)
        assets.sort(key=lambda item: item.get("value_usd") or 0.0, reverse=True)

        return {
            "value_usd": round(total_usd, 2),
            "assets": assets,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "value_usd": None,
            "assets": [],
            "status": "error",
            "error": str(exc),
        }


def get_coinbase_perps_portfolio() -> Dict[str, Any]:
    """Fetch perps/INTX portfolio valuation from Coinbase Advanced Trade."""
    try:
        balances_response = client.get_perps_portfolio_balances()
        balances = _to_dict(balances_response)
        positions: List[Dict[str, Any]] = []

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

        if hasattr(client, "get_perps_positions"):
            try:
                positions_response = client.get_perps_positions()
                for raw_position in _to_dict(positions_response).get("positions", []):
                    pos = _to_dict(raw_position)
                    symbol = pos.get("product_id") or pos.get("symbol")
                    contracts = _safe_float(pos.get("number_of_contracts") or pos.get("contracts"))
                    mark_price = _safe_float(pos.get("mark_price"))
                    notional = _safe_float(pos.get("notional_value_usd") or pos.get("notional_usd"))
                    unrealized_pnl = _safe_float(pos.get("unrealized_pnl") or pos.get("unrealized_pnl_usd"))
                    if symbol and (contracts is not None or notional is not None):
                        positions.append(
                            {
                                "symbol": symbol,
                                "contracts": round(contracts, 8) if contracts is not None else None,
                                "mark_price": round(mark_price, 6) if mark_price is not None else None,
                                "notional_usd": round(notional, 2) if notional is not None else None,
                                "unrealized_pnl_usd": round(unrealized_pnl, 2)
                                if unrealized_pnl is not None
                                else None,
                            }
                        )
            except Exception:
                positions = []

        positions.sort(key=lambda item: abs(item.get("notional_usd") or 0.0), reverse=True)

        return {
            "value_usd": round(value_usd, 2) if value_usd is not None else None,
            "positions": positions,
            "status": "ok" if value_usd is not None else "unavailable",
        }
    except Exception as exc:
        return {
            "value_usd": None,
            "positions": [],
            "status": "error",
            "error": str(exc),
        }


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

    # Pull live Coinbase portfolios (spot + advanced perps)
    spot = get_coinbase_spot_portfolio()
    perps = get_coinbase_perps_portfolio()
    spot_usd = _safe_float(spot.get("value_usd"))
    perps_usd = _safe_float(perps.get("value_usd"))
    coinbase_total = (spot_usd or 0.0) + (perps_usd or 0.0)

    # Paper trading wallet remains fixed at starting balance until paper wallet accounting is enabled.
    paper_balance = 10000.0

    total_pnl = realized_pnl + unrealized_pnl
    return {
        "balance": paper_balance,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "wallets": {
            "paper_trading": {
                "value_usd": round(paper_balance, 2),
                "cash_usd": round(paper_balance, 2),
                "unrealized_pnl": 0.0,
                "status": "ok",
            },
            "coinbase_spot": {"value_usd": spot.get("value_usd"), "status": spot.get("status")},
            "coinbase_perps": {"value_usd": perps.get("value_usd"), "status": perps.get("status")},
        },
        "coinbase": {
            "spot": spot,
            "perps": perps,
            "total_value_usd": round(coinbase_total, 2) if (spot_usd is not None or perps_usd is not None) else None,
        },
    }
