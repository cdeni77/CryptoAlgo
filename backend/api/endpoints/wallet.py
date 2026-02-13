import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from coinbase.rest import RESTClient
from database import get_db
from models.trade import PaperEquityCurve, Trade, TradeSide, TradeStatus

router = APIRouter(prefix="/wallet", tags=["wallet"])

# Coinbase client (credentials optional; endpoint degrades gracefully without them)
api_key = os.getenv("COINBASE_API_KEY")
api_secret = os.getenv("COINBASE_API_SECRET")
client = RESTClient(api_key=api_key, api_secret=api_secret)

STABLE_COINS = {"USD", "USDC", "USDT"}
ONDO_ETH_CONTRACT = "0xfAbA6f8e4a5E8Ab82F62fe7C39859FA577269BE3"
ONDO_SOL_MINT = "A3eMEJQqN3EAx2FQwPDhJGFH3M9x4W8M7mWUPR8iY5Wg"


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


def _fetch_json(url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if payload is None:
            req = request.Request(url, headers={"accept": "application/json"})
        else:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                url,
                data=body,
                headers={"accept": "application/json", "content-type": "application/json"},
                method="POST",
            )

        with request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
    except (error.HTTPError, error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return {}


def get_current_price(spot_id: str) -> Optional[float]:
    try:
        product = client.get_product(product_id=spot_id)
        product_dict = _to_dict(product)
        return _safe_float(product_dict.get("price") or product_dict.get("mid_market_price"))
    except Exception:
        return None


def get_ledger_wallets_from_env() -> Dict[str, Any]:
    """Read optional ledger wallet addresses from environment only (never committed)."""
    entries: List[Dict[str, str]] = []

    raw_json = os.getenv("LEDGER_WALLETS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    coin = str(item.get("coin", "")).strip().upper()
                    address = str(item.get("address", "")).strip()
                    if coin and address:
                        entries.append({"coin": coin, "address": address})
        except Exception:
            pass

    deduped: List[Dict[str, str]] = []
    seen = set()
    for entry in entries:
        key = (entry["coin"], entry["address"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    return {
        "status": "configured" if deduped else "unconfigured",
        "entries": deduped,
    }


def _get_btc_balance(address: str) -> Optional[float]:
    payload = _fetch_json(f"https://blockstream.info/api/address/{address}")
    chain_stats = _to_dict(payload.get("chain_stats"))
    funded = _safe_float(chain_stats.get("funded_txo_sum")) or 0.0
    spent = _safe_float(chain_stats.get("spent_txo_sum")) or 0.0
    return (funded - spent) / 1e8


def _get_eth_balance(address: str) -> Optional[float]:
    query = parse.urlencode({"module": "account", "action": "balance", "address": address})
    payload = _fetch_json(f"https://eth.blockscout.com/api?{query}")
    wei = _safe_float(payload.get("result"))
    return wei / 1e18 if wei is not None else None


def _get_erc20_balance(address: str, contract_address: str, decimals: int) -> Optional[float]:
    query = parse.urlencode(
        {
            "module": "account",
            "action": "tokenbalance",
            "contractaddress": contract_address,
            "address": address,
        }
    )
    payload = _fetch_json(f"https://eth.blockscout.com/api?{query}")
    raw_balance = _safe_float(payload.get("result"))
    if raw_balance is None:
        return None
    return raw_balance / (10 ** decimals)


def _get_sol_balance(address: str) -> Optional[float]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [address],
    }
    response = _fetch_json("https://api.mainnet-beta.solana.com", payload=payload)
    value = _safe_float(_to_dict(response.get("result")).get("value"))
    return value / 1e9 if value is not None else None


def _get_spl_token_balance(address: str, mint: str, decimals: int = 9) -> Optional[float]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [address, {"mint": mint}, {"encoding": "jsonParsed"}],
    }
    response = _fetch_json("https://api.mainnet-beta.solana.com", payload=payload)
    accounts = _to_dict(response.get("result")).get("value", [])
    total = 0.0
    seen = False
    for account in accounts:
        token_amount = (
            _to_dict(_to_dict(_to_dict(account).get("account")).get("data"))
            .get("parsed", {})
            .get("info", {})
            .get("tokenAmount", {})
        )
        amount = _safe_float(_to_dict(token_amount).get("amount"))
        if amount is None:
            continue
        total += amount / (10 ** decimals)
        seen = True
    return total if seen else None


def _get_asset_price_usd(asset: str, cache: Dict[str, Optional[float]]) -> Optional[float]:
    if asset in cache:
        return cache[asset]

    if asset in STABLE_COINS:
        cache[asset] = 1.0
        return 1.0

    price = get_current_price(f"{asset}-USD")
    if price is None:
        coingecko_map = {"ONDO": "ondo-finance"}
        coin_id = coingecko_map.get(asset)
        if coin_id:
            payload = _fetch_json(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            )
            price = _safe_float(_to_dict(payload.get(coin_id)).get("usd"))

    cache[asset] = price
    return price


def get_ledger_portfolio() -> Dict[str, Any]:
    ledger = get_ledger_wallets_from_env()
    entries = ledger.get("entries", [])
    if not entries:
        return {"status": ledger.get("status", "unconfigured"), "entries": [], "assets": [], "value_usd": None}

    detailed_entries: List[Dict[str, Any]] = []
    asset_totals: Dict[str, float] = {}
    price_cache: Dict[str, Optional[float]] = {}

    for entry in entries:
        coin = str(entry.get("coin", "")).upper()
        address = str(entry.get("address", ""))
        amount: Optional[float] = None

        if coin == "BTC":
            amount = _get_btc_balance(address)
        elif coin == "ETH":
            amount = _get_eth_balance(address)
        elif coin == "SOL":
            amount = _get_sol_balance(address)
        elif coin == "ONDO":
            amount = _get_erc20_balance(address, ONDO_ETH_CONTRACT, 18)
            if amount is None:
                amount = _get_spl_token_balance(address, ONDO_SOL_MINT, 9)

        price_usd = _get_asset_price_usd(coin, price_cache) if amount is not None else None
        value_usd = (amount * price_usd) if (amount is not None and price_usd is not None) else None

        if amount is not None and amount > 0:
            asset_totals[coin] = asset_totals.get(coin, 0.0) + amount

        detailed_entries.append(
            {
                "coin": coin,
                "address": address,
                "amount": round(amount, 8) if amount is not None else None,
                "price_usd": round(price_usd, 6) if price_usd is not None else None,
                "value_usd": round(value_usd, 2) if value_usd is not None else None,
            }
        )

    assets: List[Dict[str, Any]] = []
    for asset, amount in asset_totals.items():
        price = _get_asset_price_usd(asset, price_cache)
        value = (amount * price) if price is not None else None
        assets.append(
            {
                "asset": asset,
                "amount": round(amount, 8),
                "price_usd": round(price, 6) if price is not None else None,
                "value_usd": round(value, 2) if value is not None else None,
            }
        )

    assets.sort(key=lambda item: item.get("value_usd") or 0.0, reverse=True)
    total_usd = sum((_safe_float(asset.get("value_usd")) or 0.0) for asset in assets)
    ok_entries = [entry for entry in detailed_entries if entry.get("amount") is not None]

    return {
        "status": "ok" if ok_entries else "unavailable",
        "entries": detailed_entries,
        "assets": assets,
        "value_usd": round(total_usd, 2),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_coinbase_spot_portfolio() -> Dict[str, Any]:
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
                    if asset in STABLE_COINS:
                        value_usd = amount
                    else:
                        px = get_current_price(f"{asset}-USD")
                        value_usd = amount * px if px is not None else None

                existing = asset_map.get(asset)
                if not existing:
                    existing = {"asset": asset, "amount": 0.0, "value_usd": 0.0}
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

                if currency in STABLE_COINS:
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
                        "price_usd": round(price_usd, 6),
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
    realized_pnl = db.query(sa.func.sum(Trade.net_pnl)).filter(Trade.status == TradeStatus.CLOSED).scalar() or 0.0

    open_trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()
    unrealized_pnl = 0.0
    for trade in open_trades:
        current_price = get_current_price(trade.coin)
        if current_price is None:
            continue
        multiplier = 1 if trade.side == TradeSide.LONG else -1
        pnl = (current_price - trade.entry_price) * multiplier * trade.contracts
        unrealized_pnl += pnl

    spot = get_coinbase_spot_portfolio()
    perps = get_coinbase_perps_portfolio()
    ledger = get_ledger_portfolio()

    spot_usd = _safe_float(spot.get("value_usd"))
    perps_usd = _safe_float(perps.get("value_usd"))
    ledger_usd = _safe_float(ledger.get("value_usd"))
    portfolio_total = (spot_usd or 0.0) + (perps_usd or 0.0) + (ledger_usd or 0.0)

    paper_balance = 10000.0

    equity_points = (
        db.query(PaperEquityCurve)
        .order_by(sa.desc(PaperEquityCurve.timestamp))
        .limit(200)
        .all()
    )
    equity_points.reverse()
    portfolio_history = [
        {
            "timestamp": point.timestamp.isoformat() if point.timestamp else None,
            "paper_equity_usd": round(point.equity, 2),
            "external_usd": round(portfolio_total, 2),
            "total_value_usd": round(point.equity + portfolio_total, 2),
        }
        for point in equity_points
        if point.timestamp is not None
    ]

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
            "ledger": {
                "value_usd": ledger.get("value_usd"),
                "status": ledger.get("status"),
                "address_count": len(ledger.get("entries", [])),
            },
        },
        "coinbase": {
            "spot": spot,
            "perps": perps,
            "total_value_usd": round(portfolio_total, 2)
            if (spot_usd is not None or perps_usd is not None or ledger_usd is not None)
            else None,
        },
        "ledger": ledger,
        "portfolio_history": portfolio_history,
    }
