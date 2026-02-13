import os

from fastapi import APIRouter, HTTPException, Path, Query
from typing import Dict, List
from datetime import datetime, timezone
from coinbase.rest import RESTClient
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/coins", tags=["coins"])

def get_coinbase_client():
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")

    # Public market-data endpoints do not require auth.
    # If API keys are available we still pass them through.
    if api_key and api_secret:
        return RESTClient(api_key=api_key, api_secret=api_secret)

    return RESTClient()

# Spot products on Coinbase
COINBASE_PRODUCTS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
}

# CDE (Contract for Difference / Perpetual) product mappings
# These are the nano perpetual futures on Coinbase CDE
CDE_PRODUCTS = {
    "BTC": {
        "symbol": "BIP-20DEC30-CDE",
        "code": "BIP",
        "units_per_contract": 0.01,
        "approx_contract_value": 675.70,
        "fee_pct": 0.00100,  # 0.100% per side
    },
    "ETH": {
        "symbol": "ETP-20DEC30-CDE",
        "code": "ETP",
        "units_per_contract": 0.1,
        "approx_contract_value": 196.50,
        "fee_pct": 0.00102,  # 0.102% per side
    },
    "SOL": {
        "symbol": "SLP-20DEC30-CDE",
        "code": "SLP",
        "units_per_contract": 5,
        "approx_contract_value": 400.90,
        "fee_pct": 0.00100,  # 0.100% per side
    },
    "XRP": {
        "symbol": "XPP-20DEC30-CDE",
        "code": "XPP",
        "units_per_contract": 500,
        "approx_contract_value": 690.45,
        "fee_pct": 0.00100,  # 0.100% per side
    },
    "DOGE": {
        "symbol": "DOP-20DEC30-CDE",
        "code": "DOP",
        "units_per_contract": 5000,
        "approx_contract_value": 458.35,
        "fee_pct": 0.00100,  # 0.100% per side
    },
}

VALID_SYMBOLS = list(COINBASE_PRODUCTS.keys())


@router.get("/prices", response_model=Dict[str, dict])
def get_current_prices():
    """Get current spot prices for all tracked coins."""
    client = get_coinbase_client()
    result = {}

    try:
        response = client.get_public_products()
        product_list = response.products

        if not product_list:
            raise ValueError("No products returned from Coinbase")

        for symbol, product_id in COINBASE_PRODUCTS.items():
            found = False
            for prod in product_list:
                prod_id = prod.product_id 
                if prod_id == product_id:
                    price_str = prod.price
                    change_str = prod.price_percentage_change_24h

                    result[symbol] = {
                        "price": float(price_str) if price_str else None,
                        "change24h": float(change_str) if change_str else None,
                    }
                    found = True
                    break

            if not found:
                result[symbol] = {"price": None, "change24h": None}

        if all(r["price"] is None for r in result.values()):
            raise HTTPException(status_code=503, detail="Failed to fetch prices from Coinbase")

        return result

    except Exception as e:
        print(f"Error fetching prices: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/cde-specs", response_model=Dict[str, dict])
def get_cde_specs():
    """Return CDE contract specifications for all coins."""
    return CDE_PRODUCTS


@router.get("/history/{symbol}", response_model=List[Dict])
def get_historical_prices(
    symbol: str = Path(..., enum=VALID_SYMBOLS),
    days: int | None = Query(None, ge=1, le=730, description="Number of days (for ranges ≥1d)"),
    hours: int | None = Query(None, ge=1, le=24, description="Number of hours (for 1h range)"),
    granularity: str = Query(
        None,
        enum=["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "SIX_HOUR", "ONE_DAY"]
    )
):
    """Get historical spot OHLCV data from Coinbase."""
    if symbol not in COINBASE_PRODUCTS:
        raise HTTPException(400, "Invalid symbol")

    if days is None and hours is None:
        raise HTTPException(400, "Either 'days' or 'hours' must be provided")

    product_id = COINBASE_PRODUCTS[symbol]
    client = get_coinbase_client()

    # Determine total time window and default granularity
    if hours is not None:
        total_seconds = hours * 3600
        granularity = granularity or "ONE_MINUTE"
    else:
        total_seconds = days * 24 * 3600
        if days <= 1:
            granularity = granularity or "FIFTEEN_MINUTE"
        elif days <= 7:
            granularity = granularity or "ONE_HOUR"
        else:
            granularity = granularity or "ONE_DAY"

    # Map granularity to seconds for chunking
    gran_seconds = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "ONE_HOUR": 3600,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400,
    }
    interval = gran_seconds.get(granularity, 3600)

    # Coinbase returns max 300 candles per request; chunk if needed
    max_candles = 300
    chunk_seconds = max_candles * interval

    now = int(datetime.now(timezone.utc).timestamp())
    start = now - total_seconds

    all_candles = []
    current_start = start

    try:
        while current_start < now:
            current_end = min(current_start + chunk_seconds, now)

            response = client.get_public_candles(
                product_id=product_id,
                start=str(current_start),
                end=str(current_end),
                granularity=granularity,
            )

            candles = response.candles if hasattr(response, 'candles') else response.get("candles", [])

            for c in candles:
                ts = int(c.start) if hasattr(c, 'start') else int(c["start"])
                all_candles.append({
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "open": float(c.open if hasattr(c, 'open') else c["open"]),
                    "high": float(c.high if hasattr(c, 'high') else c["high"]),
                    "low": float(c.low if hasattr(c, 'low') else c["low"]),
                    "close": float(c.close if hasattr(c, 'close') else c["close"]),
                    "volume": float(c.volume if hasattr(c, 'volume') else c["volume"]),
                })

            current_start = current_end

        all_candles.sort(key=lambda x: x["timestamp"])

        # Deduplicate
        seen = set()
        deduped = []
        for candle in all_candles:
            if candle["timestamp"] not in seen:
                seen.add(candle["timestamp"])
                deduped.append(candle)

        return deduped

    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        raise HTTPException(status_code=503, detail=str(e))
