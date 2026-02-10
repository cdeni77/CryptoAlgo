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
    
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="Coinbase API credentials not configured"
        )
    
    return RESTClient(api_key=api_key, api_secret=api_secret)

COINBASE_PRODUCTS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD"
}

@router.get("/prices", response_model=Dict[str, dict])
def get_current_prices():
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


@router.get("/history/{symbol}", response_model=List[Dict])
def get_historical_prices(
    symbol: str = Path(..., enum=["BTC", "ETH", "SOL"]),
    days: int | None = Query(None, ge=1, le=730, description="Number of days (for ranges ≥1d)"),
    hours: int | None = Query(None, ge=1, le=24, description="Number of hours (for 1h range)"),
    granularity: str = Query(
        None,  # will be set automatically if not provided
        enum=["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "SIX_HOUR", "ONE_DAY"]
    )
):
    if symbol not in COINBASE_PRODUCTS:
        raise HTTPException(400, "Invalid symbol")

    if days is None and hours is None:
        raise HTTPException(400, "Either 'days' or 'hours' must be provided")

    product_id = COINBASE_PRODUCTS[symbol]
    client = get_coinbase_client()

    # Determine total time window and default granularity
    if hours is not None:
        total_seconds = hours * 3600
        # Default to 1-minute for short ranges
        granularity = granularity or "ONE_MINUTE"
    else:
        total_seconds = days * 24 * 3600
        # Choose sensible default granularity based on days
        if days <= 1:
            granularity = granularity or "FIFTEEN_MINUTE"
        elif days <= 7:
            granularity = granularity or "ONE_HOUR"
        else:
            granularity = granularity or "ONE_DAY"

    granularity_map = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "ONE_HOUR": 3600,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400
    }
    interval_sec = granularity_map.get(granularity, 900)

    MAX_CANDLES_PER_REQUEST = 280

    all_history = []
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_limit_ts = end_ts - total_seconds

    while end_ts > start_limit_ts:
        start_ts = max(end_ts - (MAX_CANDLES_PER_REQUEST * interval_sec), start_limit_ts)

        try:
            response = client.get_public_candles(
                product_id=product_id,
                start=str(start_ts),
                end=str(end_ts),
                granularity=granularity,
                limit=MAX_CANDLES_PER_REQUEST
            )

            chunk = [{
                "timestamp": datetime.fromtimestamp(int(candle.start), tz=timezone.utc).isoformat(),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume)
            } for candle in response.candles]

            all_history.extend(chunk)

            if len(chunk) == 0:
                break

            end_ts = start_ts - 1

        except Exception as e:
            print(f"Error fetching candles {symbol} {granularity} {start_ts}-{end_ts}: {e}")
            break

    # Sort + deduplicate
    all_history.sort(key=lambda x: x["timestamp"])
    seen = set()
    unique = [entry for entry in all_history if not (entry["timestamp"] in seen or seen.add(entry["timestamp"]))]

    return unique