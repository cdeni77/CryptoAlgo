#!/usr/bin/env python3
"""
Unified Data Pipeline for Coinbase Perps Trading Bot

This script combines:
1. Dynamic resolution of Coinbase "Smart Perp" product IDs
2. OHLCV backfill from Coinbase (perps, or spot under --spot-universe)
3. Funding rate and open interest snapshots from the product endpoint
4. Real-time data collection via WebSocket

Usage:
    # Full backfill + real-time collection
    python run_pipeline.py --backfill-days 365

    # Backfill only (no real-time)
    python run_pipeline.py --backfill-days 365 --backfill-only

    # Skip OHLCV, only fetch funding rates
    python run_pipeline.py --funding-only --backfill-days 365

    # Real-time only (skip all backfill)
    python run_pipeline.py --skip-backfill

    # Include open interest data
"""

import argparse
import asyncio
import re
import logging
import os
import sys

from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.coinbase_connector import CoinbaseRESTClient
from data_collection.pipeline import create_pipeline, PipelineConfig, ensure_naive_utc
from data_collection.timeutil import epoch_millis_to_naive_utc, utc_now
from data_collection.models import OHLCVBar, TickerUpdate, FundingRate, OpenInterest
from data_collection.ingest import Ingestor
from data_collection.storage import SQLiteDatabase

# Configuration

# Map your desired Assets to Coinbase "Smart Perp" Codes
ASSET_TO_CODE_MAP = {
    "BTC": "BIP",
    "ETH": "ETP",
    "SOL": "SLP",
    "XRP": "XPP",
    "DOGE": "DOP",
    # Batch 2 — verified 20DEC30 CDE codes from Coinbase API
    "AVAX": "AVP",
    "ADA": "ADP",
    "LINK": "LNP",
    "LTC": "LCP",
    # Batch 3 — new 20DEC30-CDE additions 2026-04-03
    "NEAR": "NER",
    "SUI": "SUP",
    "BCH": "BCP",
    "XLM": "XLP",
    "DOT": "POP",
    "SHIB": "SHP",
    "PEPE": "PEP",
    # Listed after the rest, so they carry less history than the panel.
    "HYPE": "HYP",
    "ONDO": "OND",
}

# Spot spellings: an instrument quoted directly in fiat or a stablecoin. Funding
# is a perpetual-contract cash flow, so these never have one.
SPOT_QUOTES = re.compile(r'-(USD|USDC|USDT)$', re.IGNORECASE)


def perpetual_symbols(symbols: Iterable[str]) -> List[str]:
    """The symbols that can carry funding at all.

    One predicate, two callers: the funding product map skips spot so it cannot
    file a perp's rate under a spot key, and the scrape's exit code demands
    funding only when something in the run was supposed to have it. Those two had
    to agree — a spot-only run collecting zero funding is a correct run, and
    treating it as a failure aborted the orchestrator's reference-venue cycle.
    """
    return [symbol for symbol in symbols if not SPOT_QUOTES.search(symbol)]

# Hourly only. The daily bars were collected and then read by nothing: the
# research store's `bars` dataset has no timeframe column, and `from_sqlite`
# filters `WHERE timeframe = ?` defaulting to '1h' (as does
# migrate_to_research_store), so daily rows never reached the panel. Nothing in
# backend/api touches the ohlcv table either — the dashboard's '1d' is a *range*
# (`days=1`), not a candle granularity. So they cost 400 days x 18 contracts of
# requests on the first cycle to populate rows nothing consumes, and sat in SQLite
# where a `--timeframe 1d` migration could silently mix granularities into one
# venue/symbol/event_time key space. Ask for them explicitly if you ever need them;
# hourly resamples to daily anyway.
DEFAULT_TIMEFRAMES = ["1h"]
DEFAULT_SYMBOLS = [
    "BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP", "DOGE-PERP",
    "AVP-20DEC30-CDE", "ADP-20DEC30-CDE", "LNP-20DEC30-CDE", "LCP-20DEC30-CDE",
    # Batch 3
    "NER-20DEC30-CDE", "SUP-20DEC30-CDE", "BCP-20DEC30-CDE",
    "XLP-20DEC30-CDE", "POP-20DEC30-CDE", "SHP-20DEC30-CDE", "PEP-20DEC30-CDE",
    "HYP-20DEC30-CDE", "OND-20DEC30-CDE",
]

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Helper Classes

class BackfillProgress:
    """Track and display backfill progress."""
    
    def __init__(self, total_tasks: int, description: str = "Backfill"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = datetime.now()
        self.description = description
    
    def update(self, symbol: str, timeframe: str = "", bars: int = 0):
        """Update progress (placeholder for UI)."""
        pass
    
    def task_complete(self, symbol: str, timeframe: str = "", count: int = 0):
        """Mark a task as complete."""
        self.completed_tasks += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        pct = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 100
        
        tf_str = f" {timeframe}" if timeframe else ""
        logger.info(
            f"✅ [{self.completed_tasks}/{self.total_tasks}] {symbol}{tf_str}: "
            f"{count} records | {pct:.0f}% complete | {elapsed:.0f}s elapsed"
        )
    
    def summary(self):
        """Print summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"🏁 {self.description} complete: {self.completed_tasks} tasks in {elapsed:.0f}s")


# Callbacks for Real-time Data

_ticker_last_log: Dict[str, datetime] = defaultdict(lambda: datetime.min)
TICKER_LOG_INTERVAL = timedelta(minutes=1)


def on_new_candle(bar: OHLCVBar):
    """Callback for new OHLCV candles."""
    logger.info(f"🕯  NEW CANDLE | {bar.symbol} {bar.timeframe} | C={bar.close:.2f} V={bar.volume:.2f}")


def on_ticker_update(ticker: TickerUpdate):
    """Callback for ticker updates (throttled)."""
    now = utc_now()
    last_logged = _ticker_last_log[ticker.symbol]
    
    if now - last_logged >= TICKER_LOG_INTERVAL:
        logger.info(
            f"💱 TICKER     | {ticker.symbol} | {ticker.price:.2f} | "
            f"B={ticker.best_bid:.2f} A={ticker.best_ask:.2f}"
        )
        _ticker_last_log[ticker.symbol] = now


def on_funding_rate(funding: FundingRate):
    """Callback for funding rate updates."""
    logger.info(f"🏦 FUNDING    | {funding.symbol} | {funding.rate*100:.6f}%")


# Backfill Functions

async def backfill_ohlcv(
    pipeline,
    symbols: List[str],
    timeframes: List[str],
    start_time: datetime,
    end_time: datetime,
):
    """
    OHLCV backfill from Coinbase. There is no longer a second venue.

    This was a hybrid, falling through to CCXT whenever Coinbase returned
    nothing. Every use of that fallback turned out to be a mistake dressed as
    resilience: the "gap" it filled was the span before a contract was listed
    (BIP began 2025-07-18, so a 400-day request legitimately misses 265 days),
    and it filled it with `BTC/USDT:USDT` from another exchange — a different
    quote currency, contract size, funding and participant set, stored under this
    symbol's name.

    Coinbase serves both venues this system reads: the CDE perps under
    `coinbase`, and spot under `coinbase_spot` via `--spot-universe`. Neither
    needs a proxy, and the reference venue is a real Coinbase market rather than
    a geo-blocked stand-in.
    """
    print("\n" + "=" * 70)
    print("📊 OHLCV BACKFILL (Coinbase)")
    print("=" * 70)
    print(f"Period: {start_time.date()} to {end_time.date()}")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print()
    
    progress = BackfillProgress(len(symbols) * len(timeframes), "OHLCV Backfill")
    
    for symbol in symbols:
        for tf in timeframes:
            progress.update(symbol, tf)
            try:
                logger.info(f"   Fetching {symbol} {tf} from Coinbase...")
                await pipeline.backfill(start_time, end_time, [symbol], [tf])

                df = pipeline.get_ohlcv(symbol, tf, start_time, end_time)
                cb_count = len(df) if df is not None and not df.empty else 0

                # An empty result is a fact about the instrument or the request,
                # and it is reported as one. It used to trigger a fetch from
                # another exchange, which is how a symbol with no Coinbase
                # history ended up holding another venue's prices.
                if cb_count == 0:
                    logger.warning(
                        "   %s %s: Coinbase returned no bars for %s -> %s",
                        symbol, tf, start_time.date(), end_time.date(),
                    )
                else:
                    first_bar_time = df.index.min()
                    if hasattr(first_bar_time, 'to_pydatetime'):
                        first_bar_time = first_bar_time.to_pydatetime()
                    gap = first_bar_time - start_time
                    if gap > timedelta(hours=12):
                        # Not a gap: the instrument did not exist yet.
                        logger.info(
                            "   %s history starts %s; the %d day(s) requested "
                            "before that pre-date the contract, so nothing is "
                            "missing.",
                            symbol, first_bar_time.date(), gap.days,
                        )
                
                # 2. Gap Filling Logic
                # Final count
                df = pipeline.get_ohlcv(symbol, tf, start_time, end_time)
                final_count = len(df) if df is not None and not df.empty else 0
                progress.task_complete(symbol, tf, final_count)
                
            except Exception as e:
                logger.error(f"❌ OHLCV backfill failed for {symbol} {tf}: {e}")
                import traceback
                traceback.print_exc()
    
    progress.summary()


def _extract_coin_code(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    if "-" in symbol:
        prefix = symbol.split("-")[0].upper()
        if prefix in ASSET_TO_CODE_MAP.values():
            return prefix
        return ASSET_TO_CODE_MAP.get(prefix)
    return ASSET_TO_CODE_MAP.get(symbol.upper())


async def resolve_coinbase_funding_product_map(
    api_key: str,
    api_secret: str,
    symbols: List[str],
) -> Dict[str, str]:
    """Map requested symbols to active Coinbase CDE product IDs for funding queries."""
    client = CoinbaseRESTClient(api_key, api_secret)
    try:
        target_codes = sorted({c for c in (_extract_coin_code(s) for s in symbols) if c})
        products = await client.get_perpetual_products(target_codes=target_codes)
        code_to_product: Dict[str, str] = {}
        for product in products:
            product_id = product.get("product_id", "")
            if not product_id:
                continue
            code = product_id.split("-")[0].upper()
            code_to_product.setdefault(code, product_id)

        mapping: Dict[str, str] = {}
        for symbol in symbols:
            # Spot pays no funding. `_extract_coin_code('BTC-USD')` resolves to
            # 'BIP', so without this a spot run fetched the *perp's* funding rate
            # and filed it under the spot symbol — the right number under a key
            # that has no such thing, once per settlement.
            if not perpetual_symbols([symbol]):
                continue
            code = _extract_coin_code(symbol)
            if code and code in code_to_product:
                mapping[symbol] = code_to_product[code]
            elif symbol.endswith("-CDE"):
                mapping[symbol] = symbol
        return mapping
    finally:
        await client.close()


async def backfill_funding_rates(
    symbols: List[str],
    start: datetime,
    end: datetime,
    db: SQLiteDatabase,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    venue_label: Optional[str] = None,
):
    """
    Backfill funding rates with Coinbase-native hourly history as primary source.

    Coinbase-native only. There used to be a CCXT fallback writing
    `binance_proxy` rates; funding feeds the target's carry component, so
    another venue's rate is a cash flow this account never receives — and
    `proxy_funding_symbols` is a promotion gate with a threshold of zero, so
    those rows could only ever have blocked a candidate.
    All persisted rates are normalized to decimal/hour.
    """
    print("\n" + "=" * 70)
    print("🏦 FUNDING RATE BACKFILL")
    print("=" * 70)
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Symbols: {symbols}")
    print()

    ingestor = Ingestor(db)

    coinbase_client = CoinbaseRESTClient(api_key, api_secret) if api_key and api_secret else None
    funding_product_map: Dict[str, str] = {}
    if coinbase_client:
        try:
            funding_product_map = await resolve_coinbase_funding_product_map(api_key, api_secret, symbols)
        except Exception as e:
            logger.warning(f"Failed to resolve Coinbase funding products: {e}")

    source_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "coinbase": 0,
        "start": None,
        "end": None,
    })

    try:
        progress = BackfillProgress(len(symbols), "Funding Rate Backfill")
        total_inserted = 0
        oi_inserted = 0

        for symbol in symbols:
            logger.info(f"\n📊 Processing funding rates for {symbol}")
            symbol_inserted = 0

            # Every missing window, not just the ends. Building only
            # (start, MIN) and (MAX, end) left any interior hole bracketed
            # forever — a failed request mid-history stayed a hole and the
            # features were computed straight across it.
            windows = db.find_gaps(
                'funding_rates', symbol, start, end, step_seconds=3600,
            )
            if not windows:
                logger.info('%s: no funding gaps in the requested range', symbol)

            for win_start, win_end in windows:
                if win_end <= win_start:
                    continue

                rates: List[FundingRate] = []
                coinbase_product = funding_product_map.get(symbol)
                if coinbase_client and coinbase_product:
                    try:
                        rates = await coinbase_client.get_funding_rate_history(coinbase_product, win_start, win_end)
                        for r in rates:
                            r.symbol = symbol
                            r.funding_source = "coinbase"
                    except Exception as e:
                        logger.warning(f"Coinbase funding fetch failed for {symbol}: {e}")

                # No cross-venue fallback. Funding feeds the `carry` component
                # of the net-return target, so another venue's rate trains the
                # carry head on a cash flow this account will never receive —
                # which is why `proxy_funding_symbols` is a promotion gate with a
                # threshold of zero. A fallback that writes data the gates then
                # refuse is a trap, not a safety net, so it is gone; the gap loop
                # stays for the day DCC credentials make CDE history reachable.
                source_used = "coinbase"

                if rates:
                    outcome = ingestor.ingest_funding(rates, venue=source_used)
                    inserted = outcome.inserted
                    symbol_inserted += inserted
                    source_metrics[symbol][source_used] += inserted
                    source_metrics[symbol]["start"] = win_start if source_metrics[symbol]["start"] is None else min(source_metrics[symbol]["start"], win_start)
                    source_metrics[symbol]["end"] = win_end if source_metrics[symbol]["end"] is None else max(source_metrics[symbol]["end"], win_end)
                    logger.info(f"  ✓ Inserted {inserted} normalized hourly funding rates via {source_used}")
                else:
                    logger.warning(f"  ⚠️ No funding rates found for {symbol} in {win_start.date()}->{win_end.date()}")

            # The current rate, once per symbol, outside the gap loop and with
            # no window test. CDE publishes no funding history, so this snapshot
            # is the only way the series ever grows — and treating it as a gap
            # fill broke that twice over. `funding_time` is the settlement the
            # rate applies to, which can sit *after* `end` (the backfill runs up
            # to "now", the next settlement is later that hour), so the window
            # test dropped it; and being inside the loop meant a range with no
            # gaps collected nothing at all, which is the steady state once an
            # hour has a row. Collection would have been silently intermittent
            # on exactly the data that cannot be re-fetched.
            #
            # Unconditional is safe: the store is keyed on
            # (venue, symbol, event_time), so repeated runs inside one
            # settlement hour upsert rather than duplicate.
            coinbase_product = funding_product_map.get(symbol)
            if coinbase_client and coinbase_product:
                try:
                    current, oi = await coinbase_client.get_contract_snapshot(
                        coinbase_product
                    )
                except Exception as exc:                      # noqa: BLE001
                    logger.warning('current funding failed for %s: %s', symbol, exc)
                    current = oi = None

                # Open interest rides the same payload as funding, on the
                # contract actually traded. It used to come from CCXT because
                # this client had no method for it, and a comment recorded that
                # gap as "Coinbase exposes no open-interest endpoint" — so six
                # features described gate's BTC/USDT book (21,579,279) instead of
                # BIP's (268,164). A snapshot, like funding: forward-only.
                if oi is not None:
                    oi.symbol = symbol
                    oi_outcome = ingestor.ingest_open_interest(
                        [oi], venue=venue_label or 'coinbase'
                    )
                    oi_inserted += oi_outcome.inserted
                    logger.info(
                        '%s: open interest %.0f contracts (snapshot — no history '
                        'endpoint)', symbol, oi.open_interest_contracts,
                    )

                if current is not None:
                    current.symbol = symbol
                    current.funding_source = 'coinbase'
                    outcome = ingestor.ingest_funding(
                        [current], venue=venue_label or 'coinbase'
                    )
                    symbol_inserted += outcome.inserted
                    source_metrics[symbol]['coinbase'] += outcome.inserted
                    logger.info(
                        '%s: current funding %.4f bp/hour at %s (snapshot — CDE '
                        'publishes no history)',
                        symbol, current.rate * 10_000, current.event_time,
                    )

            total_inserted += symbol_inserted
            progress.task_complete(symbol, count=symbol_inserted)

        progress.summary()

        print("\n📈 Funding Data Summary:")
        stats = db.get_funding_stats()
        for symbol, row in stats.items():
            daily_cost = row['avg_rate_bps'] * 24
            print(f"  {symbol}: {row['count']} records, avg {row['avg_rate_bps']:.4f} bps/hour (~{daily_cost:.4f} bps/day)")

        print("\n📋 Funding Source Coverage:")
        for symbol, m in source_metrics.items():
            coverage_start = m['start'].date().isoformat() if m['start'] else 'n/a'
            coverage_end = m['end'].date().isoformat() if m['end'] else 'n/a'
            logger.info(
                "Funding coverage %s [%s -> %s] coinbase=%s",
                symbol,
                coverage_start,
                coverage_end,
                m['coinbase'],
            )
            print(f"  {symbol}: coinbase={m['coinbase']} ({coverage_start} -> {coverage_end})")

        print(f"\nTotal funding rates inserted: {total_inserted}")
        print(f"Total open interest snapshots inserted: {oi_inserted}")
        return total_inserted

    finally:
        if coinbase_client:
            await coinbase_client.close()


# Symbol Resolution

async def resolve_coinbase_symbols(api_key: str, api_secret: str) -> List[str]:
    """
    Resolve active Coinbase Perpetual contract IDs.

    Returns product IDs found on Coinbase (e.g. "BIP-20DEC30-CDE") PLUS
    "-PERP" fallback symbols for any configured asset not covered by Coinbase,
    Assets with no CDE listing are named and excluded, not substituted.
    """
    logger.info("🔍 Resolving active Coinbase Perpetual contracts...")

    try:
        client = CoinbaseRESTClient(api_key, api_secret)
        target_codes = list(ASSET_TO_CODE_MAP.values())

        # Ask for everything first, then filter. "Found 16 contracts" used to be
        # the count of hardcoded codes that matched, not the count the venue
        # lists — and `ASSET_TO_CODE_MAP` has exactly 16 entries, so the filter
        # was fully binding and a newly-listed contract was invisible. Reporting
        # what was skipped turns "add a coin" from a discovery problem into a
        # one-line data change.
        all_products = await client.get_perpetual_products()
        products = await client.get_perpetual_products(target_codes=target_codes)
        await client.close()

        coinbase_symbols = [p['product_id'] for p in products]
        skipped = sorted(
            p['product_id'] for p in all_products
            if p.get('product_id') not in set(coinbase_symbols)
        )

        if coinbase_symbols:
            logger.info(f"✅ Found {len(coinbase_symbols)} Coinbase CDE contracts:")
            for s in coinbase_symbols:
                logger.info(f"   -> {s}")
        if skipped:
            logger.info(
                "ℹ️  %d further contract(s) the venue lists and this system does "
                "not model. Add the code to ASSET_TO_CODE_MAP, CONTRACT_UNITS, "
                "COIN_PROFILES and SPOT_PRODUCTS to include one:",
                len(skipped),
            )
            for s in skipped:
                logger.info(f"   (skipped) {s}")
        else:
            logger.warning("⚠️ No matching perpetuals found from Coinbase API")

        # An asset in ASSET_TO_CODE_MAP with no CDE listing is simply not
        # tradable here, and it is named rather than substituted. A `-PERP`
        # placeholder used to be appended for each one so CCXT could serve it,
        # which put another exchange's contract in the universe under a symbol
        # this account cannot trade.
        covered_codes = {sym.split('-')[0].upper() for sym in coinbase_symbols}
        unlisted = sorted(
            asset for asset, code in ASSET_TO_CODE_MAP.items()
            if code.upper() not in covered_codes
        )
        if unlisted:
            logger.warning(
                "%d modelled asset(s) have no CDE contract and are excluded: %s",
                len(unlisted), ', '.join(unlisted),
            )

        return coinbase_symbols

    except Exception as e:
        logger.error(f"⚠️ Failed to resolve Coinbase symbols: {e}")
        return []


# Main

async def main() -> int:
    # Collected rather than raised, so one instrument's failure does not abort the
    # rest — but a non-empty list becomes a non-zero exit, which is what makes a
    # failed scrape visible to the orchestrator.
    failures: list[str] = []

    parser = argparse.ArgumentParser(
        description="Unified Data Pipeline for Coinbase Perps Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --backfill-days 365
  python run_pipeline.py --backfill-days 365 --backfill-only
  python run_pipeline.py --funding-only --backfill-days 365
  python run_pipeline.py --skip-backfill
        """
    )
    
    # Time range
    parser.add_argument("--backfill-days", type=float, default=365,
                        help="Days of history to fetch. Fractional is fine: the "
                             "hourly cycle wants 0.25, not 1.")
    parser.add_argument("--backfill-hours", type=float, default=None,
                        help="Hours of history to fetch. Overrides --backfill-days. "
                             "The incremental cycle used to be quantised up to a "
                             "whole day, so 6h fetched 24h.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD), overrides --backfill-days")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    # Data types
    parser.add_argument("--timeframes", type=str, default=None, help="Comma-separated timeframes (default: 1h,1d)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: auto-detect from Coinbase)")
    
    # Modes
    parser.add_argument("--skip-backfill", action="store_true", help="Skip all backfill, only run real-time")
    parser.add_argument("--backfill-only", action="store_true", help="Only backfill, don't start real-time")
    parser.add_argument("--funding-only", action="store_true", help="Only backfill funding rates")
    parser.add_argument("--ohlcv-only", action="store_true", help="Only backfill OHLCV data")
    
    # Paths
    parser.add_argument("--spot-universe", action="store_true",
                        help="Scrape the Coinbase spot product for every "
                             "instrument the trader models, instead of naming "
                             "symbols. Implies --venue-label coinbase_spot. Use "
                             "this rather than typing --symbols: the hand-written "
                             "list was nine products against sixteen contracts.")
    parser.add_argument("--venue-label", type=str, default=None,
                        help="Store Coinbase-native rows under this venue label "
                             "instead of 'coinbase'. Use 'coinbase_spot' when "
                             "scraping {COIN}-USD spot: the perp and its spot "
                             "index resolve to the same base, so one label makes "
                             "the cross-venue basis a comparison with itself.")
    parser.add_argument("--db-path", type=str, default="./data/trading.db", help="Database path")
    
    # Network
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_time = datetime.strptime(args.start, "%Y-%m-%d")
        end_time = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.start:
        start_time = datetime.strptime(args.start, "%Y-%m-%d")
        end_time = utc_now()
    else:
        end_time = utc_now()
        start_time = end_time - (
            timedelta(hours=args.backfill_hours) if args.backfill_hours
            else timedelta(days=args.backfill_days)
        )
    
    # Ensure naive UTC
    start_time = ensure_naive_utc(start_time)
    end_time = ensure_naive_utc(end_time)
    
    timeframes = args.timeframes.split(",") if args.timeframes else DEFAULT_TIMEFRAMES
    
    # API credentials
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    
    print("=" * 70)
    print("🚀 UNIFIED DATA PIPELINE - Coinbase Perps Trading")
    print("=" * 70)
    span_hours = (end_time - start_time).total_seconds() / 3600.0
    print(
        f"Date Range: {start_time:%Y-%m-%d %H:%M} to {end_time:%Y-%m-%d %H:%M} "
        f"({span_hours:,.1f}h). Computed from the resolved bounds, so --start/--end "
        f"are reported honestly rather than echoing --backfill-days."
    )
    print(f"Timeframes: {timeframes}")
    print(f"Database: {args.db_path}")
    print(f"Venue label: {args.venue_label or 'coinbase'}")
    print()
    
    # Step 1: Resolve Symbols
    if args.spot_universe:
        from core.costs import spot_universe
        from core.profiles import COIN_PROFILES

        symbols = spot_universe(sorted(COIN_PROFILES))
        if not args.venue_label:
            # Spot and the perp resolve to the same base, so sharing a venue
            # label would make the cross-venue basis a comparison with itself.
            args.venue_label = 'coinbase_spot'
        logger.info(
            "spot universe: %d products for %d modelled contracts, venue label %r",
            len(symbols), len(COIN_PROFILES), args.venue_label,
        )
    elif args.symbols:
        # User-specified symbols
        symbols = [s.strip() for s in args.symbols.split(",")]
        logger.info(f"Using user-specified symbols: {symbols}")
    elif api_key and api_secret:
        # Auto-detect from Coinbase
        symbols = await resolve_coinbase_symbols(api_key, api_secret)
        if not symbols:
            logger.warning("Falling back to default symbols")
            symbols = DEFAULT_SYMBOLS
    else:
        # Default symbols (for funding-only mode without API keys)
        symbols = DEFAULT_SYMBOLS
        logger.info(f"Using default symbols: {symbols}")
    
    if not symbols:
        logger.error("❌ No symbols to process!")
        return
    
    # Step 2: Initialize Database
    db = SQLiteDatabase(args.db_path)
    db.initialize()
    logger.info(f"✓ Database initialized: {args.db_path}")
    
    # Step 3: Backfill Data
    pipeline = None
    
    if not args.skip_backfill:
        
        # OHLCV Backfill
        if not args.funding_only:
            if api_key and api_secret:
                # Initialize pipeline for OHLCV
                config = PipelineConfig(
                    symbols=symbols,
                    timeframes=timeframes,
                    coinbase_api_key=api_key,
                    coinbase_api_secret=api_secret,
                    db_path=args.db_path,
                    backfill_days=args.backfill_days,
                    venue_label=args.venue_label,
                )
                pipeline = await create_pipeline(config)
                
                # Register callbacks
                pipeline.on_ohlcv(on_new_candle)
                pipeline.on_ticker(on_ticker_update)
                pipeline.on_funding(on_funding_rate)
                
                await backfill_ohlcv(
                    pipeline, symbols, timeframes, start_time, end_time,
                )
            else:
                # An empty price history is not a successful scrape. This used to
                # warn and return 0, so `live_orchestrator._run_step` — which
                # inspects only the exit code — read it as a clean step.
                logger.error(
                    "no Coinbase API keys: OHLCV backfill skipped, so no price "
                    "history was collected"
                )
                failures.append("ohlcv: COINBASE_API_KEY/SECRET not set")
        
        # Funding Rate Backfill
        if not args.ohlcv_only:
            funding_rows = await backfill_funding_rates(
                symbols, start_time, end_time, db, api_key, api_secret,
                venue_label=args.venue_label,
            )
            # Zero funding is not a partial success. Carry is the edge this
            # system is built to capture, and `core/targets.py` decomposes net
            # return into price AND carry, so a price-only scrape cannot test the
            # hypothesis. The run used to exit 0 on this, which made a scrape that
            # collected nothing usable look successful.
            #
            # But only where funding exists to collect. A spot run has none by
            # definition — `SPOT_QUOTES` is why the funding map skips those
            # symbols in the first place — so demanding it here made the reference
            # scrape exit 1 on a completely correct run, which
            # `live_orchestrator._run_step` turns into an aborted cycle.
            perp_symbols = perpetual_symbols(symbols)
            if not funding_rows and perp_symbols:
                failures.append(
                    "funding: 0 rates collected for all "
                    f"{len(perp_symbols)} perpetual symbols — the carry features "
                    "and the carry component of every target will be empty"
                )
        
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("📊 DATA SUMMARY")
    print("=" * 70)
    if failures:
        for problem in failures:
            logger.error("scrape incomplete: %s", problem)
    
    # OHLCV summary
    print("\n📈 OHLCV Data:")
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*) as count, 
                   MIN(event_time) as earliest, MAX(event_time) as latest
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        for row in cursor.fetchall():
            print(f"  {row['symbol']} {row['timeframe']}: {row['count']} bars "
                  f"({row['earliest'][:10] if row['earliest'] else 'N/A'} to "
                  f"{row['latest'][:10] if row['latest'] else 'N/A'})")
    
    # Funding summary
    print("\n🏦 Funding Rates:")
    stats = db.get_funding_stats()
    if stats:
        for symbol, s in stats.items():
            daily_cost = s['avg_rate_bps'] * 24
            print(f"  {symbol}: {s['count']} records, avg {s['avg_rate_bps']:.4f} bps/hour (~{daily_cost:.2f} bps/day)")
    else:
        print("  No funding rate data")
    
    # OI summary. Unconditional now that it comes from the traded contract's own
    # product payload rather than an opt-in trip to another exchange.
    print("\n📊 Open Interest:")
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, COUNT(*) as count, AVG(open_interest_contracts) as avg_oi,
                   MAX(open_interest_contracts) as max_oi, venue
            FROM open_interest
            GROUP BY symbol, venue
        """)
        rows = cursor.fetchall()
        if rows:
            dead = []
            for row in rows:
                print(f"  {row['symbol']} @{row['venue']}: {row['count']} records, "
                      f"avg OI: {row['avg_oi']:,.0f} contracts")
                if not row['max_oi']:
                    dead.append(f"{row['symbol']}@{row['venue']} ({row['count']} rows)")
            # A series that never exceeds zero is not a statistic, it is a defect.
            # 16 contracts x 720 hours of zeros printed as "avg OI: 0 contracts"
            # and read as data: five features went all-NaN and
            # `liquidation_cascade_24h` carried on with its OI term disabled.
            if dead:
                logger.error(
                    "open interest is identically zero for %d series — these rows "
                    "are not measurements and the positioning features built on "
                    "them are not either: %s",
                    len(dead), ', '.join(dead[:8]) + (' ...' if len(dead) > 8 else ''),
                )
        else:
            print("  No open interest data")
    
    # Step 5: Real-time Collection
    if args.backfill_only:
        logger.info("\n🏁 Backfill-only mode, exiting.")
        db.close()
        if pipeline:
            await pipeline.stop()
        return 1 if failures else 0

    if not api_key or not api_secret:
        logger.error("\nno API keys: cannot start real-time collection")
        db.close()
        return 1
    
    # Ensure pipeline is initialized
    if pipeline is None:
        config = PipelineConfig(
            symbols=symbols,
            timeframes=timeframes,
            coinbase_api_key=api_key,
            coinbase_api_secret=api_secret,
            db_path=args.db_path,
            backfill_days=args.backfill_days,
            venue_label=args.venue_label,
        )
        pipeline = await create_pipeline(config)
        pipeline.on_ohlcv(on_new_candle)
        pipeline.on_ticker(on_ticker_update)
        pipeline.on_funding(on_funding_rate)
    
    try:
        await pipeline.start()
        logger.info("\n🚀 Real-time collection started. Press Ctrl+C to stop.")
        
        while True:
            await asyncio.sleep(3600)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    finally:
        if pipeline:
            await pipeline.stop()
        db.close()
        logger.info("👋 Pipeline shut down cleanly")


if __name__ == "__main__":
    # `main()` returned None on every path and the exit code was discarded, so
    # `live_orchestrator._run_step` — which inspects only the code — read a total
    # scrape failure as a successful step. An int return now becomes the status.
    raise SystemExit(asyncio.run(main()) or 0)
