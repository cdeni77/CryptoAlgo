#!/usr/bin/env python3
"""Pre-launch data and feature QA report for all tracked coins."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.coin_profiles import COIN_PROFILES

PREFIX_FOR_COIN = {
    "BTC": "BIP",
    "ETH": "ETP",
    "SOL": "SLP",
    "XRP": "XPP",
    "DOGE": "DOP",
}


@dataclass
class CoinPreflight:
    coin: str
    symbol: str | None
    ohlcv_rows_365d: int
    ohlcv_rows_30d: int
    funding_rows_365d: int
    oi_rows_365d: int
    feature_file: str | None
    feature_rows: int
    selected_feature_coverage_pct: float
    pruned_features_present: bool
    checks_passed: bool
    issues: List[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight QA checks for trader pipeline")
    parser.add_argument("--coins", type=str, default="BTC,ETH,SOL,XRP,DOGE")
    parser.add_argument("--db-path", type=str, default="./data/trading.db")
    parser.add_argument("--features-dir", type=str, default="./data/features")
    parser.add_argument("--output", type=str, default="./scripts/optimization_results/preflight_report.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any coin fails checks")
    return parser.parse_args()


def _pick_symbol(conn: sqlite3.Connection, prefix: str) -> str | None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol
        FROM ohlcv
        WHERE timeframe='1h' AND symbol LIKE ?
        GROUP BY symbol
        ORDER BY MAX(timestamp) DESC
        LIMIT 1
        """,
        (f"{prefix}%",),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _count_rows(conn: sqlite3.Connection, table: str, symbol: str, start_iso: str, timeframe: str | None = None) -> int:
    cur = conn.cursor()
    if table == "ohlcv":
        cur.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=? AND timestamp>=?",
            (symbol, timeframe or "1h", start_iso),
        )
    elif table == "funding_rates":
        cur.execute("SELECT COUNT(*) FROM funding_rates WHERE symbol=? AND timestamp>=?", (symbol, start_iso))
    elif table == "open_interest":
        cur.execute("SELECT COUNT(*) FROM open_interest WHERE symbol=? AND timestamp>=?", (symbol, start_iso))
    else:
        return 0
    row = cur.fetchone()
    return int(row[0] if row else 0)


def _feature_file(features_dir: Path, symbol: str) -> Path:
    return features_dir / f"{symbol.replace('-', '_')}_features.csv"


def _compute_feature_coverage(csv_path: Path, expected_features: List[str]) -> tuple[int, float]:
    if not csv_path.exists():
        return 0, 0.0
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0, 0.0
    present = [c for c in expected_features if c in df.columns]
    if not present:
        return len(df), 0.0
    non_null = df[present].notna().sum().sum()
    total = len(df) * len(present)
    coverage = (non_null / total) * 100.0 if total else 0.0
    return len(df), float(round(coverage, 2))


def main() -> int:
    args = _parse_args()
    now = datetime.now(timezone.utc)
    start_365 = (now - timedelta(days=365)).isoformat()
    start_30 = (now - timedelta(days=30)).isoformat()

    db_path = Path(args.db_path)
    features_dir = Path(args.features_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    rows: List[CoinPreflight] = []

    if not db_path.exists():
        payload = {
            "generated_at": now.isoformat(),
            "error": f"DB not found: {db_path}",
            "coins": [],
            "summary": {"coins_total": len(coins), "coins_passed": 0, "coins_failed": len(coins)},
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"❌ DB not found at {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        for coin in coins:
            prefix = PREFIX_FOR_COIN.get(coin, coin)
            symbol = _pick_symbol(conn, prefix)
            issues: List[str] = []
            if not symbol:
                rows.append(CoinPreflight(coin, None, 0, 0, 0, 0, None, 0, 0.0, False, False, ["missing_symbol_data"]))
                continue

            ohlcv_365 = _count_rows(conn, "ohlcv", symbol, start_365, timeframe="1h")
            ohlcv_30 = _count_rows(conn, "ohlcv", symbol, start_30, timeframe="1h")
            fr_365 = _count_rows(conn, "funding_rates", symbol, start_365)
            oi_365 = _count_rows(conn, "open_interest", symbol, start_365)

            feature_path = _feature_file(features_dir, symbol)
            profile = COIN_PROFILES.get(coin)
            expected = profile.feature_columns if profile else []
            feature_rows, coverage = _compute_feature_coverage(feature_path, expected)

            pruned_path = features_dir / f"pruned_features_{coin.lower()}.json"
            pruned_present = pruned_path.exists()

            if ohlcv_365 < 1500:
                issues.append("low_ohlcv_365d")
            if ohlcv_30 < 500:
                issues.append("low_ohlcv_30d")
            if fr_365 < 200:
                issues.append("low_funding_365d")
            if feature_rows < 1000:
                issues.append("low_feature_rows")
            if coverage < 90.0:
                issues.append("low_feature_coverage")
            if not pruned_present:
                issues.append("missing_pruned_features")

            rows.append(
                CoinPreflight(
                    coin=coin,
                    symbol=symbol,
                    ohlcv_rows_365d=ohlcv_365,
                    ohlcv_rows_30d=ohlcv_30,
                    funding_rows_365d=fr_365,
                    oi_rows_365d=oi_365,
                    feature_file=str(feature_path) if feature_path.exists() else None,
                    feature_rows=feature_rows,
                    selected_feature_coverage_pct=coverage,
                    pruned_features_present=pruned_present,
                    checks_passed=(len(issues) == 0),
                    issues=issues,
                )
            )
    finally:
        conn.close()

    passed = sum(1 for r in rows if r.checks_passed)
    payload = {
        "generated_at": now.isoformat(),
        "db_path": str(db_path),
        "features_dir": str(features_dir),
        "coins": [asdict(r) for r in rows],
        "summary": {
            "coins_total": len(rows),
            "coins_passed": passed,
            "coins_failed": len(rows) - passed,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print("=" * 70)
    print("🔎 PRELAUNCH QA SUMMARY")
    print("=" * 70)
    for r in rows:
        icon = "✅" if r.checks_passed else "⚠️"
        print(
            f"{icon} {r.coin:5s} sym={r.symbol or '-':18s} ohlcv365={r.ohlcv_rows_365d:5d} "
            f"fund365={r.funding_rows_365d:5d} feat_rows={r.feature_rows:5d} "
            f"cov={r.selected_feature_coverage_pct:5.1f}% pruned={'Y' if r.pruned_features_present else 'N'}"
        )
        if r.issues:
            print(f"   issues: {', '.join(r.issues)}")
    print(f"\nReport saved: {output_path}")

    if args.strict and any(not r.checks_passed for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
