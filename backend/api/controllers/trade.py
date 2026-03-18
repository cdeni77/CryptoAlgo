from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.trade import Trade, TradeStatus


def get_trade(db: Session, trade_id: int) -> Optional[Trade]:
    return db.query(Trade).filter(Trade.id == trade_id).first()


def get_all_trades(db: Session, skip: int = 0, limit: int = 100) -> List[Trade]:
    return db.query(Trade).offset(skip).limit(limit).all()


def get_open_trades(db: Session) -> List[Trade]:
    return db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()


def get_closed_trades(db: Session) -> List[Trade]:
    return db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).all()


def get_trades_by_coin(db: Session, coin: str) -> List[Trade]:
    return db.query(Trade).filter(Trade.coin == coin).all()


def get_trade_stats(db: Session) -> Dict[str, Any]:
    """Aggregate trade performance stats."""
    all_trades = db.query(Trade).all()
    closed = [t for t in all_trades if t.status == TradeStatus.CLOSED]
    open_trades = [t for t in all_trades if t.status == TradeStatus.OPEN]

    wins = [t for t in closed if (t.net_pnl or 0) > 0]
    total_pnl = sum(t.net_pnl or 0 for t in closed)
    avg_pnl = total_pnl / len(closed) if closed else 0.0
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    by_coin: Dict[str, Dict[str, Any]] = {}
    for t in all_trades:
        coin = t.coin
        if coin not in by_coin:
            by_coin[coin] = {"coin": coin, "total": 0, "closed": 0, "wins": 0, "total_pnl": 0.0}
        by_coin[coin]["total"] += 1
        if t.status == TradeStatus.CLOSED:
            by_coin[coin]["closed"] += 1
            by_coin[coin]["total_pnl"] += t.net_pnl or 0
            if (t.net_pnl or 0) > 0:
                by_coin[coin]["wins"] += 1

    for coin_stats in by_coin.values():
        c = coin_stats["closed"]
        coin_stats["win_rate"] = round(coin_stats["wins"] / c * 100, 1) if c else 0.0
        coin_stats["avg_pnl"] = round(coin_stats["total_pnl"] / c, 4) if c else 0.0
        coin_stats["total_pnl"] = round(coin_stats["total_pnl"], 4)

    return {
        "total_trades": len(all_trades),
        "open_trades": len(open_trades),
        "closed_trades": len(closed),
        "wins": len(wins),
        "win_rate": round(win_rate, 1),
        "total_realized_pnl": round(total_pnl, 4),
        "avg_pnl_per_trade": round(avg_pnl, 4),
        "by_coin": sorted(by_coin.values(), key=lambda x: x["total"], reverse=True),
    }