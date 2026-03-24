"""Paper trading simulator: virtual account, order book, positions, P&L."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

INITIAL_CAPITAL = 1_000_000.0
BID_ASK_SPREAD_BPS = 5.0


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * (self.current_price or self.avg_cost)

    @property
    def unrealized_pnl(self) -> float:
        return self.shares * ((self.current_price or self.avg_cost) - self.avg_cost)

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Order:
    ticker: str
    side: str  # "buy" | "sell"
    shares: float
    limit_price: Optional[float] = None
    status: str = "pending"
    fill_price: Optional[float] = None
    filled_shares: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PaperTradingEngine:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.realized_pnl: float = 0.0
        self._lock = threading.Lock()

    def update_prices(self, prices: dict[str, float]) -> None:
        with self._lock:
            for t, p in prices.items():
                if t in self.positions:
                    self.positions[t].current_price = p

    def submit_order(self, ticker: str, side: str, shares: float, limit_price: Optional[float] = None) -> Order:
        order = Order(ticker=ticker, side=side, shares=shares, limit_price=limit_price)
        with self._lock:
            self.orders.append(order)
            self._try_fill(order)
        return order

    def _try_fill(self, order: Order) -> None:
        pos = self.positions.get(order.ticker)
        spread = BID_ASK_SPREAD_BPS / 10000.0

        if order.side == "buy":
            cost_per_share = order.limit_price or 0.0
            if cost_per_share <= 0:
                order.status = "rejected"
                return
            cost_per_share *= 1 + spread / 2
            total = order.shares * cost_per_share
            if total <= self.cash:
                order.status = "filled"
                order.fill_price = cost_per_share
                order.filled_shares = order.shares
                self.cash -= total
                if order.ticker in self.positions:
                    p = self.positions[order.ticker]
                    new_shares = p.shares + order.shares
                    new_cost = (p.cost_basis + total) / new_shares
                    self.positions[order.ticker] = Position(
                        ticker=order.ticker, shares=new_shares, avg_cost=new_cost, current_price=p.current_price
                    )
                else:
                    self.positions[order.ticker] = Position(
                        ticker=order.ticker, shares=order.shares, avg_cost=cost_per_share, current_price=cost_per_share
                    )
            else:
                order.status = "rejected"
        else:
            if not pos or pos.shares < order.shares:
                order.status = "rejected"
                return
            price = order.limit_price or pos.current_price or pos.avg_cost
            price *= 1 - spread / 2
            proceeds = order.shares * price
            order.status = "filled"
            order.fill_price = price
            order.filled_shares = order.shares
            self.cash += proceeds
            pnl = order.shares * (price - pos.avg_cost)
            self.realized_pnl += pnl
            new_shares = pos.shares - order.shares
            if new_shares < 1e-6:
                del self.positions[order.ticker]
            else:
                self.positions[order.ticker] = Position(
                    ticker=order.ticker, shares=new_shares, avg_cost=pos.avg_cost, current_price=pos.current_price
                )

    @property
    def total_equity(self) -> float:
        with self._lock:
            mv = sum(p.market_value for p in self.positions.values())
            return self.cash + mv

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_positions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "ticker": p.ticker,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in self.positions.values()
            ]

    def get_summary(self) -> dict:
        eq = self.total_equity
        return {
            "cash": self.cash,
            "total_equity": eq,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "positions": self.get_positions(),
        }


_engine: Optional[PaperTradingEngine] = None


def get_engine() -> PaperTradingEngine:
    global _engine
    if _engine is None:
        _engine = PaperTradingEngine()
    return _engine
