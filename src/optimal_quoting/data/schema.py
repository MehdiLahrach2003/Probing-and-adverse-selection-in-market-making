from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Trade:
    ts: datetime
    price: float
    size: float
    side: Side


@dataclass(frozen=True)
class TopOfBook:
    ts: datetime
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None

    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    def spread(self) -> float:
        return self.ask - self.bid
