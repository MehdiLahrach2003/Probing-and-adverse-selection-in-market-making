from __future__ import annotations

from dataclasses import dataclass

from optimal_quoting.strategy.quotes import Quotes
from optimal_quoting.model.avellaneda_stoikov import ASParams, as_deltas


@dataclass(frozen=True)
class ASStrategyConfig:
    gamma: float


def compute_as_quotes(mid: float, q: float, t: float, T: float, sigma: float, k: float, cfg: ASStrategyConfig) -> Quotes:
    p = ASParams(gamma=cfg.gamma, sigma=sigma, k=k, T=T)
    delta_bid, delta_ask = as_deltas(q=q, t=t, p=p)
    bid = mid - delta_bid
    ask = mid + delta_ask
    return Quotes(bid=bid, ask=ask, delta_bid=delta_bid, delta_ask=delta_ask)
