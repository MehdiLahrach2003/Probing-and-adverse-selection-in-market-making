from __future__ import annotations
from optimal_quoting.strategy.avellaneda_stoikov import ASStrategyConfig, compute_as_quotes
from optimal_quoting.strategy.probing import ProbingConfig, compute_probing_quotes
from dataclasses import dataclass
import numpy as np
import pandas as pd

from optimal_quoting.model.intensity import intensity_exp
from optimal_quoting.sim.poisson import event_happens
from optimal_quoting.strategy.quotes import compute_quotes


@dataclass(frozen=True)
class MMParams:
    dt: float
    T: float
    mid0: float
    sigma: float               # per-step stdev for toy mid dynamics
    A: float
    k: float
    base_spread: float
    phi: float
    order_size: float
    fee_bps: float
    seed: int = 42
    probing_p: float = 0.0
    probing_jitter: float = 0.0
    probing_widen_only: bool = True
    policy: str = "baseline"   # "baseline" | "probing" | "as"
    gamma: float = 0.1



def run_mm_toy(p: MMParams) -> pd.DataFrame:
    rng = np.random.default_rng(p.seed)
    n = int(p.T / p.dt) + 1

    mid = np.empty(n, dtype=float)
    mid[0] = p.mid0

    q = 0.0
    cash = 0.0
    fee = p.fee_bps * 1e-4

    rows = []
    for t in range(n):
        if t > 0:
            mid[t] = max(0.01, mid[t - 1] + rng.normal(0.0, p.sigma))

        m = float(mid[t])
        if p.probing_p > 0.0 and p.probing_jitter > 0.0:
            qcfg = ProbingConfig(p_explore=p.probing_p, jitter=p.probing_jitter, widen_only=p.probing_widen_only)
            t_now = t * p.dt
            if p.policy == "as":
                quotes = compute_as_quotes(
                mid=m,
                q=q,
                t=t_now,
                T=p.T,
                sigma=p.sigma,
                k=p.k,
                cfg=ASStrategyConfig(gamma=p.gamma),
                )
            elif p.policy == "probing" and p.probing_p > 0.0 and p.probing_jitter > 0.0:
                qcfg = ProbingConfig(p_explore=p.probing_p, jitter=p.probing_jitter, widen_only=p.probing_widen_only)
                quotes = compute_probing_quotes(m, q, p.base_spread, p.phi, qcfg, rng)
            else:
                quotes = compute_quotes(m, q, p.base_spread, p.phi)

        else:
                quotes = compute_quotes(m, q, p.base_spread, p.phi)
        lam_bid = intensity_exp(p.A, p.k, quotes.delta_bid)
        lam_ask = intensity_exp(p.A, p.k, quotes.delta_ask)

        fill_bid = event_happens(lam_bid, p.dt, rng)
        fill_ask = event_happens(lam_ask, p.dt, rng)

        if fill_bid:
            q += p.order_size
            cash -= quotes.bid * p.order_size
            cash -= fee * quotes.bid * p.order_size

        if fill_ask:
            q -= p.order_size
            cash += quotes.ask * p.order_size
            cash -= fee * quotes.ask * p.order_size

        equity = cash + q * m
        rows.append((t * p.dt, m, q, cash, equity, quotes.bid, quotes.ask, fill_bid, fill_ask))

    return pd.DataFrame(
        rows,
        columns=["time_s", "mid", "inventory", "cash", "equity", "bid", "ask", "fill_bid", "fill_ask"],
    )
