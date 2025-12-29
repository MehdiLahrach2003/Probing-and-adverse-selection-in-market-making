from __future__ import annotations

import numpy as np
import pandas as pd


def pnl_series(df: pd.DataFrame) -> np.ndarray:
    return df["equity"].values


def returns_from_equity(equity: np.ndarray) -> np.ndarray:
    return np.diff(equity)


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-12) -> float:
    mu = np.mean(returns)
    sig = np.std(returns)
    if sig < eps:
        return 0.0
    return mu / sig


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(np.min(dd))


def inventory_stats(df: pd.DataFrame) -> dict[str, float]:
    q = df["inventory"].values
    return {
        "inv_mean": float(np.mean(q)),
        "inv_std": float(np.std(q)),
        "inv_max_abs": float(np.max(np.abs(q))),
    }


def performance_summary(df: pd.DataFrame) -> dict[str, float]:
    equity = pnl_series(df)
    rets = returns_from_equity(equity)

    out = {
        "pnl_final": float(equity[-1]),
        "pnl_mean": float(np.mean(rets)),
        "pnl_std": float(np.std(rets)),
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(equity),
    }
    out.update(inventory_stats(df))
    return out
