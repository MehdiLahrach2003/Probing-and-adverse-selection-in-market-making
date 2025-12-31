from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSummary:
    pnl_final: float
    pnl_mean: float
    pnl_std: float
    inv_mean: float
    inv_std: float
    inv_max_abs: float


def summarize_backtest(df: pd.DataFrame) -> Dict[str, float]:
    """
    Robust summary used by scripts (benchmark / stress).
    Expected columns when available:
      - equity (preferred) or pnl
      - inventory
    """
    if "equity" in df.columns:
        pnl_series = df["equity"].astype(float)
    elif "pnl" in df.columns:
        pnl_series = df["pnl"].astype(float)
    else:
        # last resort: zeros
        pnl_series = pd.Series(np.zeros(len(df), dtype=float))

    if "inventory" in df.columns:
        inv = df["inventory"].astype(float)
    elif "inv" in df.columns:
        inv = df["inv"].astype(float)
    else:
        inv = pd.Series(np.zeros(len(df), dtype=float))

    pnl_final = float(pnl_series.iloc[-1]) if len(pnl_series) else 0.0
    pnl_incr = pnl_series.diff().dropna()
    pnl_mean = float(pnl_incr.mean()) if len(pnl_incr) else 0.0
    pnl_std = float(pnl_incr.std(ddof=0)) if len(pnl_incr) else 0.0

    inv_mean = float(inv.mean()) if len(inv) else 0.0
    inv_std = float(inv.std(ddof=0)) if len(inv) else 0.0
    inv_max_abs = float(np.max(np.abs(inv.values))) if len(inv) else 0.0

    return {
        "pnl_final": pnl_final,
        "pnl_mean": pnl_mean,
        "pnl_std": pnl_std,
        "inv_mean": inv_mean,
        "inv_std": inv_std,
        "inv_max_abs": inv_max_abs,
    }
