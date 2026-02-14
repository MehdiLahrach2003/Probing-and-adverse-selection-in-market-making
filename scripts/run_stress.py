from __future__ import annotations

import inspect
import itertools
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

from optimal_quoting.backtest.engine import MMParams
import optimal_quoting.backtest.engine as engine_mod


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_params(cfg: dict, A: float, k: float, seed: int, policy: str) -> MMParams:
    return MMParams(
        dt=float(cfg["dt"]),
        T=float(cfg["T"]),
        mid0=float(cfg["mid0"]),
        sigma=float(cfg["sigma"]),
        A=float(A),
        k=float(k),
        base_spread=float(cfg["base_spread"]),
        phi=float(cfg["phi"]),
        order_size=float(cfg["order_size"]),
        fee_bps=float(cfg["fee_bps"]),
        seed=int(seed),
        policy=str(policy),
        probing_p=float(cfg["probing_p"]),
        probing_jitter=float(cfg["probing_jitter"]),
        probing_widen_only=bool(cfg["probing_widen_only"]),
        gamma=float(cfg["gamma"]),
    )


def _find_engine_runner() -> Callable[[MMParams], pd.DataFrame]:
    """
    Find a function in optimal_quoting.backtest.engine that:
      - is callable
      - accepts (p: MMParams) as first arg (possibly with optional args)
      - returns a pandas DataFrame
    """

    # First try common names (fast path)
    preferred = [
        "run_mm_toy",
        "run_mm",
        "run_backtest",
        "run",
        "simulate",
        "simulate_mm",
    ]
    for name in preferred:
        fn = getattr(engine_mod, name, None)
        if callable(fn):
            return lambda p, _fn=fn: _fn(p)

    # Otherwise, scan all callables in module with signature compatible with (MMParams, ...)
    candidates: list[Callable] = []
    for name, obj in vars(engine_mod).items():
        if not callable(obj):
            continue
        if name.startswith("_"):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue

        params = list(sig.parameters.values())
        if not params:
            continue

        # accept first parameter named like p/params/config and allow extra optional args
        first = params[0]
        if first.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            continue

        # All remaining params must have defaults (optional)
        if any(p.default is inspect._empty and i > 0 for i, p in enumerate(params)):
            continue

        candidates.append(obj)

    # Try candidates by executing once with a tiny run (copy p and shrink T)
    def _try(fn: Callable) -> Optional[Callable[[MMParams], pd.DataFrame]]:
        def _runner(p: MMParams) -> pd.DataFrame:
            return fn(p)
        return _runner

    # We can't safely call everything here without your p, so just pick the first candidate.
    # If it's wrong, we show a readable error with available names.
    if candidates:
        return _try(candidates[0])  # type: ignore[arg-type]

    # Hard fail with guidance
    available = sorted([k for k, v in vars(engine_mod).items() if callable(v) and not k.startswith("_")])
    raise ImportError(
        "Could not find a runnable backtest function in optimal_quoting.backtest.engine.\n"
        "Expected a callable like run_mm(p: MMParams) -> pd.DataFrame.\n"
        f"Callable symbols found: {available}"
    )


def _pick_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    cols = set(df.columns)
    for n in names:
        if n in cols:
            return n
    return None


def summarize_df(df: pd.DataFrame) -> dict:
    """
    Robust summary from a backtest DataFrame.

    We try to infer column names:
      pnl:  ['pnl', 'PnL', 'equity', 'cash', 'wealth']
      inv:  ['inv', 'inventory', 'position', 'q']
    """

    pnl_col = _pick_col(df, ["pnl", "PnL", "equity", "wealth", "cash"])
    inv_col = _pick_col(df, ["inv", "inventory", "position", "q"])

    out: dict = {}

    if pnl_col is not None:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna().to_numpy()
        if pnl.size:
            out["pnl_final"] = float(pnl[-1])
            out["pnl_mean"] = float(np.mean(np.diff(pnl))) if pnl.size > 1 else float(pnl[-1])
            out["pnl_std"] = float(np.std(np.diff(pnl))) if pnl.size > 1 else 0.0
    else:
        out["pnl_final"] = np.nan
        out["pnl_mean"] = np.nan
        out["pnl_std"] = np.nan

    if inv_col is not None:
        inv = pd.to_numeric(df[inv_col], errors="coerce").dropna().to_numpy()
        if inv.size:
            out["inv_std"] = float(np.std(inv))
            out["inv_max_abs"] = float(np.max(np.abs(inv)))
    else:
        out["inv_std"] = np.nan
        out["inv_max_abs"] = np.nan

    # sanity metrics
    out["n_rows"] = int(len(df))
    return out


def main() -> None:
    cfg = load_cfg("configs/stress.yaml")
    out_path = Path(cfg["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_engine = _find_engine_runner()

