""" 
Ce script permet de transformer la simulation (run_mm_toy) en chiffres de performance.
Il répond à la question : est-ce que ma stratégie est bonne ou pas ?
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

import numpy as np
import pandas as pd



""" 
La fonction ci-dessous récupère l'equity dans le temps. 
equity = cash + inventory × mid
"""
def pnl_series(df: pd.DataFrame) -> np.ndarray:
    return df["equity"].values



""" 
La fonction ci-dessous calcule les variations d'equity, 
donc gains/pertes à chaque pas de temps
"""
def returns_from_equity(equity: np.ndarray) -> np.ndarray:
    return np.diff(equity)



""" 
La fonction ci-dessous mesure la rentabilité ajustée au risque. 
Interprétation : 
- grand → stratégie stable et rentable
- petit → bruit / risque élevé
"""
def sharpe_ratio(returns: np.ndarray, eps: float = 1e-12) -> float:
    mu = np.mean(returns)
    sig = np.std(returns)
    if sig < eps:
        return 0.0
    return mu / sig



""" 
La fonction ci-dessous mesure la plus grosse perte depuis un pic. 
Interprétation : 
- pire chute subie par la stratégie
"""
def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(np.min(dd))



""" 
La fonction ci-dessous mesure le risque d'inventaire 
"""
def inventory_stats(df: pd.DataFrame) -> dict[str, float]:
    q = df["inventory"].values
    return {
        "inv_mean": float(np.mean(q)),
        "inv_std": float(np.std(q)),
        "inv_max_abs": float(np.max(np.abs(q))),
    }



""" 
La fonction ci-dessous regroupe toutes les fonctions précédentes
"""
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
