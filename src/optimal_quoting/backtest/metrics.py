"""
Ce script permet de résumer un backtest en quelques métriques clés.

L’objectif est de transformer une trajectoire complète
(prix, inventory, cash, equity, etc.)
en un petit nombre de statistiques permettant de comparer des stratégies.

On mesure principalement deux choses :

- la performance (PnL)
- le risque (inventory)

Ces métriques sont utilisées dans les expériences
pour comparer différentes stratégies ou paramètres.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd



"""
Structure optionnelle pour stocker les résultats.

Elle n’est pas utilisée directement dans la fonction,
mais elle documente clairement les métriques calculées.
"""
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
    Cette fonction prend un DataFrame de backtest
    et retourne un résumé des performances.

    Elle est robuste :
    - accepte différentes conventions de noms de colonnes,
    - gère les cas où certaines colonnes sont absentes.
    """

    """
    Étape 1 : récupération du PnL.

    On privilégie la colonne "equity",
    qui représente la valeur totale du portefeuille.

    Sinon, on utilise "pnl".

    Sinon, on met des zéros.
    """
    if "equity" in df.columns:
        pnl_series = df["equity"].astype(float)
    elif "pnl" in df.columns:
        pnl_series = df["pnl"].astype(float)
    else:
        pnl_series = pd.Series(np.zeros(len(df), dtype=float))

    """
    Étape 2 : récupération de l’inventaire.

    Même logique :
    - "inventory"
    - ou "inv"
    - sinon zéros
    """
    if "inventory" in df.columns:
        inv = df["inventory"].astype(float)
    elif "inv" in df.columns:
        inv = df["inv"].astype(float)
    else:
        inv = pd.Series(np.zeros(len(df), dtype=float))

    """
    Étape 3 : PnL final.

    C’est la valeur finale du portefeuille.
    """
    pnl_final = float(pnl_series.iloc[-1]) if len(pnl_series) else 0.0

    """
    Étape 4 : incréments de PnL.

    On calcule les variations entre chaque instant.
    """
    pnl_incr = pnl_series.diff().dropna()

    """
    Étape 5 : statistiques du PnL.

    - moyenne : gain moyen par pas de temps
    - std : volatilité du PnL
    """
    pnl_mean = float(pnl_incr.mean()) if len(pnl_incr) else 0.0
    pnl_std = float(pnl_incr.std(ddof=0)) if len(pnl_incr) else 0.0

    """
    Étape 6 : statistiques de l’inventaire.

    - moyenne : biais directionnel
    - std : volatilité du risque
    - max_abs : exposition maximale
    """
    inv_mean = float(inv.mean()) if len(inv) else 0.0
    inv_std = float(inv.std(ddof=0)) if len(inv) else 0.0
    inv_max_abs = float(np.max(np.abs(inv.values))) if len(inv) else 0.0

    """
    Retour final sous forme de dictionnaire.
    """
    return {
        "pnl_final": pnl_final,
        "pnl_mean": pnl_mean,
        "pnl_std": pnl_std,
        "inv_mean": inv_mean,
        "inv_std": inv_std,
        "inv_max_abs": inv_max_abs,
    }