"""
Ce script définit des fonctions permettant de construire des features
de microstructure à partir d’un DataFrame de marché.

L’objectif est d’enrichir les données avec des variables classiques en finance :

- le mid price (prix central),
- le spread (coût de transaction),
- les log-returns (rendements logarithmiques),
- la volatilité réalisée.

Ces variables sont utiles pour :
- analyser la dynamique du marché,
- vérifier les propriétés statistiques (volatilité, bruit),
- enrichir les diagnostics et les expériences.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

import numpy as np
import pandas as pd



def add_mid_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cette fonction ajoute deux colonnes :

    - mid : prix médian entre bid et ask
    - spread : différence entre ask et bid

    Le mid est une approximation du "vrai" prix du marché.
    Le spread représente le coût implicite de transaction.
    """

    # On copie le DataFrame pour éviter de modifier l’original
    out = df.copy()

    # Calcul du mid price
    out["mid"] = 0.5 * (out["bid"] + out["ask"])

    # Calcul du spread
    out["spread"] = out["ask"] - out["bid"]

    return out



def add_log_returns(df: pd.DataFrame, price_col: str = "mid") -> pd.DataFrame:
    """
    Cette fonction calcule les log-returns d’un prix.

    Par défaut, on utilise le mid price.

    Formule :
        logret_t = log(price_t) - log(price_{t-1})

    Les log-returns sont largement utilisés en finance car :
    - ils sont additifs dans le temps,
    - ils sont plus adaptés aux modèles stochastiques.
    """

    # Copie du DataFrame
    out = df.copy()

    # On récupère la colonne de prix
    p = out[price_col].astype(float)

    # Calcul des log-returns
    out["logret"] = np.log(p).diff()

    return out



def realized_vol(df: pd.DataFrame, ret_col: str = "logret") -> float:
    """
    Cette fonction calcule la volatilité réalisée à partir des log-returns.

    Formule :
        vol = racine carrée de la moyenne des carrés des rendements

    Cela correspond à une estimation empirique de la volatilité du prix.
    """

    # On enlève les valeurs NaN (premier diff généralement)
    r = df[ret_col].dropna().to_numpy(dtype=float)

    # Si aucun rendement n’est disponible, on retourne NaN
    if len(r) == 0:
        return float("nan")

    # Calcul de la volatilité réalisée
    return float(np.sqrt(np.mean(r * r)))