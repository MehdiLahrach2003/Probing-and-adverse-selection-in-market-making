"""
Ce script sert à construire le dataset utilisé pour la calibration de l’intensité.

L’objectif est de transformer les résultats du backtest (run_mm_toy),
c’est-à-dire un DataFrame contenant :

- les prix mid, bid, ask,
- les fills (fill_bid, fill_ask),

en un dataset exploitable pour l’estimation par maximum de vraisemblance.

On veut obtenir des couples de la forme :

    (delta_t, n_t)

où :

- delta_t = distance entre la quote et le mid,
- n_t = nombre de fills observés à cet instant (0 ou 1).

Important :

On utilise les deux côtés du carnet :

- côté bid :
    delta_bid = mid - bid
    n_bid = fill_bid

- côté ask :
    delta_ask = ask - mid
    n_ask = fill_ask

Puis on concatène tout pour obtenir un dataset deux fois plus grand.

Donc ce script fait le lien entre :

    simulation → dataset → MLE

C’est une étape essentielle car sans ce dataset,
on ne peut pas estimer les paramètres A et k.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

import numpy as np
import pandas as pd



def build_intensity_dataset_from_mm(df: pd.DataFrame, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cette fonction construit le dataset (delta, n) à partir du DataFrame
    généré par le market maker.

    Entrée :
    --------
    df : DataFrame contenant les colonnes :
        - mid
        - bid
        - ask
        - fill_bid
        - fill_ask

    dt : pas de temps (utile pour cohérence avec le modèle)

    Sortie :
    --------
    delta : array de taille 2T
        distances positives entre quotes et mid

    n : array de taille 2T
        nombre de fills observés (0 ou 1)
    """

    """
    Vérification de base :
    le pas de temps doit être strictement positif
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")



    """
    Vérification que le DataFrame contient bien toutes les colonnes nécessaires.

    On a besoin de :
    - mid : prix médian
    - bid : prix d'achat
    - ask : prix de vente
    - fill_bid : indicateur si le bid a été exécuté
    - fill_ask : indicateur si le ask a été exécuté
    """
    required = {"mid", "bid", "ask", "fill_bid", "fill_ask"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns in df: {sorted(missing)}")



    """
    Calcul des deltas côté bid et côté ask.

    delta_bid = mid - bid
    delta_ask = ask - mid

    Ces quantités doivent être positives si les quotes sont cohérentes.
    """
    delta_bid = (df["mid"] - df["bid"]).to_numpy(dtype=float)
    delta_ask = (df["ask"] - df["mid"]).to_numpy(dtype=float)



    """
    Vérification de cohérence :

    Les deltas doivent être positifs.
    Si ce n’est pas le cas, cela signifie que :
    - bid > mid, ou
    - ask < mid

    ce qui est incohérent dans un market making standard.
    """
    if (delta_bid < 0).any() or (delta_ask < 0).any():
        raise ValueError("Found negative deltas; check bid/ask vs mid consistency")



    """
    Conversion des fills en variables numériques.

    fill_bid et fill_ask sont des booléens (True / False).
    On les convertit en :

        True  -> 1
        False -> 0

    Cela correspond exactement à n_t dans le modèle de Poisson.
    """
    n_bid = df["fill_bid"].astype(int).to_numpy(dtype=float)
    n_ask = df["fill_ask"].astype(int).to_numpy(dtype=float)



    """
    Construction du dataset final.

    On concatène :

    - les deltas bid et ask
    - les fills bid et ask

    Donc si on avait T observations initiales,
    on obtient 2T observations au total.

    Cela permet d’utiliser toute l’information disponible.
    """
    delta = np.concatenate([delta_bid, delta_ask])
    n = np.concatenate([n_bid, n_ask])



    """
    Sortie finale :

    delta : distances utilisées dans le modèle d’intensité
    n : nombre de fills associés

    Ces deux tableaux seront utilisés dans le MLE.
    """
    return delta, n