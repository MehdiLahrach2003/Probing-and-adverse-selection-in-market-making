"""
Ce script permet de charger des données de marché depuis un fichier CSV
et de les transformer en un format standard utilisable dans le projet.

L’objectif est de convertir un CSV brut (avec noms de colonnes arbitraires)
en un DataFrame propre contenant :

- ts : timestamp
- bid : meilleur prix d’achat
- ask : meilleur prix de vente
- bid_size : quantité au bid (optionnel)
- ask_size : quantité au ask (optionnel)

Ce script joue donc le rôle de normalisation des données.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd



@dataclass(frozen=True)
class CSVSpec:
    """
    Cette classe décrit comment lire un CSV.

    Elle permet d’adapter le loader à différents formats de données.

    Exemple :
        ts_col = "time"
        bid_col = "best_bid"
        ask_col = "best_ask"
    """

    # Nom de la colonne timestamp dans le CSV
    ts_col: str = "timestamp"

    # Nom des colonnes bid et ask
    bid_col: str = "bid"
    ask_col: str = "ask"

    # Colonnes optionnelles pour les tailles
    bid_size_col: str | None = None
    ask_size_col: str | None = None



def load_top_of_book_csv(path: str | Path, spec: CSVSpec) -> pd.DataFrame:
    """
    Cette fonction charge un fichier CSV et le convertit en DataFrame standard.

    Étapes :
    - lecture du CSV
    - conversion des colonnes
    - validation des données
    - tri temporel
    """

    # Conversion du chemin en objet Path
    path = Path(path)

    # Lecture du CSV brut
    df = pd.read_csv(path)

    """
    Vérification que la colonne timestamp existe.
    """
    if spec.ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {spec.ts_col}")

    # DataFrame de sortie
    out = pd.DataFrame()

    """
    Conversion du timestamp en datetime.
    """
    out["ts"] = pd.to_datetime(df[spec.ts_col], utc=False)

    """
    Conversion des colonnes bid et ask.

    On force le type float.
    Si une valeur est invalide, une erreur est levée.
    """
    for col_name, out_name in [(spec.bid_col, "bid"), (spec.ask_col, "ask")]:
        if col_name not in df.columns:
            raise ValueError(f"Missing column: {col_name}")
        out[out_name] = pd.to_numeric(df[col_name], errors="raise")

    """
    Colonnes optionnelles : bid_size et ask_size.

    Elles sont ajoutées seulement si elles existent dans le CSV.
    """
    if spec.bid_size_col and spec.bid_size_col in df.columns:
        out["bid_size"] = pd.to_numeric(df[spec.bid_size_col], errors="raise")

    if spec.ask_size_col and spec.ask_size_col in df.columns:
        out["ask_size"] = pd.to_numeric(df[spec.ask_size_col], errors="raise")

    """
    Vérification de cohérence :

    Le ask doit être strictement supérieur au bid.
    Sinon, les données sont incohérentes.
    """
    if (out["ask"] <= out["bid"]).any():
        bad = out.index[out["ask"] <= out["bid"]][0]
        raise ValueError(f"Found ask <= bid at row {bad}")

    """
    Tri des données par timestamp pour garantir
    un ordre temporel correct.
    """
    out = out.sort_values("ts").reset_index(drop=True)

    return out