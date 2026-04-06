"""
Test du loader de données de marché (top of book).

Ce test vérifie que :

1. un fichier CSV peut être correctement chargé,
2. les colonnes sont bien interprétées,
3. les données respectent les contraintes du marché (ask > bid).
"""



# Pandas pour créer un faux dataset
import pandas as pd

# Loader à tester
from optimal_quoting.data.loader import CSVSpec, load_top_of_book_csv



def test_load_top_of_book_csv(tmp_path):
    """
    Test du chargement d’un fichier CSV.

    tmp_path est un répertoire temporaire fourni par pytest,
    qui permet de créer des fichiers sans polluer le projet.
    """

    # Création d’un fichier CSV temporaire
    p = tmp_path / "top.csv"

    pd.DataFrame(
        {
            # Timestamps sous forme de chaînes
            "timestamp": ["2025-01-01 00:00:00", "2025-01-01 00:00:01"],

            # Bid prices
            "bid": [100.0, 100.1],

            # Ask prices (toujours > bid)
            "ask": [100.2, 100.3],
        }
    ).to_csv(p, index=False)


    """
    Définition du schéma de lecture.

    On indique quelles colonnes correspondent à :
    - timestamp
    - bid
    - ask
    """
    spec = CSVSpec(ts_col="timestamp", bid_col="bid", ask_col="ask")


    # Chargement du CSV
    df = load_top_of_book_csv(p, spec)


    """
    Vérification 1 : bon nombre de lignes
    """
    assert len(df) == 2


    """
    Vérification 2 : cohérence du marché

    Pour chaque ligne :
        ask > bid

    C’est une propriété fondamentale du marché.
    """
    assert (df["ask"] > df["bid"]).all()