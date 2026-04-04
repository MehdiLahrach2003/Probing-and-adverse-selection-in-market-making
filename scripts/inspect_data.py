"""
Ce script permet d'inspecter des données de marché réelles (ou simulées).

L’idée est :

- charger un fichier CSV contenant bid / ask,
- le transformer en format propre,
- ajouter des variables de microstructure,
- calculer des statistiques simples,
- visualiser les séries.

C’est un script de vérification des données avant calibration ou backtest.
"""


from __future__ import annotations

# Gestion des fichiers
from pathlib import Path

# Visualisation
import matplotlib.pyplot as plt

# Lecture YAML
import yaml

# Chargement des données
from optimal_quoting.data.loader import CSVSpec, load_top_of_book_csv

# Features microstructure
from optimal_quoting.features.microstructure import add_log_returns, add_mid_spread, realized_vol


def main() -> None:
    
    
    """
    Étape 1 : lecture de la configuration
    
    Le fichier YAML contient :
    - chemin du CSV
    - noms des colonnes
    """
    cfg = yaml.safe_load(Path("configs/data_example.yaml").read_text(encoding="utf-8"))
    d = cfg["data"]


    """
    Étape 2 : définir le schéma des données
    
    Permet d’indiquer :
    - quelle colonne correspond au timestamp,
    - quelle colonne correspond au bid,
    - quelle colonne correspond au ask.
    """
    spec = CSVSpec(
        ts_col=d["ts_col"],
        bid_col=d["bid_col"],
        ask_col=d["ask_col"],
    )


    """
    Étape 3 : charger et nettoyer les données
    
    Le loader :
    - convertit les types,
    - vérifie bid < ask,
    - trie les données par temps.
    """
    df = load_top_of_book_csv(d["path"], spec)


    """
    Étape 4 : ajouter des features de microstructure
    
    - mid = (bid + ask) / 2
    - spread = ask - bid
    - log returns = variation du mid
    """
    df = add_mid_spread(df)
    df = add_log_returns(df)


    """
    Étape 5 : afficher des statistiques simples
    """
    print("Rows:", len(df))
    print("Mean spread:", float(df["spread"].mean()))
    print("Realized vol (per step):", realized_vol(df))


    """
    Étape 6 : créer dossier de sortie
    """
    Path("reports/figures").mkdir(parents=True, exist_ok=True)


    """
    Étape 7 : tracer le mid price
    """
    plt.figure()
    plt.plot(df["ts"], df["mid"])
    plt.title("Mid price")
    plt.tight_layout()
    plt.savefig("reports/figures/mid.png")
    plt.close()


    """
    Étape 8 : tracer le spread
    """
    plt.figure()
    plt.plot(df["ts"], df["spread"])
    plt.title("Spread")
    plt.tight_layout()
    plt.savefig("reports/figures/spread.png")
    plt.close()


    print("Saved plots to reports/figures/: mid.png, spread.png")


# Point d’entrée
if __name__ == "__main__":
    main()