"""
Ce script permet de tester uniquement la stratégie Avellaneda–Stoikov.

L’idée est simple :

On prend un environnement de marché simulé,
on applique uniquement la stratégie AS,
et on observe son comportement dans le temps.

Contrairement au benchmark :
- ici on ne compare pas plusieurs stratégies,
- on regarde uniquement AS isolément.

L’objectif est de comprendre :
- comment évolue le PnL,
- si la stratégie est stable,
- comment elle gère le risque.
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations



# Outils pour gérer les chemins de fichiers
from pathlib import Path



# Outils de visualisation
import matplotlib.pyplot as plt



# Lecture des fichiers YAML
import yaml



# Moteur de simulation du market maker
from optimal_quoting.backtest.engine import MMParams, run_mm_toy



def main() -> None:
    
    
    
    """
    Étape 1 : lire la configuration spécifique à AS
    
    Le fichier as_toy.yaml contient :
    - les paramètres du marché simulé,
    - les paramètres de l’intensité,
    - les paramètres de la stratégie AS (gamma).
    """
    cfg = yaml.safe_load(Path("configs/as_toy.yaml").read_text(encoding="utf-8"))



    """
    Étape 2 : construire les paramètres du market maker
    
    On crée un objet MMParams qui regroupe :
    - les paramètres de marché,
    - les paramètres d’intensité,
    - les paramètres de stratégie.
    
    IMPORTANT :
    policy = "as" → active la stratégie Avellaneda–Stoikov
    gamma → contrôle l’aversion au risque
    """
    p = MMParams(
        dt=float(cfg["dt"]),                     # pas de temps
        T=float(cfg["T"]),                       # horizon total
        mid0=float(cfg["mid0"]),                 # prix initial
        sigma=float(cfg["sigma"]),               # volatilité

        # paramètres d’intensité (monde simulé)
        A=float(cfg["intensity"]["A"]),
        k=float(cfg["intensity"]["k"]),

        # paramètres de quoting
        base_spread=float(cfg["strategy"]["base_spread"]),
        phi=float(cfg["strategy"]["phi"]),
        order_size=float(cfg["strategy"]["order_size"]),

        # coûts
        fee_bps=float(cfg["costs"]["fee_bps"]),

        # aléa
        seed=int(cfg["seed"]),

        # stratégie utilisée
        policy=str(cfg["policy"]["name"]),       # normalement "as"
        gamma=float(cfg["policy"]["gamma"]),     # aversion au risque
    )



    """
    Étape 3 : lancer la simulation
    
    run_mm_toy va :
    - simuler l’évolution du mid,
    - calculer les quotes avec AS,
    - simuler les fills,
    - mettre à jour cash + inventory,
    - construire toute la trajectoire.
    """
    df = run_mm_toy(p)



    """
    Étape 4 : créer le dossier pour les figures si nécessaire
    """
    Path("reports/figures").mkdir(parents=True, exist_ok=True)



    """
    Étape 5 : tracer la courbe d’equity
    
    equity = cash + inventory × mid
    
    Cette courbe représente :
    - la richesse totale du market maker dans le temps
    """
    plt.figure()
    plt.plot(df["time_s"], df["equity"])
    plt.title("Equity curve (AS policy)")
    plt.tight_layout()
    plt.savefig("reports/figures/as_equity.png")
    plt.close()



    """
    Étape 6 : afficher quelques lignes du résultat
    
    df contient :
    - mid
    - inventory
    - cash
    - equity
    - bid / ask
    - fills
    """
    print(df.tail())



    print("Saved reports/figures/as_equity.png")



# Point d’entrée du script
if __name__ == "__main__":
    main()