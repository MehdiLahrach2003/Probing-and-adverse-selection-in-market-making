"""
Ce script lance un benchmark de plusieurs stratégies de market making.

L’objectif est de comparer différentes politiques
dans un même environnement simulé, en répétant les expériences
sur plusieurs seeds aléatoires.

Pour chaque stratégie et chaque seed, le script :

1. construit les paramètres du market maker,
2. lance un backtest complet,
3. calcule les métriques de performance,
4. stocke les résultats,
5. agrège les résultats par stratégie.

Ce script répond donc à la question :

    quelle stratégie est la meilleure en moyenne
    du point de vue trading (PnL, risque, stabilité) ?
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Outil pour lire / écrire des fichiers proprement
from pathlib import Path

# Import inutile ici dans cette version du script
import copy

# Outils de données
import pandas as pd

# Lecture YAML
import yaml

# Simulation du market maker
from optimal_quoting.backtest.engine import MMParams, run_mm_toy

# Mesures de performance
from optimal_quoting.metrics.performance import performance_summary



def main() -> None:
    """
    Fonction principale du benchmark.

    Elle :
    - lit la config benchmark,
    - boucle sur les stratégies et les seeds,
    - lance un backtest pour chaque cas,
    - calcule les métriques,
    - sauvegarde les résultats,
    - affiche les moyennes par stratégie.
    """

    # -------------------------------------------------------------
    # Étape 1 : lecture de la configuration benchmark
    # -------------------------------------------------------------
    cfg = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))

    # Paramètres communs à toutes les stratégies
    base = cfg["base"]

    # Liste des seeds pour répéter les expériences
    seeds = cfg["seeds"]

    # Dictionnaire des politiques à tester
    policies = cfg["policies"]

    # Liste qui contiendra les résultats de tous les runs
    rows = []

    # -------------------------------------------------------------
    # Étape 2 : boucle sur les stratégies et les seeds
    # -------------------------------------------------------------
    for name, pcfg in policies.items():
        for seed in seeds:

            """
            Construction des paramètres MMParams pour cette stratégie
            et cette seed.

            On combine :
            - les paramètres communs du bloc base
            - les paramètres propres à la politique
            """
            p = MMParams(
                # Paramètres de temps et de prix
                dt=float(base["dt"]),
                T=float(base["T"]),
                mid0=float(base["mid0"]),
                sigma=float(base["sigma"]),

                # Paramètres d’intensité du monde simulé
                A=float(base["intensity"]["A"]),
                k=float(base["intensity"]["k"]),

                # Paramètres de quoting de base
                base_spread=float(base["strategy"]["base_spread"]),
                phi=float(base["strategy"]["phi"]),
                order_size=float(base["strategy"]["order_size"]),

                # Coûts
                fee_bps=float(base["costs"]["fee_bps"]),

                # Seed de simulation
                seed=int(seed),

                # Choix de la politique
                policy=pcfg["policy"],

                # Paramètres spécifiques à certaines stratégies
                gamma=float(pcfg.get("gamma", 0.0)),
                probing_p=float(pcfg.get("probing_p", 0.0)),
                probing_jitter=float(pcfg.get("probing_jitter", 0.0)),
                probing_widen_only=bool(pcfg.get("probing_widen_only", True)),
            )

            # ---------------------------------------------------------
            # Étape 3 : lancement du backtest
            # ---------------------------------------------------------
            df = run_mm_toy(p)

            # ---------------------------------------------------------
            # Étape 4 : calcul des métriques de performance
            # ---------------------------------------------------------
            stats = performance_summary(df)

            # Ajout du nom de la stratégie et de la seed aux résultats
            stats["policy"] = name
            stats["seed"] = seed

            # Stockage
            rows.append(stats)

    # -------------------------------------------------------------
    # Étape 5 : construction du DataFrame final
    # -------------------------------------------------------------
    res = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Étape 6 : sauvegarde des résultats
    # -------------------------------------------------------------
    Path("reports").mkdir(exist_ok=True)
    res.to_csv("reports/benchmark_results.csv", index=False)

    # -------------------------------------------------------------
    # Étape 7 : affichage des moyennes par stratégie
    # -------------------------------------------------------------
    print(res.groupby("policy").mean(numeric_only=True))
    print("Saved reports/benchmark_results.csv")


# Point d’entrée standard du script
if __name__ == "__main__":
    main()