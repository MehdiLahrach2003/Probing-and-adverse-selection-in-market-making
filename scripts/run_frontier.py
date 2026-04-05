"""
Ce script lance une campagne complète d’expériences sur le probing.

L’objectif est de faire varier plusieurs paramètres de probing :

- la probabilité d’exploration p_explore,
- l’amplitude de perturbation jitter,
- plusieurs seeds aléatoires,

puis, pour chaque combinaison :

- exécuter le backtest,
- mesurer la performance de trading,
- estimer A et k par maximum de vraisemblance,
- calculer l’erreur sur k,
- sauvegarder les résultats,
- tracer des heatmaps et une frontière de compromis.

Ce script est l’un des plus importants du projet,
car il permet de visualiser le compromis entre :

- performance de trading,
- risque,
- qualité d’identification du paramètre k.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Outil pour manipuler les chemins
from pathlib import Path

# Lecture YAML
import yaml

# Manipulation de tableaux de résultats
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt

# Paramètres du market maker
from optimal_quoting.backtest.engine import MMParams

# Moteur expérimental de probing frontier
from optimal_quoting.experiments.probing_frontier import FrontierConfig, run_probing_frontier



def load_config(path: str) -> dict:
    """
    Fonction simple de lecture d’un fichier YAML.

    Retourne un dictionnaire Python.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def plot_heatmap_mean(df: pd.DataFrame, value_col: str, title: str, out_path: str) -> None:
    """
    Cette fonction trace une heatmap de la moyenne d’une métrique
    en fonction de :

    - p_explore
    - jitter

    Les moyennes sont calculées sur les différentes seeds.
    """

    # -------------------------------------------------------------
    # Construction de la matrice à afficher
    # -------------------------------------------------------------

    """
    On groupe les résultats par :
    - p_explore
    - jitter

    puis on prend la moyenne de la métrique demandée.
    Enfin, on transforme cela en tableau 2D avec unstack.
    """
    pivot = (
        df.groupby(["p_explore", "jitter"])[value_col]
        .mean()
        .unstack()
    )

    # -------------------------------------------------------------
    # Tracé de la heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    im = plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    # Barre de couleur
    plt.colorbar(im, label=f"Mean {value_col}")

    # Axes : on affiche les vraies valeurs numériques
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])

    # Labels
    plt.xlabel("jitter")
    plt.ylabel("p_explore")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def main() -> None:
    """
    Fonction principale du script.

    Elle :
    - charge les paramètres de base,
    - lit la grille de probing à tester,
    - lance toute la campagne d’expériences,
    - sauvegarde les résultats,
    - trace plusieurs heatmaps,
    - trace une frontière performance vs identifiabilité.
    """

    # -------------------------------------------------------------
    # Étape 1 : lecture de la configuration
    # -------------------------------------------------------------
    cfg = load_config("configs/mm_toy.yaml")

    # Vérification de la présence de la section mm_params
    if "mm_params" not in cfg:
        raise KeyError("configs/mm_toy.yaml must define a top-level `mm_params:` section.")

    # Paramètres de base du market maker
    base_params = MMParams(**cfg["mm_params"])

    # -------------------------------------------------------------
    # Étape 2 : lecture de la grille de probing
    # -------------------------------------------------------------
    frontier = cfg.get("frontier", {})

    # Liste des probabilités d’exploration à tester
    p_grid = list(map(float, frontier.get("p_grid", [0.0, 0.05, 0.1, 0.2, 0.3])))

    # Liste des amplitudes de jitter à tester
    jitter_grid = list(map(float, frontier.get("jitter_grid", [0.0, 0.02, 0.05, 0.1])))

    # Seeds aléatoires pour répéter les expériences
    seeds = list(map(int, frontier.get("seeds", [0, 1, 2, 3, 4])))

    # -------------------------------------------------------------
    # Étape 3 : paramètres de calibration MLE
    # -------------------------------------------------------------
    calib = cfg.get("intensity_calibration", {})

    # Bornes de recherche pour k
    k_bounds = tuple(map(float, calib.get("k_bounds", [0.0, 5.0])))

    # Taille de la grille du MLE
    grid_size = int(calib.get("grid_size", 300))

    # Construction de la configuration de frontier
    frontier_cfg = FrontierConfig(
        p_grid=p_grid,
        jitter_grid=jitter_grid,
        seeds=seeds,
        k_bounds=(k_bounds[0], k_bounds[1]),
        grid_size=grid_size,
    )

    # -------------------------------------------------------------
    # Étape 4 : lancement de toute la campagne d’expériences
    # -------------------------------------------------------------
    df = run_probing_frontier(base_params, frontier_cfg)

    # -------------------------------------------------------------
    # Étape 5 : sauvegarde des résultats bruts
    # -------------------------------------------------------------
    Path("reports").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    out_csv = "reports/frontier_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # -------------------------------------------------------------
    # Étape 6 : heatmap du PnL final
    # -------------------------------------------------------------
    plot_heatmap_mean(
        df,
        value_col="pnl_final",
        title="PnL final (mean over seeds)",
        out_path="reports/figures/frontier_pnl_heatmap.png",
    )

    # -------------------------------------------------------------
    # Étape 7 : heatmap du risque d’inventaire
    # -------------------------------------------------------------
    if "inv_max_abs" in df.columns:
        plot_heatmap_mean(
            df,
            value_col="inv_max_abs",
            title="Max |inventory| (mean over seeds)",
            out_path="reports/figures/frontier_inv_heatmap.png",
        )

    # -------------------------------------------------------------
    # Étape 8 : heatmap de l’erreur sur k
    # -------------------------------------------------------------
    plot_heatmap_mean(
        df,
        value_col="k_abs_error",
        title="|k_hat - k_true| (mean over seeds)",
        out_path="reports/figures/frontier_kerr_heatmap.png",
    )

    # -------------------------------------------------------------
    # Étape 9 : frontière performance vs identifiabilité
    # -------------------------------------------------------------

    """
    On agrège les résultats par couple (p_explore, jitter)
    en prenant la moyenne sur les seeds.
    """
    agg = df.groupby(["p_explore", "jitter"]).mean(numeric_only=True).reset_index()

    plt.figure(figsize=(7, 5))
    plt.scatter(agg["k_abs_error"].values, agg["pnl_final"].values, s=40)

    plt.xlabel("|k_hat - k_true|")
    plt.ylabel("PnL final")
    plt.title("PnL vs Identifiability (mean over seeds)")
    plt.tight_layout()
    plt.savefig("reports/figures/frontier_pnl_vs_kerr.png")
    plt.close()

    print("Saved reports/figures/frontier_*.png")


# Point d’entrée standard du script
if __name__ == "__main__":
    main()