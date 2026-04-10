"""
Ce script lance une campagne d’expériences sur le probing.

L’idée générale est la suivante :

on va faire varier les paramètres du probing, c’est-à-dire :
- la probabilité d’explorer,
- l’amplitude du jitter,

puis, pour chaque combinaison :
- on exécute le backtest,
- on mesure la performance de trading,
- on construit le dataset d’intensité,
- on estime les paramètres A et k par MLE,
- on compare k_hat au vrai k utilisé dans la simulation.

Donc ce script sert à étudier la frontière entre :

- exploration,
- qualité d’estimation,
- performance,
- risque d’inventaire.

C’est probablement l’un des scripts les plus importants du projet,
car il relie directement :
    stratégie -> données -> estimation -> performance
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

# Outils de simulation du market maker
from optimal_quoting.backtest.engine import MMParams, run_mm_toy

# Outils d’évaluation de la performance
from optimal_quoting.metrics.performance import performance_summary

# Outils de construction du dataset d’intensité
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm

# Outils d’estimation des paramètres d’intensité
from optimal_quoting.calibration.mle import fit_intensity_exp_mle



"""
Cette classe contient la configuration de la campagne d’expériences.

Elle décrit :
- les probabilités d’exploration à tester,
- les amplitudes de jitter à tester,
- les seeds aléatoires à tester,
- et quelques paramètres de calibration MLE.
"""
@dataclass(frozen=True)
class FrontierConfig:
    """
    Liste des probabilités d’exploration à tester.

    Exemple :
    - 0.0  : pas d’exploration
    - 0.1  : exploration de temps en temps
    - 0.5  : exploration fréquente
    """
    p_grid: list[float]

    """
    Liste des amplitudes de jitter à tester.

    Plus le jitter est grand,
    plus les deltas explorés peuvent s’écarter de la baseline.
    """
    jitter_grid: list[float]

    """
    Liste des seeds utilisées pour répéter les expériences
    avec plusieurs tirages aléatoires.
    """
    seeds: list[int]

    """
    Bornes de recherche pour le paramètre k dans le MLE.
    """
    k_bounds: tuple[float, float] = (0.0, 5.0)

    """
    Taille de la grille de recherche initiale dans le MLE.
    """
    grid_size: int = 300



def run_probing_frontier(base: MMParams, cfg: FrontierConfig) -> pd.DataFrame:
    """
    Cette fonction exécute toute la campagne expérimentale.

    Entrées :
    ---------
    base : paramètres de base du market maker

    cfg : configuration de la campagne de probing

    Sortie :
    --------
    Un DataFrame contenant, pour chaque run :
    - les paramètres de probing utilisés,
    - les métriques de performance de trading,
    - les paramètres d’intensité estimés,
    - l’erreur d’estimation sur k.
    """

    # Liste qui stockera une ligne de résultats par expérience
    rows: list[dict] = []

    """
    On fait une copie des paramètres de base sous forme de dictionnaire.

    Cela permet de reconstruire facilement un nouvel objet MMParams
    à chaque expérience, en modifiant uniquement :
    - la seed,
    - la policy,
    - les paramètres de probing.
    """
    base_dict = dict(base.__dict__)

    # Petite redondance dans le code original, sans conséquence
    base_dict = dict(base.__dict__)

    """
    On enlève du dictionnaire les champs qui seront redéfinis
    explicitement à chaque run.
    """
    for k in ("seed", "policy", "probing_p", "probing_jitter", "probing_widen_only"):
        base_dict.pop(k, None)

    """
    Triple boucle :
    - on parcourt toutes les probabilités d’exploration,
    - toutes les amplitudes de jitter,
    - et toutes les seeds.
    """
    for p_explore in cfg.p_grid:
        for jitter in cfg.jitter_grid:
            for seed in cfg.seeds:

                """
                Choix de la politique :

                - si p_explore et jitter sont strictement positifs,
                  on utilise la politique probing ;
                - sinon, on reste sur la baseline.
                """
                policy = "probing" if (p_explore > 0.0 and jitter > 0.0) else "baseline"

                """
                Reconstruction des paramètres complets pour cette expérience.
                """
                p = MMParams(
                    **base_dict,
                    seed=int(seed),
                    policy=policy,
                    probing_p=float(p_explore),
                    probing_jitter=float(jitter),
                    probing_widen_only=True,
                )

                # ---------------------------------------------------------
                # Étape 1 : exécution du backtest
                # ---------------------------------------------------------
                df = run_mm_toy(p)

                # ---------------------------------------------------------
                # Étape 2 : calcul des métriques de performance
                # ---------------------------------------------------------
                perf = performance_summary(df)

                # ---------------------------------------------------------
                # Étape 3 : construction du dataset d’intensité
                # ---------------------------------------------------------
                delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

                # ---------------------------------------------------------
                # Étape 4 : estimation MLE des paramètres A et k
                # ---------------------------------------------------------
                est = fit_intensity_exp_mle(
                    delta,
                    n,
                    dt=p.dt,
                    k_bounds=cfg.k_bounds,
                    grid_size=cfg.grid_size,
                )

                """
                On stocke les résultats de cette expérience.

                On enregistre :
                - les paramètres du probing,
                - la seed,
                - A_hat et k_hat estimés,
                - l’erreur absolue sur k,
                - toutes les métriques de performance.
                """
                rows.append(
                    {
                        "p_explore": float(p_explore),
                        "jitter": float(jitter),
                        "seed": int(seed),

                        # Paramètres estimés
                        "A_hat": float(est.A),
                        "k_hat": float(est.k),

                        # Erreur d’estimation de k
                        "k_abs_error": float(abs(est.k - p.k)),

                        # Métriques de trading
                        **perf,
                    }
                )

    # Retour final : tableau complet des résultats expérimentaux
    return pd.DataFrame(rows)