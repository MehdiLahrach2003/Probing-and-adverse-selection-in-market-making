"""
Tests end-to-end de la calibration MLE.

Ce fichier teste l’idée centrale du projet :

1. Avec probing, les données sont plus informatives,
   donc le MLE doit retrouver correctement les paramètres A et k.

2. Sans probing, les deltas observés sont trop peu variés,
   donc k devient mal identifiable et l’estimation doit être biaisée.

Ces tests ne vérifient donc pas seulement que le code tourne :
ils vérifient la validité statistique du pipeline complet.
"""



# Outils numériques
import numpy as np

# Moteur de simulation
from optimal_quoting.backtest.engine import MMParams, run_mm_toy

# Construction du dataset (delta, n)
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm

# Estimation MLE
from optimal_quoting.calibration.mle import fit_intensity_exp_mle



def _run_calibration(dt: float, probing: bool, seed: int):
    """
    Petite calibration end-to-end utilisée par les tests.

    Cette fonction :
    1. construit un scénario de simulation,
    2. lance le backtest,
    3. construit le dataset (delta, n),
    4. estime A et k par MLE,
    5. renvoie l’estimation et les vrais paramètres.

    Retour :
    --------
    est : résultat du MLE (A_hat, k_hat, nll)
    p   : paramètres vrais du scénario simulé
    """

    # Horizon long pour avoir assez de données
    T = 20000.0

    """
    Construction des paramètres de simulation.

    On fait varier :
    - dt
    - activation ou non du probing
    - seed

    Le reste est fixé de manière cohérente
    avec le cadre toy du projet.
    """
    
    """
    Si dt est petit, on réduit sigma
    pour garder une volatilité cohérente à l’échelle choisie.
    """
    
    """
    On adapte un peu le spread de base
    lorsque dt est petit, pour conserver des quotes raisonnables.
    """
    p = MMParams(
        dt=float(dt),
        T=T,
        mid0=100.0,

        sigma=0.006 if dt < 1.0 else 0.02,

        # Vrais paramètres du modèle d’intensité
        A=1.2,
        k=1.0,

        base_spread=0.4 if dt < 1.0 else 0.2,

        phi=0.0,
        order_size=0.01,
        fee_bps=0.0,
        seed=int(seed),

        # Politique choisie selon l’argument probing
        policy="probing" if probing else "baseline",

        # Paramètres de probing si activé
        probing_p=0.20 if probing else 0.0,
        probing_jitter=0.80 if probing else 0.0,
        probing_widen_only=True,
    )

    # -------------------------------------------------------------
    # Étape 1 : lancer la simulation
    # -------------------------------------------------------------
    df = run_mm_toy(p)

    # -------------------------------------------------------------
    # Étape 2 : construire le dataset (delta, n)
    # -------------------------------------------------------------
    delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

    # -------------------------------------------------------------
    # Étape 3 : lancer le MLE
    # -------------------------------------------------------------
    est = fit_intensity_exp_mle(delta, n, dt=p.dt, k_bounds=(0.0, 5.0), grid_size=300)

    return est, p



def test_mle_recovers_params_with_probing():
    """
    Ce test vérifie qu’avec probing,
    le pipeline de calibration retrouve correctement les vrais paramètres.

    On choisit ici :
    - dt = 0.1
    - probing activé

    car ce régime doit produire assez de variation dans les deltas
    pour rendre le problème identifiable.
    """

    est, p = _run_calibration(dt=0.1, probing=True, seed=123)

    # Vérification de base : les estimations doivent être finies et positives
    assert np.isfinite(est.A) and est.A > 0
    assert np.isfinite(est.k) and est.k > 0

    """
    On vérifie ensuite que les paramètres estimés
    sont proches des vrais paramètres du simulateur.

    Tolérance choisie ici : 0.3
    """
    assert abs(est.A - p.A) < 0.3
    assert abs(est.k - p.k) < 0.3



def test_mle_is_biased_without_probing():
    """
    Ce test vérifie qu’en absence de probing,
    le paramètre k est mal estimé.

    Ici, on choisit :
    - dt = 1.0
    - probing désactivé

    L’idée est que ce régime produit des deltas peu variés,
    donc k devient mal identifiable.
    """

    est, p = _run_calibration(dt=1.0, probing=False, seed=123)

    # Vérification minimale : k_hat doit être une valeur finie
    assert np.isfinite(est.k)

    """
    Ici, on ne veut PAS un bon fit.

    Au contraire, on s’attend à une erreur significative sur k,
    car le problème est non identifiable sans exploration suffisante.
    """
    assert abs(est.k - p.k) > 0.2 
    