"""
Ce script sert à estimer les paramètres du modèle d’intensité exponentielle.

On suppose que l’intensité des fills suit la forme :

    lambda de delta égale A fois exponentielle de moins k fois delta

où :

- A représente l’intensité maximale, c’est-à-dire le niveau d’activité lorsque delta vaut zéro,
- k représente la vitesse de décroissance de l’intensité quand delta augmente,
- delta représente la distance entre la quote et le mid.

L’objectif du script est donc le suivant :

à partir d’un dataset composé de :
- deltas observés,
- nombres de fills observés à chaque instant,
- et d’un pas de temps dt,

on veut estimer les paramètres A et k par maximum de vraisemblance.

L’idée mathématique principale est la suivante :

1. On suppose que le nombre de fills observés sur chaque petit intervalle de temps
   suit une loi de Poisson d’intensité lambda_t fois dt.

2. Comme lambda_t dépend de delta_t selon la formule :
       lambda_t égale A fois exponentielle de moins k fois delta_t,
   on peut écrire la log-vraisemblance du modèle.

3. Pour un k fixé, on peut calculer explicitement le meilleur A.
   Cela réduit fortement le problème.

4. Il reste alors à chercher le meilleur k.
   Pour cela, on fait :
   - d’abord une recherche grossière sur une grille,
   - puis un raffinement local plus précis avec une golden-section search.

Donc ce script est le cœur de la calibration :
il apprend la loi d’intensité du marché à partir des données.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np



"""
La classe suivante sert simplement à stocker le résultat final de l'estimation.

Elle contient :

- A : estimation finale de l’intensité maximale,
- k : estimation finale de la vitesse de décroissance,
- nll : valeur finale de la negative log-likelihood, c’est-à-dire
        la valeur de la fonction objectif au point estimé.
"""
@dataclass(frozen=True)
class IntensityMLE:
    A: float
    k: float
    nll: float  # negative log-likelihood (Poisson approx)



def _A_hat_given_k(delta: np.ndarray, n: np.ndarray, dt: float, k: float) -> float:
    """
    Cette fonction calcule le meilleur A possible lorsque k est fixé.

    Formellement, si on suppose k connu, alors l'estimateur du maximum de vraisemblance
    de A admet une forme explicite :

        A_hat de k égale somme des n_t
        divisée par
        somme des dt fois exponentielle de moins k fois delta_t

    Autrement dit :
    - au numérateur, on compte le nombre total de fills observés,
    - au dénominateur, on cumule l’exposition pondérée par les deltas.

    Cela permet de ne pas faire une optimisation simultanée sur A et k.
    On optimise seulement sur k, ce qui simplifie beaucoup le problème.
    """

    # Vérification : le pas de temps doit être strictement positif
    if dt <= 0:
        raise ValueError("dt must be > 0")

    # Vérification : ici on autorise k nul, mais pas négatif
    if k < 0:
        raise ValueError("k must be >= 0")

    # Calcul des poids exponentiels : exp(-k * delta_t)
    w = np.exp(-k * delta)

    # Dénominateur de la formule de A_hat(k)
    denom = float(dt * np.sum(w))

    # Numérateur de la formule de A_hat(k)
    num = float(np.sum(n))

    # Sécurité : on évite un cas dégénéré où le dénominateur serait nul ou quasi nul
    if denom <= 0:
        raise ValueError("Degenerate denominator in A_hat(k)")

    """
    Si aucun événement n'est observé, alors num peut être nul et A_hat vaudrait zéro.
    Or plus loin, on prendra un logarithme, donc A = 0 poserait problème.
    On met donc un petit plancher numérique.
    """
    return max(num / denom, 1e-12)



def _poisson_nll(delta: np.ndarray, n: np.ndarray, dt: float, A: float, k: float) -> float:
    """
    Cette fonction calcule la negative log-likelihood du modèle de Poisson,
    à constante additive près.

    On suppose que, pour chaque instant t :

        lambda_t égale A fois exponentielle de moins k fois delta_t

    et que le nombre de fills observés suit un modèle de Poisson.

    La quantity calculée ici est :

        nll égale somme de :
            lambda_t fois dt
            moins
            n_t fois log de lambda_t

    Cette fonction mesure la qualité du fit :
    - plus elle est petite, meilleur est l’ajustement,
    - plus elle est grande, moins le modèle explique bien les données.
    """

    # Vérification : A doit être strictement positif
    if A <= 0:
        raise ValueError("A must be > 0")

    # Vérification : k doit être positif ou nul
    if k < 0:
        raise ValueError("k must be >= 0")

    # Vérification : le pas de temps doit être strictement positif
    if dt <= 0:
        raise ValueError("dt must be > 0")

    # Calcul de l’intensité lambda_t pour tous les instants t
    lam = A * np.exp(-k * delta)

    # Sécurité numérique : on évite log(0)
    lam = np.maximum(lam, 1e-18)

    # Calcul de la negative log-likelihood
    return float(np.sum(lam * dt - n * np.log(lam)))



def fit_intensity_exp_mle(
    delta: np.ndarray,
    n: np.ndarray,
    dt: float,
    k_bounds: tuple[float, float] = (0.0, 20.0),
    grid_size: int = 200,
) -> IntensityMLE:
    """
    Fonction principale d’estimation.

    Elle estime les paramètres A et k du modèle :

        lambda(delta) = A exp(-k delta)

    en utilisant :

    - la formule explicite de A_hat(k),
    - puis une recherche numérique sur k.

    La méthode suit deux grandes étapes :

    1. Recherche grossière sur une grille de valeurs de k
    2. Raffinement local autour du meilleur k trouvé

    Paramètres
    ----------
    delta : array de taille T
        distances observées, avec delta >= 0

    n : array de taille T
        nombres de fills observés à chaque instant
        souvent 0 ou 1 dans ce projet, mais peut aussi être un entier non négatif

    dt : float
        pas de temps

    k_bounds : couple (k_min, k_max)
        intervalle dans lequel on cherche k

    grid_size : int
        nombre de points de grille pour la recherche grossière
    """

    # Conversion des entrées en tableaux numpy de type float
    delta = np.asarray(delta, dtype=float)
    n = np.asarray(n, dtype=float)

    """
    Vérifications standards :
    - delta et n doivent être des tableaux unidimensionnels,
    - ils doivent avoir la même longueur,
    - delta doit être positif ou nul,
    - n doit être positif ou nul,
    - dt doit être strictement positif.
    """
    if delta.ndim != 1 or n.ndim != 1 or len(delta) != len(n):
        raise ValueError("delta and n must be 1D arrays with same length")
    if (delta < 0).any():
        raise ValueError("delta must be >= 0")
    if (n < 0).any():
        raise ValueError("n must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    # Lecture et vérification des bornes de recherche de k
    k_min, k_max = k_bounds
    if not (0 <= k_min < k_max):
        raise ValueError("Invalid k_bounds")

    # -----------------------------------------------------------------
    # Étape 1 : recherche grossière sur une grille de k
    # -----------------------------------------------------------------

    """
    On construit une grille régulière de valeurs candidates pour k.
    Pour chacune :
    - on calcule le meilleur A associé,
    - on calcule la nll correspondante,
    - puis on garde la meilleure combinaison.
    """
    ks = np.linspace(k_min, k_max, grid_size)

    # best contiendra :
    # (meilleure nll trouvée, A correspondant, k correspondant)
    best = (math.inf, None, None)

    for k in ks:
        kf = float(k)

        # Pour ce k fixé, on calcule le meilleur A
        A = _A_hat_given_k(delta, n, dt, kf)

        # Puis on évalue la qualité de ce couple (A, k)
        nll = _poisson_nll(delta, n, dt, A, kf)

        # Si c'est meilleur que ce qu'on a vu jusqu'ici, on met à jour
        if nll < best[0]:
            best = (nll, A, kf)

    # Le meilleur k issu de la recherche grossière
    _, _, k0 = best

    # -----------------------------------------------------------------
    # Étape 2 : raffinement local autour du meilleur point de grille
    # -----------------------------------------------------------------

    """
    Après la grille grossière, on affine la recherche autour du meilleur k trouvé.
    L’idée est de construire un petit intervalle autour de k0,
    puis de minimiser la nll plus précisément sur cet intervalle.
    """

    # Largeur d’un pas de grille
    step = (k_max - k_min) / max(grid_size - 1, 1)

    # Petit intervalle autour du meilleur point
    a = max(k_min, k0 - step)
    b = min(k_max, k0 + step)

    """
    Fonction auxiliaire :
    pour un k donné, on calcule d’abord le meilleur A,
    puis la nll correspondante.
    Ainsi, on réduit le problème à une fonction d’une seule variable : k.
    """
    def f(k: float) -> float:
        A = _A_hat_given_k(delta, n, dt, k)
        return _poisson_nll(delta, n, dt, A, k)

    # -----------------------------------------------------------------
    # Golden-section search
    # -----------------------------------------------------------------

    """
    On utilise maintenant une golden-section search,
    qui est une méthode classique pour minimiser une fonction
    sur un intervalle en une dimension, sans dérivées.
    """

    phi = (1 + math.sqrt(5)) / 2
    invphi = 1 / phi

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(60):
        # Si l’intervalle devient très petit, on s’arrête
        if abs(b - a) < 1e-8:
            break

        # On garde la moitié de l’intervalle la plus prometteuse
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)

    # Estimation finale de k : milieu du dernier intervalle
    k_hat = float(0.5 * (a + b))

    # Une fois k_hat trouvé, on recalcule le meilleur A correspondant
    A_hat = float(_A_hat_given_k(delta, n, dt, k_hat))

    # Valeur finale de la nll
    nll_hat = float(_poisson_nll(delta, n, dt, A_hat, k_hat))

    # On renvoie le résultat final de l’estimation
    return IntensityMLE(A=A_hat, k=k_hat, nll=nll_hat)



def profile_nll_over_k(
    delta: np.ndarray,
    n: np.ndarray,
    dt: float,
    k_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cette fonction ne sert pas à produire directement l’estimation finale.

    Elle sert à faire du diagnostic.

    Pour chaque valeur de k dans une grille donnée :
    - on calcule A_hat(k),
    - on calcule la nll(k),

    puis on renvoie :
    - le tableau des A_hat(k),
    - le tableau des nll(k).

    Cela permet par exemple :
    - de tracer le profil de la vraisemblance,
    - de voir si le minimum est bien net,
    - d’évaluer si k est bien identifiable.
    """

    # Conversion en tableaux numpy
    delta = np.asarray(delta, dtype=float)
    n = np.asarray(n, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)

    # Tableaux de sortie
    A_hats = np.empty_like(k_grid, dtype=float)
    nlls = np.empty_like(k_grid, dtype=float)

    # Pour chaque valeur de k dans la grille
    for i, k in enumerate(k_grid):
        kf = float(k)

        # Meilleur A associé à ce k
        A = _A_hat_given_k(delta, n, dt, kf)

        # Negative log-likelihood correspondante
        nll = _poisson_nll(delta, n, dt, A, kf)

        # Stockage
        A_hats[i] = A
        nlls[i] = nll

    return A_hats, nlls