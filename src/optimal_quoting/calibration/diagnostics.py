"""
Ce script sert à construire une estimation empirique de l’intensité.

Contrairement au fichier MLE, ici on ne suppose pas directement
une forme paramétrique du type :

    lambda de delta égale A fois exponentielle de moins k fois delta

À la place, on fait quelque chose de plus direct :

1. On découpe les deltas en plusieurs intervalles appelés bins.
2. Dans chaque bin, on compte :
   - combien de fills ont eu lieu,
   - combien de temps on a été exposé dans cette zone.
3. On estime alors l’intensité empirique dans le bin par :

    lambda_hat du bin = nombre d'événements / temps d'exposition

Cette approche sert surtout à faire du diagnostic :

- visualiser la courbe empirique de l’intensité,
- comparer cette courbe à la courbe théorique estimée,
- voir si certaines zones de delta sont mal explorées,
- vérifier si la calibration semble cohérente.

Donc ce script est un outil de vérification et d’analyse,
pas le cœur de l’estimation paramétrique.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
import numpy as np



"""
Cette classe sert à stocker le résultat final de l’estimation empirique.

Elle contient :

- bin_centers : les centres des bins de delta
- lambda_hat  : l’intensité empirique estimée dans chaque bin
- counts      : le nombre d’événements observés dans chaque bin
- exposure    : le temps total d’exposition dans chaque bin
"""
@dataclass(frozen=True)
class EmpiricalIntensity:
    bin_centers: np.ndarray
    lambda_hat: np.ndarray
    counts: np.ndarray
    exposure: np.ndarray



def empirical_intensity_binned(
    delta: np.ndarray,
    n: np.ndarray,
    dt: float,
    nbins: int = 40,
    dmax_quantile: float = 0.995,
) -> EmpiricalIntensity:
    """
    Cette fonction construit une estimation empirique de l’intensité
    en regroupant les observations par bins de delta.

    Entrées :
    ---------
    delta : array
        distances observées entre quotes et mid

    n : array
        nombres de fills observés, souvent 0 ou 1

    dt : float
        pas de temps

    nbins : int
        nombre de bins utilisés pour discrétiser delta

    dmax_quantile : float
        quantile maximal retenu pour construire les bins,
        afin d’éviter que quelques valeurs extrêmes déforment la grille

    Sortie :
    --------
    Un objet EmpiricalIntensity contenant :
    - centres des bins
    - intensités empiriques
    - comptes d’événements
    - expositions
    """

    # Conversion en tableaux numpy
    delta = np.asarray(delta, dtype=float)
    n = np.asarray(n, dtype=float)

    """
    Vérifications standards :
    - dt doit être strictement positif
    - delta et n doivent être des tableaux 1D de même longueur
    - delta doit être positif ou nul
    - n doit être positif ou nul
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if delta.ndim != 1 or n.ndim != 1 or len(delta) != len(n):
        raise ValueError("delta and n must be 1D arrays with same length")
    if (delta < 0).any():
        raise ValueError("delta must be >= 0")
    if (n < 0).any():
        raise ValueError("n must be >= 0")

    """
    On choisit une borne supérieure dmax pour construire les bins.

    Plutôt que de prendre le maximum absolu de delta,
    on prend un quantile élevé, par défaut 99,5 pour cent.

    Cela permet d’ignorer quelques valeurs extrêmes
    qui pourraient déformer inutilement les bins.
    """
    dmax = float(np.quantile(delta, dmax_quantile))

    # Petit plancher numérique pour éviter un cas dégénéré
    dmax = max(dmax, 1e-12)

    """
    Construction des bornes des bins entre 0 et dmax.

    Si nbins = 40, alors on obtient 40 intervalles.
    """
    edges = np.linspace(0.0, dmax, nbins + 1)

    """
    Attribution de chaque observation delta à un bin.

    np.digitize renvoie un indice de bin.
    On retire 1 pour revenir à une indexation naturelle commençant à 0.
    Puis on clippe pour garantir que tous les indices restent valides.
    """
    idx = np.digitize(delta, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    # Tableaux de stockage :
    # counts  = nombre total d’événements observés dans chaque bin
    # samples = nombre total d’observations tombant dans chaque bin
    counts = np.zeros(nbins, dtype=float)
    samples = np.zeros(nbins, dtype=float)

    """
    Pour chaque bin :
    - mask sélectionne les observations appartenant au bin
    - samples[b] compte combien d’observations sont dans ce bin
    - counts[b] additionne les n associés à ce bin
    """
    for b in range(nbins):
        mask = idx == b
        samples[b] = float(np.sum(mask))
        counts[b] = float(np.sum(n[mask]))

    """
    Calcul du temps d’exposition dans chaque bin.

    Si un bin contient samples[b] observations
    et que chaque observation dure dt,
    alors le temps total d’exposition est :

        exposure[b] = samples[b] * dt
    """
    exposure = samples * dt

    """
    Estimation empirique de l’intensité dans chaque bin :

        lambda_hat[b] = counts[b] / exposure[b]

    Si l’exposition est nulle, on met NaN,
    car on ne peut pas estimer une intensité sans observation.
    """
    lambda_hat = np.where(exposure > 0, counts / exposure, np.nan)

    """
    Calcul du centre de chaque bin.
    Ces valeurs seront utiles pour tracer la courbe empirique
    de lambda en fonction de delta.
    """
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Retour final sous forme d’objet structuré
    return EmpiricalIntensity(
        bin_centers=bin_centers,
        lambda_hat=lambda_hat,
        counts=counts,
        exposure=exposure,
    )