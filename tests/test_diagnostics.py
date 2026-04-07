"""
Test de la fonction empirical_intensity_binned.

Cette fonction construit une estimation empirique de l’intensité λ(δ)
en regroupant les deltas dans des bins.

Ce test vérifie que les sorties ont la bonne forme,
c’est-à-dire que chaque tableau a bien une taille égale au nombre de bins.
"""



# Numpy pour générer des données artificielles
import numpy as np

# Fonction à tester
from optimal_quoting.calibration.diagnostics import empirical_intensity_binned



def test_empirical_intensity_shapes():
    """
    Test de cohérence des dimensions.

    On génère des données aléatoires :
    - delta : distances entre quotes et mid
    - n : indicateurs de fill (0 ou 1)

    Puis on vérifie que la fonction retourne
    des tableaux de taille nbins.
    """

    # Générateur aléatoire reproductible
    rng = np.random.default_rng(0)

    # 1000 deltas uniformes entre 0 et 1
    delta = rng.random(1000)

    # 1000 événements (0 ou 1)
    n = rng.integers(0, 2, size=1000)

    # Construction de l’intensité empirique avec 20 bins
    emp = empirical_intensity_binned(delta, n, dt=0.1, nbins=20)


    """
    Vérification : chaque sortie doit avoir une taille égale à nbins
    """

    # Centres des bins
    assert emp.bin_centers.shape == (20,)

    # Intensité empirique
    assert emp.lambda_hat.shape == (20,)

    # Nombre de fills dans chaque bin
    assert emp.counts.shape == (20,)

    # Temps d’exposition dans chaque bin
    assert emp.exposure.shape == (20,)