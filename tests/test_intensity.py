"""
Ce fichier contient les tests unitaires de la fonction intensity_exp.

Cette fonction implémente le modèle d’intensité exponentielle :

    lambda de delta = A fois exp de moins k fois delta

où :
- A représente l’intensité maximale lorsque delta vaut zéro,
- k représente la vitesse de décroissance,
- delta représente la distance entre la quote et le mid.

Ces tests vérifient trois choses essentielles :

1. la formule donne bien la bonne valeur au point delta = 0,
2. l’intensité décroît quand delta augmente,
3. les paramètres invalides déclenchent bien une erreur.

Ces propriétés sont fondamentales pour tout le projet,
car le modèle de fills repose sur cette intensité.
"""



# Module math standard pour comparer des floats proprement
import math

# Pytest sert à écrire et exécuter les tests
import pytest

# Fonction que l’on veut tester
from optimal_quoting.model.intensity import intensity_exp



def test_intensity_exp_basic():
    """
    Test du cas de base.

    Si delta = 0, alors :
        lambda(0) = A * exp(0) = A

    Donc avec :
        A = 2.0
        k = 1.0
        delta = 0.0

    on doit obtenir exactement 2.0.
    """

    assert math.isclose(intensity_exp(2.0, 1.0, 0.0), 2.0)



def test_intensity_exp_monotone():
    """
    Test de monotonie.

    On vérifie que l’intensité diminue quand delta augmente.

    Intuition économique :
    plus la quote est éloignée du mid,
    moins elle a de chance d’être exécutée.

    Donc :
        lambda(1.0) < lambda(0.0)
    """

    # Intensité quand delta = 0
    lam0 = intensity_exp(2.0, 1.0, 0.0)

    # Intensité quand delta = 1
    lam1 = intensity_exp(2.0, 1.0, 1.0)

    # L’intensité à delta = 1 doit être plus faible
    assert lam1 < lam0



@pytest.mark.parametrize(
    "A,k,delta",
    [
        (0.0, 1.0, 0.0),   # A invalide : doit être > 0
        (1.0, 0.0, 0.0),   # k invalide : doit être > 0
        (1.0, 1.0, -0.1),  # delta invalide : doit être >= 0
    ],
)
def test_intensity_exp_invalid(A, k, delta):
    """
    Test des cas invalides.

    On vérifie que la fonction refuse les paramètres incohérents :

    - A <= 0
    - k <= 0
    - delta < 0

    Dans chacun de ces cas, la fonction doit lever une ValueError.
    """

    with pytest.raises(ValueError):
        intensity_exp(A, k, delta)