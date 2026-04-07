"""
Tests unitaires pour le modèle Avellaneda–Stoikov.

Ce fichier vérifie deux propriétés fondamentales :

1. Symétrie lorsque l’inventaire est nul :
   si q = 0, alors delta_bid = delta_ask.

2. Effet de l’inventaire sur les quotes :
   - si q > 0 (position longue), on veut vendre :
        → ask plus loin (on vend plus cher)
        → bid plus proche (on évite d’acheter)
   - si q < 0 (position courte), on veut acheter :
        → ask plus proche
        → bid plus loin
"""

import pytest

# Paramètres du modèle AS + fonction de calcul des deltas
from optimal_quoting.model.avellaneda_stoikov import ASParams, as_deltas



def test_as_symmetry_at_zero_inventory():
    """
    Test de symétrie.

    Si l’inventaire est nul (q = 0),
    le market maker est neutre.

    Donc :
        delta_bid = delta_ask

    On vérifie ici que les deux deltas sont égaux
    (approximation float avec pytest.approx).
    """

    # Paramètres du modèle
    p = ASParams(gamma=0.1, sigma=0.02, k=1.0, T=100.0)

    # Calcul des deltas
    db, da = as_deltas(q=0.0, t=0.0, p=p)

    # Vérification de la symétrie
    assert db == pytest.approx(1.0)
    assert da == pytest.approx(1.0)



def test_as_inventory_skew_direction():
    """
    Test du comportement selon l’inventaire.

    On compare trois cas :
    - q = 0 (référence)
    - q > 0 (long)
    - q < 0 (short)

    On vérifie que les deltas évoluent dans la bonne direction.
    """

    p = ASParams(gamma=0.1, sigma=0.02, k=2.0, T=100.0)

    # Cas neutre
    db0, da0 = as_deltas(q=0.0, t=0.0, p=p)

    # Cas long (on possède l’actif)
    dbp, dap = as_deltas(q=+10.0, t=0.0, p=p)

    # Cas short (on est vendeur)
    dbn, dan = as_deltas(q=-10.0, t=0.0, p=p)


    """
    Cas q > 0 (long) :

    - on veut vendre → ask plus loin (plus cher)
    - on veut éviter d’acheter → bid plus proche

    Donc :
        delta_ask augmente
        delta_bid diminue
    """
    assert dap > da0
    assert dbp < db0


    """
    Cas q < 0 (short) :

    - on veut acheter → bid plus loin
    - on veut vendre moins facilement → ask plus proche

    Donc :
        delta_ask diminue
        delta_bid augmente
    """
    assert dan < da0
    assert dbn > db0