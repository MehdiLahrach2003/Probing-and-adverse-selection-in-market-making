"""
Ce fichier contient un test unitaire pour la stratégie probing.

L’objectif est de vérifier qu’en mode :

    widen_only = True

la stratégie probing ne peut qu’élargir les deltas,
et donc ne peut pas produire des quotes plus proches du mid
que celles de la stratégie baseline.

Intuition :

- baseline avec spread = 0.2 donne :
    delta_bid = 0.1
    delta_ask = 0.1

- si probing est activé avec p_explore = 1.0,
  alors une perturbation est toujours appliquée

- si widen_only = True,
  cette perturbation doit être non négative

Donc les deltas finaux doivent être au moins égaux à 0.1.
"""



# Numpy sert ici à créer un générateur aléatoire reproductible
import numpy as np

# On importe la configuration probing et la fonction à tester
from optimal_quoting.strategy.probing import ProbingConfig, compute_probing_quotes



def test_probing_widens_deltas_when_enabled():
    """
    Ce test vérifie que le probing, lorsqu’il est activé
    en mode widen_only=True, ne réduit jamais les deltas.

    On choisit volontairement un cas très simple :
    - q = 0
    - phi = 0
    - base_spread = 0.2

    Dans ce cas, la stratégie baseline donnerait :
        delta_bid = 0.1
        delta_ask = 0.1

    Comme p_explore = 1.0, le probing est appliqué à coup sûr.
    Comme widen_only = True, le bruit ajouté doit être >= 0.

    Donc on doit obtenir :
        delta_bid >= 0.1
        delta_ask >= 0.1
    """

    # Générateur aléatoire fixé pour rendre le test reproductible
    rng = np.random.default_rng(0)

    # Configuration probing :
    # - p_explore = 1.0  → exploration toujours activée
    # - jitter = 0.5     → perturbation possible jusqu’à 0.5
    # - widen_only=True  → perturbation uniquement positive
    cfg = ProbingConfig(p_explore=1.0, jitter=0.5, widen_only=True)

    # Cas simple sans inventaire
    q = 0.0

    # Mid fixé
    mid = 100.0

    # Spread de base
    base_spread = 0.2

    # Pas de skew d’inventaire
    phi = 0.0

    # Calcul des quotes probing
    quotes = compute_probing_quotes(mid, q, base_spread, phi, cfg, rng)

    # La baseline donnerait 0.1 de chaque côté
    # Avec widen_only=True, les deltas finaux doivent être au moins égaux à 0.1
    assert quotes.delta_bid >= 0.1
    assert quotes.delta_ask >= 0.1