"""
Test du moteur principal de simulation du market maker.

Ce test vérifie que :

1. la simulation peut être lancée sans erreur,
2. elle produit un DataFrame non vide,
3. ce DataFrame contient les colonnes essentielles (notamment equity).

Ce test ne vérifie pas la qualité du modèle,
mais uniquement que le pipeline complet fonctionne.
"""



# Import du moteur de simulation et des paramètres
from optimal_quoting.backtest.engine import MMParams, run_mm_toy



def test_run_mm_toy_runs():
    """
    Test simple du moteur de simulation.

    On crée un petit scénario artificiel,
    puis on vérifie que la simulation fonctionne correctement.
    """

    # Paramètres minimaux pour lancer une simulation
    p = MMParams(
        dt=1.0,              # pas de temps
        T=10.0,              # horizon court (test rapide)
        mid0=100.0,          # prix initial
        sigma=0.01,          # faible volatilité

        # paramètres d’intensité
        A=1.0,
        k=1.0,

        # paramètres de stratégie
        base_spread=0.2,
        phi=0.0,
        order_size=0.01,

        # coûts
        fee_bps=1.0,

        # seed pour reproductibilité
        seed=123,
    )

    # Lancement de la simulation
    df = run_mm_toy(p)


    """
    Vérification 1 : le DataFrame contient suffisamment de lignes

    On s’attend à avoir plusieurs pas de temps simulés.
    """
    assert len(df) > 5


    """
    Vérification 2 : la colonne equity existe

    equity est la variable centrale du projet :
        equity = cash + inventory × mid
    """
    assert "equity" in df.columns