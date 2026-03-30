"""
Test simple de la fonction performance_summary.

Ce test vérifie que la fonction :

- s’exécute correctement,
- retourne les métriques principales utilisées dans le projet.

On utilise un scénario artificiel très simple :
une equity qui augmente de manière constante,
sans inventaire.
"""



# Outils numériques
import numpy as np
import pandas as pd

# Fonction à tester
from optimal_quoting.metrics.performance import performance_summary



def test_performance_summary_smoke():
    """
    Test de base des métriques de performance.

    On crée un DataFrame simple avec :
    - une equity croissante,
    - un inventaire nul.

    Puis on vérifie que la fonction retourne
    les métriques attendues.
    """

    # Equity croissante : +1 à chaque étape
    df = pd.DataFrame(
        {
            "equity": np.cumsum(np.ones(100)),

            # Pas de position → pas de risque inventaire
            "inventory": np.zeros(100),
        }
    )

    # Calcul des métriques
    out = performance_summary(df)


    """
    Vérifications :

    - pnl_final doit exister (PnL final)
    - sharpe doit exister (performance ajustée du risque)
    """
    assert "pnl_final" in out
    assert "sharpe" in out
    assert out["pnl_final"] == 100
    assert out["pnl_mean"] > 0