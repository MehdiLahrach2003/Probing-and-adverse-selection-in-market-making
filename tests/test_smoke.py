"""
Test de base (smoke test) du framework de test.

Ce test ne vérifie rien de spécifique au projet.

Son seul objectif est de vérifier que :
- pytest fonctionne,
- l’environnement d’exécution est correct.

Si ce test échoue, cela signifie que le problème vient
de l’environnement et non du code du projet.
"""


def test_smoke():
    """
    Test trivial.

    Vérifie que Python et pytest fonctionnent correctement.
    """

    assert 1 + 1 == 2