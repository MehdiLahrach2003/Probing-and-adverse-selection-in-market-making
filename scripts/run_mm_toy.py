"""
Ce script lance une simulation simple de market making.

Il sert à faire tourner le moteur principal du projet
dans un cadre toy, c’est-à-dire simplifié.

Le script suit les étapes suivantes :

1. lire un fichier de configuration YAML,
2. construire les paramètres du market maker,
3. lancer la simulation via run_mm_toy,
4. récupérer la trajectoire complète,
5. tracer la courbe d’equity,
6. sauvegarder cette figure dans reports/figures.

C’est donc le script le plus simple pour voir le projet tourner de bout en bout.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Outil pratique pour manipuler des chemins de fichiers
from pathlib import Path

# Matplotlib pour tracer la figure
import matplotlib.pyplot as plt

# YAML pour lire la configuration
import yaml

# Moteur principal du backtest
from optimal_quoting.backtest.engine import MMParams, run_mm_toy



def main() -> None:
    """
    Fonction principale du script.

    Elle :
    - lit la configuration,
    - construit les paramètres,
    - lance la simulation,
    - trace et sauvegarde la courbe d’equity.
    """

    # -----------------------------------------------------------------
    # Étape 1 : lecture de la configuration YAML
    # -----------------------------------------------------------------

    """
    On lit le fichier configs/mm_toy.yaml.
    Ce fichier contient un bloc mm_params avec tous les paramètres utiles.
    """
    cfg = yaml.safe_load(Path("configs/mm_toy.yaml").read_text(encoding="utf-8"))

    # On extrait le sous-dictionnaire des paramètres du market maker
    mm = cfg["mm_params"]

    # -----------------------------------------------------------------
    # Étape 2 : construction des paramètres de simulation
    # -----------------------------------------------------------------

    """
    On construit un objet MMParams à partir du YAML.

    Chaque paramètre est lu dans le fichier config.
    Si une clé est absente, on utilise une valeur par défaut.
    """
    p = MMParams(
        # Paramètres de temps
        dt=float(mm.get("dt", 1.0)),
        T=float(mm.get("T", 20000.0)),

        # Mid initial
        mid0=float(mm.get("mid0", 100.0)),

        # Volatilité du mid dans la simulation toy
        sigma=float(mm.get("sigma", 0.02)),

        # Paramètres de l’intensité "vraie" du simulateur
        A=float(mm.get("A", 1.2)),
        k=float(mm.get("k", 1.0)),

        # Paramètres de la stratégie baseline
        base_spread=float(mm.get("base_spread", 0.2)),
        phi=float(mm.get("phi", 0.0)),
        order_size=float(mm.get("order_size", 0.01)),

        # Frais de transaction
        fee_bps=float(mm.get("fee_bps", 0.0)),

        # Seed pour reproductibilité
        seed=int(mm.get("seed", 0)),
    )

    # -----------------------------------------------------------------
    # Étape 3 : lancement de la simulation
    # -----------------------------------------------------------------

    """
    Ici on appelle le moteur principal du projet.

    Le résultat est un DataFrame contenant toute la trajectoire simulée :
    - temps
    - mid
    - inventory
    - cash
    - equity
    - bid / ask
    - fills
    """
    df = run_mm_toy(p)

    # -----------------------------------------------------------------
    # Étape 4 : création du dossier de sortie
    # -----------------------------------------------------------------

    """
    On s’assure que le dossier reports/figures existe
    avant d’essayer de sauvegarder la figure.
    """
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Étape 5 : tracé de la courbe d’equity
    # -----------------------------------------------------------------

    """
    On trace l’equity en fonction du temps.

    Cela permet de visualiser rapidement si la stratégie
    semble gagner ou perdre de l’argent au cours du temps.
    """
    plt.figure()
    plt.plot(df["time_s"], df["equity"])
    plt.title("Equity curve (toy MM)")
    plt.tight_layout()
    plt.savefig("reports/figures/mm_toy_equity.png")
    plt.close()

    # -----------------------------------------------------------------
    # Étape 6 : affichage terminal
    # -----------------------------------------------------------------

    """
    On affiche les dernières lignes du DataFrame
    pour voir l’état final de la simulation.
    """
    print(df.tail())

    # Message indiquant où la figure a été sauvegardée
    print("Saved reports/figures/mm_toy_equity.png")


# Point d’entrée standard d’un script Python
if __name__ == "__main__":
    main()