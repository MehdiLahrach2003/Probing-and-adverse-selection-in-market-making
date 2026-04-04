"""
Ce script permet de calibrer le modèle d’intensité à partir d’une simulation.

L’idée générale est la suivante :

1. on simule un market maker dans un monde où les vrais paramètres A et k sont connus,
2. on transforme la trajectoire simulée en dataset (delta, n),
3. on estime A et k par maximum de vraisemblance,
4. on compare la courbe vraie, la courbe estimée et la courbe empirique,
5. on trace aussi la profile likelihood de k pour analyser l’identifiabilité.

Donc ce script sert à valider en détail la qualité du modèle d’intensité
et de sa calibration.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Outils de gestion de fichiers
from pathlib import Path

# Visualisation
import matplotlib.pyplot as plt

# Calcul numérique
import numpy as np

# Lecture YAML
import yaml

# Simulation du market maker
from optimal_quoting.backtest.engine import MMParams, run_mm_toy

# Construction du dataset de calibration
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm

# Diagnostic empirique de l’intensité
from optimal_quoting.calibration.diagnostics import empirical_intensity_binned

# Estimation MLE et profile likelihood
from optimal_quoting.calibration.mle import fit_intensity_exp_mle, profile_nll_over_k



def main() -> None:
    """
    Fonction principale du script.

    Elle :
    - lit les paramètres de simulation,
    - lance le backtest,
    - construit le dataset (delta, n),
    - estime A et k,
    - trace plusieurs figures de diagnostic.
    """

    # -------------------------------------------------------------
    # Étape 1 : lecture de la configuration
    # -------------------------------------------------------------
    cfg = yaml.safe_load(Path("configs/mm_toy.yaml").read_text(encoding="utf-8"))

    # On récupère le bloc mm_params
    mm = cfg["mm_params"]

    # -------------------------------------------------------------
    # Étape 2 : construction des paramètres de simulation
    # -------------------------------------------------------------
    p = MMParams(
        # Paramètres de temps
        dt=float(mm.get("dt", 1.0)),
        T=float(mm.get("T", 20000.0)),

        # Mid initial
        mid0=float(mm.get("mid0", 100.0)),

        # Volatilité du mid
        sigma=float(mm.get("sigma", 0.02)),

        # Vrais paramètres d’intensité utilisés par le simulateur
        A=float(mm.get("A", 1.2)),
        k=float(mm.get("k", 1.0)),

        # Paramètres de stratégie baseline
        base_spread=float(mm.get("base_spread", 0.2)),
        phi=float(mm.get("phi", 0.0)),
        order_size=float(mm.get("order_size", 0.01)),

        # Coûts de transaction
        fee_bps=float(mm.get("fee_bps", 0.0)),

        # Seed aléatoire
        seed=int(mm.get("seed", 123)),
        
        probing_p=0.2,
        probing_jitter=0.05,
        policy="probing",
    )

    # -------------------------------------------------------------
    # Étape 3 : lancer la simulation et construire le dataset
    # -------------------------------------------------------------

    # Simulation complète du market maker
    df = run_mm_toy(p)

    # Construction du dataset (delta, n)
    delta, n = build_intensity_dataset_from_mm(df, dt=p.dt)

    # -------------------------------------------------------------
    # Étape 4 : lecture des hyperparamètres de calibration
    # -------------------------------------------------------------
    cal = cfg.get("intensity_calibration", {})

    kb = cal.get("k_bounds", [0.0, 10.0])
    k_bounds = (float(kb[0]), float(kb[1])) if len(kb) >= 2 else (0.0, 10.0)

    grid_size = int(cal.get("grid_size", 300))

    # -------------------------------------------------------------
    # Étape 5 : estimation MLE de A et k
    # -------------------------------------------------------------
    est = fit_intensity_exp_mle(delta, n, dt=p.dt, k_bounds=k_bounds, grid_size=grid_size)

    # -------------------------------------------------------------
    # Étape 6 : affichage terminal
    # -------------------------------------------------------------
    print("=== True params ===")
    print(f"A_true={p.A:.6f}, k_true={p.k:.6f}")

    print("=== MLE estimate ===")
    print(f"A_hat={est.A:.6f}, k_hat={est.k:.6f}, nll={est.nll:.3f}")

    # -------------------------------------------------------------
    # Étape 7 : création du dossier figures
    # -------------------------------------------------------------
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Figure 1 : intensité vraie vs intensité ajustée
    # -------------------------------------------------------------

    """
    On construit une grille de deltas xs,
    puis on compare :

    - la vraie intensité du simulateur
    - l’intensité estimée par le MLE
    """
    xs = np.linspace(0.0, float(np.quantile(delta, 0.995)), 200)

    lam_true = p.A * np.exp(-p.k * xs)
    lam_hat = est.A * np.exp(-est.k * xs)

    plt.figure()
    plt.plot(xs, lam_true, label="true")
    plt.plot(xs, lam_hat, label="fitted")
    plt.title("Intensity fit: λ(δ)=A exp(-kδ)")
    plt.xlabel("delta")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_fit.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2 : intensité empirique vs intensité ajustée
    # -------------------------------------------------------------

    """
    On calcule une intensité empirique par bins de delta,
    puis on compare cette courbe empirique
    à la courbe ajustée par le modèle estimé.
    """
    emp = empirical_intensity_binned(delta, n, dt=p.dt, nbins=40, dmax_quantile=0.995)

    lam_fit_centers = est.A * np.exp(-est.k * emp.bin_centers)

    plt.figure()
    plt.plot(emp.bin_centers, emp.lambda_hat, label="empirical (binned)")
    plt.plot(emp.bin_centers, lam_fit_centers, label="fitted")
    plt.title("Empirical intensity vs fitted")
    plt.xlabel("delta (bin centers)")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_empirical_fit.png")
    plt.close()

    # -------------------------------------------------------------
    # Figure 3 : profile likelihood de k
    # -------------------------------------------------------------

    """
    On trace la negative log-likelihood en fonction de k.

    Cela permet de voir si le minimum est net ou non.
    Un minimum net signifie que k est bien identifiable.
    """
    k_grid = np.linspace(0.0, 5.0, 250)

    _, nlls = profile_nll_over_k(delta, n, dt=p.dt, k_grid=k_grid)

    # On recentre la courbe pour que le minimum soit à 0
    nlls = nlls - np.nanmin(nlls)

    plt.figure()
    plt.plot(k_grid, nlls)
    plt.title("Profile NLL(k) (shifted)")
    plt.xlabel("k")
    plt.ylabel("NLL(k) - min")
    plt.tight_layout()
    plt.savefig("reports/figures/intensity_profile_nll.png")
    plt.close()

    # -------------------------------------------------------------
    # Message final
    # -------------------------------------------------------------
    print("Saved plots:")
    print(" - reports/figures/intensity_fit.png")
    print(" - reports/figures/intensity_empirical_fit.png")
    print(" - reports/figures/intensity_profile_nll.png")


# Point d’entrée standard du script
if __name__ == "__main__":
    main()