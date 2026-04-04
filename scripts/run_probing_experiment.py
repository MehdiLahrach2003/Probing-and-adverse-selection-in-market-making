"""
Ce script compare une stratégie baseline à une stratégie probing.

L’idée est la suivante :

1. on charge une configuration baseline,
2. on construit une variante probing à partir de cette baseline,
3. on lance les deux simulations,
4. on transforme les trajectoires en datasets d’intensité,
5. on estime A et k par maximum de vraisemblance,
6. on compare les estimations au vrai A et au vrai k,
7. on trace un graphique simple pour comparer k_hat.

Ce script sert donc à répondre à une question centrale du projet :

    est-ce que le probing permet de mieux retrouver le vrai k que la baseline ?
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Outils utiles pour manipuler les dataclasses immuables
from dataclasses import fields, replace

# Outils pour manipuler les chemins
from pathlib import Path

# Copie de secours si besoin
import copy

# Outils de visualisation
import matplotlib.pyplot as plt

# Outil de lecture YAML
import yaml

# Moteur principal de simulation
from optimal_quoting.backtest.engine import MMParams, run_mm_toy

# Construction du dataset de calibration
from optimal_quoting.calibration.dataset import build_intensity_dataset_from_mm

# Estimation des paramètres d’intensité par MLE
from optimal_quoting.calibration.mle import fit_intensity_exp_mle



def _mmparams_field_names() -> set[str]:
    """
    Cette petite fonction retourne l’ensemble des noms de champs
    de la dataclass MMParams.

    Cela permet ensuite de filtrer proprement les clés du YAML
    pour ne garder que celles qui sont valides.
    """
    return {f.name for f in fields(MMParams)}



def load_mm_params(path: str) -> MMParams:
    """
    Charge un fichier YAML et construit un objet MMParams.

    Cette fonction est robuste :
    - elle accepte plusieurs styles de YAML,
    - elle filtre les clés invalides,
    - elle normalise les types,
    - elle ajoute des valeurs par défaut si besoin.
    """

    # Lecture du fichier YAML
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    # On accepte soit un bloc "mm_params", soit directement le dictionnaire
    mm = cfg.get("mm_params", cfg)

    # Noms de champs autorisés
    allowed = _mmparams_field_names()

    # On ne garde que les clés compatibles avec MMParams
    filtered = {k: v for k, v in mm.items() if k in allowed}

    # -------------------------------------------------------------
    # Normalisation des types numériques
    # -------------------------------------------------------------
    for k in ("dt", "T", "mid0", "sigma", "A", "k", "base_spread", "phi", "order_size", "fee_bps", "gamma"):
        if k in filtered:
            filtered[k] = float(filtered[k])

    if "seed" in filtered:
        filtered["seed"] = int(filtered["seed"])

    if "probing_p" in filtered:
        filtered["probing_p"] = float(filtered["probing_p"])

    if "probing_jitter" in filtered:
        filtered["probing_jitter"] = float(filtered["probing_jitter"])

    if "probing_widen_only" in filtered:
        filtered["probing_widen_only"] = bool(filtered["probing_widen_only"])

    # -------------------------------------------------------------
    # Valeurs par défaut si certaines clés manquent
    # -------------------------------------------------------------
    filtered.setdefault("dt", 1.0)
    filtered.setdefault("T", 20000.0)
    filtered.setdefault("mid0", 100.0)
    filtered.setdefault("sigma", 0.02)
    filtered.setdefault("A", 1.2)
    filtered.setdefault("k", 1.0)
    filtered.setdefault("base_spread", 0.2)
    filtered.setdefault("phi", 0.0)
    filtered.setdefault("order_size", 0.01)
    filtered.setdefault("fee_bps", 0.0)
    filtered.setdefault("seed", 0)

    # Construction finale de l’objet MMParams
    return MMParams(**filtered)



def main() -> None:
    """
    Fonction principale du script.

    Elle :
    - charge la config probing,
    - charge la baseline,
    - construit la variante probing,
    - lance baseline et probing,
    - estime A et k pour les deux,
    - compare les résultats,
    - trace un graphique de comparaison de k_hat.
    """

    # -------------------------------------------------------------
    # Étape 1 : lecture de la configuration probing
    # -------------------------------------------------------------
    pcfg = yaml.safe_load(Path("configs/probing.yaml").read_text(encoding="utf-8"))

    # Chemin vers la configuration baseline de base
    base_cfg_path = pcfg["base_config"]

    # Chargement des paramètres baseline
    base = load_mm_params(base_cfg_path)

    # -------------------------------------------------------------
    # Étape 2 : construction de la variante probing
    # -------------------------------------------------------------

    # Bloc probing du YAML
    probing_cfg = pcfg["probing"]

    # Noms des champs autorisés
    allowed = _mmparams_field_names()

    # Dictionnaire des champs à modifier par rapport à la baseline
    updates: dict[str, object] = {}

    if "probing_p" in allowed:
        updates["probing_p"] = float(probing_cfg["p_explore"])

    if "probing_jitter" in allowed:
        updates["probing_jitter"] = float(probing_cfg["jitter"])

    if "probing_widen_only" in allowed:
        updates["probing_widen_only"] = bool(probing_cfg["widen_only"])

    """
    MMParams est frozen/immutable.

    Donc on ne modifie pas base directement.
    On crée un nouvel objet "probing" en partant de base
    et en remplaçant seulement les champs utiles.
    """
    probing = replace(base, **updates) if updates else copy.deepcopy(base)

    # -------------------------------------------------------------
    # Étape 3 : lecture des paramètres de calibration
    # -------------------------------------------------------------
    calib_cfg = pcfg["calibration"]
    kmin, kmax = calib_cfg["k_bounds"]
    grid_size = int(calib_cfg["grid_size"])

    # -------------------------------------------------------------
    # Étape 4 : comparaison baseline vs probing
    # -------------------------------------------------------------
    runs: list[tuple[str, float, float, float]] = []

    for name, params in [("baseline", base), ("probing", probing)]:

        # Lancement du backtest
        df = run_mm_toy(params)

        # Construction du dataset de calibration
        delta, n = build_intensity_dataset_from_mm(df, dt=params.dt)

        # Estimation MLE de A et k
        est = fit_intensity_exp_mle(
            delta,
            n,
            dt=params.dt,
            k_bounds=(float(kmin), float(kmax)),
            grid_size=grid_size,
        )

        # Stockage des résultats
        runs.append((name, float(est.A), float(est.k), float(est.nll)))

    # -------------------------------------------------------------
    # Étape 5 : affichage terminal
    # -------------------------------------------------------------

    # Paramètres vrais utilisés dans la simulation
    print("=== True ===")
    print(f"A_true={base.A:.4f}, k_true={base.k:.4f}")

    # Paramètres estimés pour baseline et probing
    print("=== Estimates ===")
    for name, Ahat, khat, nll in runs:
        print(f"{name:8s}  A_hat={Ahat:.4f}  k_hat={khat:.4f}  nll={nll:.1f}")

    # -------------------------------------------------------------
    # Étape 6 : tracé du graphique de comparaison
    # -------------------------------------------------------------
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Récupération des labels et des k_hat
    labels = [r[0] for r in runs]
    ks = [r[2] for r in runs]

    # Bar plot
    plt.figure()
    plt.bar(labels, ks)

    # Ligne horizontale au niveau du vrai k
    plt.axhline(base.k, linestyle="--")

    plt.title("k_hat: baseline vs probing")
    plt.ylabel("k_hat")
    plt.tight_layout()
    plt.savefig("reports/figures/probing_khat_comparison.png")
    plt.close()

    # Message de confirmation
    print("Saved reports/figures/probing_khat_comparison.png")


# Point d’entrée standard du script
if __name__ == "__main__":
    main()