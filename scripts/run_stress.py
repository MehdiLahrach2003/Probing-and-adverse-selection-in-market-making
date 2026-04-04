"""
Ce script lance une campagne de stress tests sur les stratégies de market making.

L’idée est de faire varier certains paramètres du marché simulé,
en particulier :

- A : niveau de l’intensité
- k : vitesse de décroissance de l’intensité

et d’évaluer plusieurs stratégies sur plusieurs seeds.

Pour chaque combinaison :
- on construit les paramètres du scénario,
- on lance un backtest,
- on résume les performances,
- on sauvegarde le tout dans un CSV final.

Ce script sert donc à tester la robustesse des stratégies
quand le régime de marché change.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Outils d’introspection du module engine
import inspect

# Produit cartésien pour explorer toutes les combinaisons
import itertools

# Gestion des chemins
from pathlib import Path

# Annotations de type
from typing import Callable, Optional

# Outils numériques et tableaux
import numpy as np
import pandas as pd

# Lecture YAML
import yaml

# Paramètres du market maker
from optimal_quoting.backtest.engine import MMParams

# Import du module engine complet pour chercher dynamiquement la fonction de run
import optimal_quoting.backtest.engine as engine_mod



def load_cfg(path: str) -> dict:
    """
    Lit un fichier YAML et retourne un dictionnaire Python.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)



def make_params(cfg: dict, A: float, k: float, seed: int, policy: str) -> MMParams:
    """
    Construit un objet MMParams pour un scénario de stress donné.

    Les paramètres fixes viennent de cfg,
    tandis que A, k, seed et policy sont injectés dynamiquement.
    """
    return MMParams(
        dt=float(cfg["dt"]),
        T=float(cfg["T"]),
        mid0=float(cfg["mid0"]),
        sigma=float(cfg["sigma"]),

        # Paramètres d’intensité du scénario courant
        A=float(A),
        k=float(k),

        # Paramètres de stratégie de base
        base_spread=float(cfg["base_spread"]),
        phi=float(cfg["phi"]),
        order_size=float(cfg["order_size"]),

        # Coûts
        fee_bps=float(cfg["fee_bps"]),

        # Seed et stratégie
        seed=int(seed),
        policy=str(policy),

        # Paramètres probing éventuels
        probing_p=float(cfg["probing_p"]),
        probing_jitter=float(cfg["probing_jitter"]),
        probing_widen_only=bool(cfg["probing_widen_only"]),

        # Paramètre AS éventuel
        gamma=float(cfg["gamma"]),
    )



def _find_engine_runner() -> Callable[[MMParams], pd.DataFrame]:
    """
    Cherche automatiquement dans optimal_quoting.backtest.engine
    une fonction de simulation compatible avec :

        f(p: MMParams) -> pd.DataFrame

    Cela rend le script plus robuste à de petites différences de nommage.
    """

    # -------------------------------------------------------------
    # Étape 1 : essayer des noms classiques
    # -------------------------------------------------------------
    preferred = [
        "run_mm_toy",
        "run_mm",
        "run_backtest",
        "run",
        "simulate",
        "simulate_mm",
    ]

    for name in preferred:
        fn = getattr(engine_mod, name, None)
        if callable(fn):
            return lambda p, _fn=fn: _fn(p)

    # -------------------------------------------------------------
    # Étape 2 : scanner toutes les fonctions du module
    # -------------------------------------------------------------
    candidates: list[Callable] = []

    for name, obj in vars(engine_mod).items():
        if not callable(obj):
            continue
        if name.startswith("_"):
            continue

        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue

        params = list(sig.parameters.values())
        if not params:
            continue

        # Le premier argument doit être positionnel
        first = params[0]
        if first.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            continue

        # Tous les autres arguments doivent être optionnels
        if any(p.default is inspect._empty and i > 0 for i, p in enumerate(params)):
            continue

        candidates.append(obj)

    # -------------------------------------------------------------
    # Étape 3 : fallback
    # -------------------------------------------------------------
    def _try(fn: Callable) -> Optional[Callable[[MMParams], pd.DataFrame]]:
        def _runner(p: MMParams) -> pd.DataFrame:
            return fn(p)
        return _runner

    if candidates:
        return _try(candidates[0])  # type: ignore[arg-type]

    available = sorted([k for k, v in vars(engine_mod).items() if callable(v) and not k.startswith("_")])
    raise ImportError(
        "Could not find a runnable backtest function in optimal_quoting.backtest.engine.\n"
        "Expected a callable like run_mm(p: MMParams) -> pd.DataFrame.\n"
        f"Callable symbols found: {available}"
    )



def _pick_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    """
    Cherche dans le DataFrame la première colonne existante
    parmi une liste de noms candidats.
    """
    cols = set(df.columns)
    for n in names:
        if n in cols:
            return n
    return None



def summarize_df(df: pd.DataFrame) -> dict:
    """
    Résumé robuste d’un DataFrame de backtest.

    Le script essaie de détecter automatiquement :
    - la colonne PnL / equity
    - la colonne inventory / position
    """

    # Colonnes candidates pour le PnL
    pnl_col = _pick_col(df, ["pnl", "PnL", "equity", "wealth", "cash"])

    # Colonnes candidates pour l’inventaire
    inv_col = _pick_col(df, ["inv", "inventory", "position", "q"])

    out: dict = {}

    # -------------------------------------------------------------
    # Résumé du PnL
    # -------------------------------------------------------------
    if pnl_col is not None:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna().to_numpy()
        if pnl.size:
            out["pnl_final"] = float(pnl[-1])
            out["pnl_mean"] = float(np.mean(np.diff(pnl))) if pnl.size > 1 else float(pnl[-1])
            out["pnl_std"] = float(np.std(np.diff(pnl))) if pnl.size > 1 else 0.0
    else:
        out["pnl_final"] = np.nan
        out["pnl_mean"] = np.nan
        out["pnl_std"] = np.nan

    # -------------------------------------------------------------
    # Résumé de l’inventaire
    # -------------------------------------------------------------
    if inv_col is not None:
        inv = pd.to_numeric(df[inv_col], errors="coerce").dropna().to_numpy()
        if inv.size:
            out["inv_std"] = float(np.std(inv))
            out["inv_max_abs"] = float(np.max(np.abs(inv)))
    else:
        out["inv_std"] = np.nan
        out["inv_max_abs"] = np.nan

    # Nombre de lignes du DataFrame, utile comme sanity check
    out["n_rows"] = int(len(df))
    return out



def main() -> None:
    """
    Fonction principale du stress test.

    Elle :
    - lit stress.yaml,
    - cherche la fonction moteur,
    - boucle sur toutes les combinaisons A, k, seed, policy,
    - lance les simulations,
    - résume les résultats,
    - sauvegarde un CSV final.
    """

    # Lecture de la config stress
    cfg = load_cfg("configs/stress.yaml")

    # Chemin de sortie du CSV
    out_path = Path(cfg["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Recherche dynamique de la fonction moteur
    run_engine = _find_engine_runner()

    # Liste des résultats
    rows = []

    # Produit cartésien sur tous les scénarios
    for A, k, seed, policy in itertools.product(cfg["A_grid"], cfg["k_grid"], cfg["seeds"], cfg["policies"]):
        # Construction des paramètres du scénario
        p = make_params(cfg, A=A, k=k, seed=seed, policy=policy)

        # Lancement du backtest
        df = run_engine(p)

        # Résumé des performances
        summary = summarize_df(df)

        # Stockage des résultats
        rows.append({"A": A, "k": k, "seed": seed, "policy": policy, **summary})

    # Construction du tableau final
    res = pd.DataFrame(rows)

    # Sauvegarde du CSV
    res.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


# Point d’entrée standard du script
if __name__ == "__main__":
    main()