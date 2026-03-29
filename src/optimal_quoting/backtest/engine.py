"""
Ce script permet de simuler un market maker dans le temps.

À chaque pas de temps :

- le prix mid évolue,
- une stratégie choisit des quotes bid et ask,
- on calcule les intensités d’exécution à partir des deltas,
- on simule si un fill a lieu côté bid ou côté ask,
- on met à jour l’inventaire, le cash et l’equity,
- on stocke toute la trajectoire.

C’est donc le moteur principal de backtest du projet.

Il relie directement :

    stratégie -> intensité -> fills -> inventory / cash -> performance
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Stratégie Avellaneda–Stoikov
from optimal_quoting.strategy.avellaneda_stoikov import ASStrategyConfig, compute_as_quotes

# Stratégie probing
from optimal_quoting.strategy.probing import ProbingConfig, compute_probing_quotes

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Modèle d’intensité exponentielle
from optimal_quoting.model.intensity import intensity_exp

# Simulation d’un événement à partir d’une intensité
from optimal_quoting.sim.poisson import event_happens

# Stratégie baseline
from optimal_quoting.strategy.quotes import compute_quotes



"""
Cette classe contient tous les paramètres nécessaires au backtest.
"""
@dataclass(frozen=True)
class MMParams:
    # Pas de temps
    dt: float

    # Horizon total de la simulation
    T: float

    # Mid initial
    mid0: float

    # Volatilité par pas de temps du mid dans la simulation toy
    sigma: float

    # Paramètres de l’intensité : lambda(delta) = A * exp(-k * delta)
    A: float
    k: float

    # Paramètres de la stratégie baseline
    base_spread: float
    phi: float

    # Taille d’un ordre exécuté
    order_size: float

    # Frais de transaction en basis points
    fee_bps: float

    # Seed aléatoire pour la reproductibilité
    seed: int = 42

    # Paramètres de probing
    probing_p: float = 0.0
    probing_jitter: float = 0.0
    probing_widen_only: bool = True

    # Politique utilisée : baseline, probing ou as
    policy: str = "baseline"

    # Aversion au risque de Stoikov
    gamma: float = 0.1



def run_mm_toy(p: MMParams) -> pd.DataFrame:
    """
    Fonction principale de simulation.

    Entrée :
    --------
    p : ensemble des paramètres du market maker

    Sortie :
    --------
    Un DataFrame contenant toute la trajectoire simulée :
    - temps
    - mid
    - inventory
    - cash
    - equity
    - bid
    - ask
    - fills côté bid et ask
    """

    # Générateur aléatoire
    rng = np.random.default_rng(p.seed)

    # Nombre total d’étapes
    n = int(p.T / p.dt) + 1

    # Tableau du mid
    mid = np.empty(n, dtype=float)
    mid[0] = p.mid0

    # Etat initial du market maker
    q = 0.0
    cash = 0.0

    # Conversion des basis points en taux
    fee = p.fee_bps * 1e-4

    # Liste qui stockera la trajectoire complète
    rows = []

    # -------------------------------------------------------------
    # Boucle principale dans le temps
    # -------------------------------------------------------------
    for t in range(n):

        """
        Mise à jour du mid.

        On utilise ici une dynamique très simple :
        le mid courant vaut le mid précédent
        plus un bruit gaussien centré de volatilité sigma.
        """
        if t > 0:
            mid[t] = max(0.01, mid[t - 1] + rng.normal(0.0, p.sigma))

        # Mid courant
        m = float(mid[t])

        # ---------------------------------------------------------
        # Choix de la stratégie et calcul des quotes
        # ---------------------------------------------------------

        """
        Si le probing est activé, on entre dans le bloc spécial.
        Sinon, on utilise la baseline.
        """
        if p.probing_p > 0.0 and p.probing_jitter > 0.0:

            # Configuration du probing
            qcfg = ProbingConfig(
                p_explore=p.probing_p,
                jitter=p.probing_jitter,
                widen_only=p.probing_widen_only,
            )

            # Temps courant en secondes
            t_now = t * p.dt

            # Stratégie Avellaneda–Stoikov
            if p.policy == "as":
                quotes = compute_as_quotes(
                    mid=m,
                    q=q,
                    t=t_now,
                    T=p.T,
                    sigma=p.sigma,
                    k=p.k,
                    cfg=ASStrategyConfig(gamma=p.gamma),
                )

            # Stratégie probing
            elif p.policy == "probing" and p.probing_p > 0.0 and p.probing_jitter > 0.0:
                qcfg = ProbingConfig(
                    p_explore=p.probing_p,
                    jitter=p.probing_jitter,
                    widen_only=p.probing_widen_only,
                )
                quotes = compute_probing_quotes(m, q, p.base_spread, p.phi, qcfg, rng)

            # Sinon, baseline
            else:
                quotes = compute_quotes(m, q, p.base_spread, p.phi)

        else:
            # Si probing désactivé, baseline
            quotes = compute_quotes(m, q, p.base_spread, p.phi)

        # ---------------------------------------------------------
        # Intensités d’exécution
        # ---------------------------------------------------------

        """
        On transforme les deltas choisis par la stratégie
        en intensités d’exécution.

        Plus un delta est petit,
        plus l’intensité est grande.
        """
        lam_bid = intensity_exp(p.A, p.k, quotes.delta_bid)
        lam_ask = intensity_exp(p.A, p.k, quotes.delta_ask)

        # ---------------------------------------------------------
        # Simulation des fills
        # ---------------------------------------------------------

        """
        On simule si un événement a lieu côté bid et côté ask.

        Chaque fill est une variable booléenne :
        - True si exécution
        - False sinon
        """
        fill_bid = event_happens(lam_bid, p.dt, rng)
        fill_ask = event_happens(lam_ask, p.dt, rng)

        # ---------------------------------------------------------
        # Mise à jour du cash et de l’inventaire
        # ---------------------------------------------------------

        """
        Si le bid est exécuté :
        le market maker achète.
        """
        if fill_bid:
            q += p.order_size
            cash -= quotes.bid * p.order_size
            cash -= fee * quotes.bid * p.order_size

        """
        Si le ask est exécuté :
        le market maker vend.
        """
        if fill_ask:
            q -= p.order_size
            cash += quotes.ask * p.order_size
            cash -= fee * quotes.ask * p.order_size

        # ---------------------------------------------------------
        # Calcul de l’equity
        # ---------------------------------------------------------

        """
        L’equity est la valeur totale du portefeuille :

            equity = cash + inventory * mid
        """
        equity = cash + q * m

        # ---------------------------------------------------------
        # Stockage de la ligne courante
        # ---------------------------------------------------------
        rows.append(
            (
                t * p.dt,
                m,
                q,
                cash,
                equity,
                quotes.bid,
                quotes.ask,
                fill_bid,
                fill_ask,
            )
        )

    # -------------------------------------------------------------
    # Retour final : DataFrame complet
    # -------------------------------------------------------------
    return pd.DataFrame(
        rows,
        columns=[
            "time_s",
            "mid",
            "inventory",
            "cash",
            "equity",
            "bid",
            "ask",
            "fill_bid",
            "fill_ask",
        ],
    )