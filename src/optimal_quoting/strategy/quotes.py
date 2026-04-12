""" 
Ce script donne la stratégie la plus simple. 

La fonction de ce script dit comment choisir bid/ask sans Stoïkov.

On part d'un spread fixe, puis on l'ajuste avec l'inventaire. 

La formule : 
    - 𝛿𝑎s𝑘 = 𝑠/2 + 𝜙𝑞
    - δbid = s/2 - ϕq
avec : 
    - s = base_spread
    - q = inventaire
    - ϕ = force de correction
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quotes:
    bid: float
    ask: float
    delta_bid: float
    delta_ask: float


def compute_quotes(mid: float, q: float, base_spread: float, phi: float) -> Quotes:
    """
    Baseline quoting with inventory skew:
        δa = s/2 + φ q
        δb = s/2 - φ q
    where s = base_spread.

    q > 0 means long inventory -> widen ask (sell faster), tighten bid (buy slower).
    """
    if base_spread < 0:
        raise ValueError("base_spread must be >= 0")

    # Moitié du spread 
    half = 0.5 * base_spread
    
    # On ajoute un skew basé sur l'inventaire 
    delta_ask = max(0.0, half + phi * q)
    delta_bid = max(0.0, half - phi * q)
    
    # Transformation en prix 
    ask = mid + delta_ask
    bid = mid - delta_bid
    
    # On renvoie : bid, ask, delta_bid, delta_ask
    return Quotes(bid=bid, ask=ask, delta_bid=delta_bid, delta_ask=delta_ask)
