""" 
Ce script définit la stratégie probing.

L’idée est simple :

On part de la stratégie baseline,
puis, de temps en temps,
on modifie volontairement les deltas,
pour explorer d’autres quotes que celles qu’on aurait choisies naturellement.

Donc ici, le probing, c’est vraiment : ajouter de l’aléatoire contrôlé aux quotes.
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# On importe la stratégie baseline car le probing repose sur la baseline
from optimal_quoting.strategy.quotes import Quotes, compute_quotes



""" 
La classe suivante contient les paramètres du probing 
"""



@dataclass(frozen=True)
class ProbingConfig:
    
    """
    La probabilité d'explorer à chaque étape
    Par exemple : p_explore = 0.2 veut dire :
    - vingt pour cent du temps, on modifie les quotes,
    - quatre-vingts pour cent du temps, on garde la baseline.
    """
    p_explore: float          
    
    """ 
    L’amplitude maximale de la perturbation ajoutée aux deltas.
    Donc plus jitter est grand, plus l’exploration est forte.
    """
    jitter: float             
    
    """ 
    Si True :
    - on ne fait qu’augmenter les deltas,
    - donc on éloigne les quotes du mid.
    Si False :
    - on peut soit augmenter, soit diminuer les deltas.
    Donc :
    - True = exploration prudente,
    - False = exploration plus large.
    """
    widen_only: bool = True   


def compute_probing_quotes(
    mid: float,
    q: float,
    base_spread: float,
    phi: float,
    cfg: ProbingConfig,
    rng: np.random.Generator,
) -> Quotes:
    
    
    
    """
    Baseline quotes + randomized probing to diversify deltas for calibration.

    - With prob p_explore, we perturb deltas by jitter.
    - widen_only=True keeps perturbations non-negative and only increases deltas.
    """
    
    
    
    """ 
    Contrôles standards :
    - une probabilité doit être entre zéro et un,
    - le jitter doit être positif
    """
    if not (0.0 <= cfg.p_explore <= 1.0):
        raise ValueError("p_explore must be in [0,1]")
    if cfg.jitter < 0:
        raise ValueError("jitter must be >= 0")


    # On calcule d’abord la baseline
    # Baseline d’abord, puis perturbation éventuelle. Donc la baseline est le point de départ.
    base = compute_quotes(mid, q, base_spread, phi)



    """ 
    on tire un nombre aléatoire entre zéro et un,
    si ce nombre est plus grand que p_explore, on n’explore pas,
    ou si jitter = 0, il n’y a rien à faire.

    Donc dans ce cas on garde exactement la quote baseline.
    """
    if rng.random() >= cfg.p_explore or cfg.jitter == 0.0:
        return base



    """ 
    Si widen_only = True, alors on ajoute un bruit uniforme entre zéro et jitter.
    Donc : 
    - eps_b est positif,
    - eps_a est positif.
    Par conséquent :
    - delta_bid augmente,
    - delta_ask augmente.
    Donc les quotes s’éloignent du mid.
    C’est une exploration qui élargit seulement.
    """
    
    """ 
    Si widen_only = False, alors on ajoute un bruit uniforme entre -jitter et jitter.
    Donc : 
    - parfois on élargit,
    - parfois on resserre.
    C’est une exploration plus riche.
    """
    
    if cfg.widen_only:
        # Only widen deltas: add U(0, jitter)
        eps_b = rng.random() * cfg.jitter
        eps_a = rng.random() * cfg.jitter
        
    else:
        eps_b = (2.0 * rng.random() - 1.0) * cfg.jitter
        eps_a = (2.0 * rng.random() - 1.0) * cfg.jitter



    """ 
    On prend les deltas baseline, puis on ajoute la perturbation.
    Le max(0.0, ...) évite d’avoir des deltas négatifs.
    """
    delta_bid = max(0.0, base.delta_bid + eps_b)
    delta_ask = max(0.0, base.delta_ask + eps_a)

    # Conversion en pri
    bid = mid - delta_bid
    ask = mid + delta_ask

    return Quotes(bid=bid, ask=ask, delta_bid=delta_bid, delta_ask=delta_ask)