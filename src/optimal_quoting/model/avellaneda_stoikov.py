""" 
Ce script implémente une stratégie de market making théorique optimale. 
C'est la stratégie de référence du projet. 
Elle répond à la question : comment choisir mes quotes (bid/ask) en fonction de :
- mon inventaire,
- le temps restant,
- le risque,
- la dynamique du marché.
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
import math



""" 
La classe suivante définit le conteneur de paramètres du modèle. 
"""



@dataclass(frozen=True)
class ASParams:
    
    
    
    """
    Paramètres de cotation Avellaneda–Stoikov (forme réduite)

    On suppose :
    
      - dynamique du mid : dS ~ sigma dW (discrétisé ailleurs)
      (c'est du Brownian Motion. Ça veut dire que le prix bouge aléatoirement, avec volatilité sigma
      donc si sigma est grand, alors le marché est instable, sinon il est calme)
      
      - intensité d'exécution : lambda(delta) = A exp(-k delta)
      
      - utilité : exponentielle avec aversion au risque gamma

    Sachant que le delta_ask est la distance optimale entre le ask que le trader va poser et le mid actuel, 
    les deltas canoniques en forme fermée (réduite) donnés par le modèle sont :
    
      - delta_ask = 1/k + (gamma * sigma^2 * tau / 2) * q
      
      - delta_bid = 1/k - (gamma * sigma^2 * tau / 2) * q

    où tau = (T - t) est le temps restant jusqu'à l'horizon (>= 0).
    """
    
    
    
    """ 
    gamma : aversion au risque
    - grand gamma → très prudent
    - petit gamma → agressif
    """
    gamma: float
    
    
    
    """ 
    sigma : volatilité du prix 
    - mesure à quel point le prix bouge 
    - plus sigma est grand, plus le risque d'inventaire est dangereux
    """
    sigma: float
    
    
    
    """ 
    k : paramètre de l'intensité
    - contrôle la décroissance des fills 
    """
    k: float         
    
    
    
    """ 
    T : horizon final (en secondes)
    - moment où on veut être "flat" (pas d'inventaire)
    """
    T: float         



""" 
La fonction ci-dessous retourne les distances optimales au mid 
"""



def as_deltas(q: float, t: float, p: ASParams) -> tuple[float, float]:
    
    if p.gamma <= 0:
        raise ValueError("gamma must be > 0")
    if p.sigma < 0:
        raise ValueError("sigma must be >= 0")
    if p.k <= 0:
        raise ValueError("k must be > 0")
    if p.T <= 0:
        raise ValueError("T must be > 0")
    if t < 0 or t > p.T:
        raise ValueError("t must be in [0, T]")

    # temps restant jusqu'à l'horizon
    tau = p.T - t 
    
    # spread "structurel"
    base = 1.0 / p.k
    
    # effet inventaire + risque 
    skew = 0.5 * p.gamma * (p.sigma ** 2) * tau * q

    # On empêche les valeurs négatives 
    delta_bid = max(0.0, base - skew)
    delta_ask = max(0.0, base + skew)

    return delta_bid, delta_ask



""" 
La fonction suivante transforme les deltas en vrais prix 
"""



def as_quotes(mid: float, q: float, t: float, p: ASParams) -> tuple[float, float, float, float]:
    d_bid, d_ask = as_deltas(q=q, t=t, p=p)
    bid = mid - d_bid
    ask = mid + d_ask
    return bid, ask, d_bid, d_ask
