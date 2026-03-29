""" 
Ce script permet de répondre à la question : est-ce qu'un trade se produit ou pas ? 
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

import math
import numpy as np



""" 
La fonction ci-dessous renvoie True si un trade a lieu, et False si rien ne se passe 

lambda correspond à l'intensité, donc à la vitesse à laquelle les trades sont réalisés 
"""



def event_happens(lmbda: float, dt: float, rng: np.random.Generator) -> bool:
    
    
    
    """
    Poisson arrival within dt:
        P(event) = 1 - exp(-λ dt)
    """
    
    
    
    # l'intensité doit toujours être positive 
    if lmbda < 0:
        raise ValueError("lambda must be >= 0")
    
    # le petit intervalle de temps aussi 
    if dt <= 0:
        raise ValueError("dt must be > 0")
    
    # probabilité qu'un événement arrive dans dt 
    p = 1.0 - math.exp(-lmbda * dt)
    
    return bool(rng.random() < p)
