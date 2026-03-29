""" 
Ce script modélise à quelle vitesse les ordres sont exécutés en fonction de la quote. 

La fonction de ce script implémente : λ(δ)=A×exp(−kδ) où : 

- δ (delta) = distance entre la quote et le mid
- A = intensité maximale (quand δ = 0) 
- k = vitesse de décroissance
- λ(δ) = intensité d’exécution

Toute la stratégie de market making repose sur ça : choisir δ optimal

Le probing permet d’explorer plusieurs δ et donc de reconstruire λ(δ). 
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# Pour utiliser exp pour l'exponentielle
import math


def intensity_exp(A: float, k: float, delta: float) -> float:
    
    # Une intensité doit être positive 
    if A <= 0:
        raise ValueError("A must be > 0")
    
    # Sinon la fonction ne décroît pas correctement 
    if k <= 0:
        raise ValueError("k must be > 0")
    
    # une distance ne peut pas être négative
    if delta < 0:
        raise ValueError("delta must be >= 0")
    
    return A * math.exp(-k * delta)
