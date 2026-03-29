""" 
Ce script permet de définir le vocabulaire minimal utilisé partout ailleurs dans le projet. 
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass

# Optional pour dire qu'une valeur peut être soit d'un certain type, soit None.
from typing import Dict, Optional



""" 
La classe suivante représente une quote.

On utilise Optional parce que parfois la stratégie peut décider : 
- de ne pas afficher le bid ; 
- ou de ne pas afficher le ask. 

Par exemple : 
- Quote(bid=99.8, ask=100.2) signifie que je côte des deux côtés
- Quote(bid=None, ask=100.2) signifie que je ne veux pas acheter, je veux seulement vendre

Le (frozen = True) signifie que une fois on crée une classe, on ne peux plus modifier ses paramètres
"""



@dataclass(frozen=True)
class Quote:
    
    # Le prix actuel auquel il est prêt à acheter
    bid: Optional[float]
    
    # Le prix actuel auquel il est prêt à vendre
    ask: Optional[float]



""" 
Cette classe représente l'état minimal d'une stratégie à un instant donné. 

Cette classe est importante parce qu'une stratégie de quotation a besoin d'un état d'entrée. 

En général, une stratégie ne décide pas ses quotes "dans le vide". Elle regarde au moins : le temps, l'inventaire et le cash.
"""



@dataclass(frozen=True)
class StrategyState:
    
    # Le temps courant 
    t: float
    
    # L'inventaire courant 
    inventory: float
    
    # La trésorerie courante 
    cash: float



""" 
On définit le type alias StrategyOutput. 

Cela veut dire qu'une StrategyOutput est simplement un dictionnaire dont les clés sont des chaînes de caractère
et les valeurs sont des flottants. 
"""



StrategyOutput = Dict[str, float]
