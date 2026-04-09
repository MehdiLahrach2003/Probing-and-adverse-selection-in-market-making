"""
Ce script définit les structures de données de base pour représenter
des informations de marché.

On y trouve deux objets principaux :

- Trade : une transaction exécutée
- TopOfBook : l’état du carnet au meilleur bid/ask

L’objectif est de structurer proprement les données,
comme on le ferait dans un système réel de trading.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal



"""
Définition du type Side.

Un trade peut être soit :
- "buy"
- "sell"

Cela permet d’éviter les erreurs de type.
"""
Side = Literal["buy", "sell"]



@dataclass(frozen=True)
class Trade:
    """
    Représente une transaction exécutée sur le marché.

    Attributs :
    -----------
    ts : timestamp du trade
    price : prix d’exécution
    size : quantité échangée
    side : direction du trade (achat ou vente)
    """

    ts: datetime
    price: float
    size: float
    side: Side



@dataclass(frozen=True)
class TopOfBook:
    """
    Représente l’état du carnet d’ordres au meilleur niveau.

    Attributs :
    -----------
    ts : timestamp
    bid : meilleur prix d’achat
    ask : meilleur prix de vente
    bid_size : quantité disponible au bid (optionnel)
    ask_size : quantité disponible au ask (optionnel)
    """

    ts: datetime
    bid: float
    ask: float

    # Les tailles peuvent être absentes (None)
    bid_size: float | None = None
    ask_size: float | None = None


    def mid(self) -> float:
        """
        Calcule le prix médian.

        Formule :
            mid = (bid + ask) / 2

        C’est une approximation du prix "juste" du marché.
        """
        return 0.5 * (self.bid + self.ask)


    def spread(self) -> float:
        """
        Calcule le spread.

        Formule :
            spread = ask - bid

        Cela représente le coût implicite de transaction.
        """
        return self.ask - self.bid