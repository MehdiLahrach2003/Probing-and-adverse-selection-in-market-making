"""
Ce script définit une fonction utilitaire pour créer un logger propre.

Un logger permet d’afficher des messages structurés dans le terminal,
par exemple :

    [INFO] optimal_quoting - Simulation started
    [WARNING] optimal_quoting - Low liquidity detected

L’objectif est :
- d’avoir des messages lisibles,
- d’éviter les doublons,
- de centraliser la configuration du logging.
"""


# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# On utilise le module standard logging de Python
import logging as py_logging



def get_logger(name: str = "optimal_quoting", level: int = py_logging.INFO) -> py_logging.Logger:
    """
    Cette fonction retourne un logger configuré pour le projet.

    Paramètres :
    ------------
    name : nom du logger (utile pour identifier la source des messages)
    level : niveau de log (INFO, DEBUG, WARNING, etc.)

    Retour :
    --------
    Un objet logger prêt à être utilisé.
    """

    # On récupère (ou crée) un logger avec ce nom
    logger = py_logging.getLogger(name)

    """
    Si le logger a déjà des handlers,
    cela signifie qu’il a déjà été configuré.

    Dans ce cas :
    - on met simplement à jour le niveau,
    - on évite d’ajouter un nouveau handler,
      sinon les messages seraient affichés plusieurs fois.
    """
    if logger.handlers:
        logger.setLevel(level)
        return logger

    # On définit le niveau du logger (INFO par défaut)
    logger.setLevel(level)

    """
    On crée un handler qui envoie les logs vers la sortie standard (terminal).
    """
    handler = py_logging.StreamHandler()

    """
    On définit le format des messages :

    - levelname : INFO, WARNING, etc.
    - name : nom du logger
    - message : contenu du message
    """
    fmt = py_logging.Formatter("[%(levelname)s] %(name)s - %(message)s")

    # On applique ce format au handler
    handler.setFormatter(fmt)

    # On ajoute le handler au logger
    logger.addHandler(handler)

    """
    On désactive la propagation pour éviter que les messages
    soient envoyés vers d’autres loggers parents,
    ce qui pourrait créer des doublons.
    """
    logger.propagate = False

    # On retourne le logger prêt à l’emploi
    return logger