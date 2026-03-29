""" 
Ce script seert à charger des configurations d'expériences. 

Les paramètres du projet (intensité, horizon, paramètres de probing, etc.) ne sont pas codés en dur
dans le Python, mais stockés dans des fichiers .yaml

La fonction de ce script permet de lire ces fichiers YAML et de les transformer en dictionnaires Python. 
"""



# Pour éviter certains problèmes lorsque des types sont référencés avant d'être complètement définis
from __future__ import annotations

# On utilise Path pour manipuler des chemins de fichiers de manière propre
from pathlib import Path

# on importe Any, qui est un type générique (peut contenir n'importe quoi)
from typing import Any, Dict

# On utilise la bibliothèque YAML
import yaml



""" 
La fonction suivante fait plusieurs choses : 

Path(path).read_text(encoding="utf-8") :

- Elle ouvre le fichier situé à Path
- Elle lit tout son contenu sous forme de texte
Donc on obtient une chaîne de caractère contenant le YAML. 
Par exemple : 
alpha: 0.5
beta: 1.2
devient : 
{"alpha": 0.5, "beta": 1.2}

safe_load : 

YAML peut contenir du code potentiellement dangereux. Donc safe_load : 
- évite d'exécuter du code arbitraire
- ne charge que des structures simples

or {} : 

Si le YAML est vide, alors la fonction renvoie {}. 
Donc on garantit que la fonction renvoie toujours un dictionnaire, et jamais None. 
"""



def load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
