# search engine


Sorbonne Université

M1 DAC

Année Universitaire 2019/2020


Module: Recherche d'information


## Prérequis:
```bash
pip install -r requirements.txt
```

## Structure:
### data:
Contient les données. 
### index_files:
Contient les fichiers d'indexation.
### src:
Le sous-repertoire `Indexation` contient les classes relatives à l'indexation et le parsing de la collection de documents. (TME1)

Le sous-repertoire `Appariemment` contient le code des modèles d'appariement. (TME2)

Le sous-repertoire `Evaluation` contient le code des mesures utilisées pour l'évaluation des modèles d'appariement. (TME3) 

Les bonus 2.4 (TME2) et 2.6 et 2.7 (TME3) ont été traités.

### tests:

notebooks contenant les tests effectués.

`test_Indexation.ipynb` : Indexation et parsing des documents.

`test_QueryParsing.ipynb` : Parsing des requetes.

`test_Models.ipynb` : Modèles d'appariement et mesures d'évaluation.

`test_QueryParsing.ipynb` : optimisation des hyper-paramètres du modèle de langue et OkapiBM25.


