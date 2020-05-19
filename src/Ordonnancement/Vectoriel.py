import sys
sys.path.append(".")
from abc import abstractmethod
import numpy as np

from Evaluation.Mesure import AveragePrecision
from Indexation.Parser import *
from Indexation.Index import *
from .Weighter import *
from .IRModel import *

class Vectoriel(IRModel):
    """
    Modèle Vectoriel.
    ----------------------------------------------------
    Parameters:
        - weighter : Objet Weighter
        - normalized : booléen permettant de définir la fonction de score, si vrai score cosine sinon produit scalaire
    """
    name = "Vectoriel"
    def __init__(self, index, weighter, normalized=False):
        super().__init__(index)
        self.weighter = weighter
        self.normalized = normalized
        # on calcule les normes des documents ici pour ne pas le faire à chaque requête
        self.d_weights = {idoc : weighter.getWeightsForDoc(idoc) for idoc in index.docs.keys()}
        self.d_norm = {idoc : np.linalg.norm(np.array(list(doc.values()))) for (idoc, doc) in self.d_weights.items()}

    

    def normalize(self, d):
        norm = float(np.sum(d.values()))
        return {key: value / norm for (key, value) in d.items()}

    def getScores(self, query):
        scores = {}
        weights = self.weighter.getWeightsForQuery(query)
        
        if self.normalized:
            normQ = np.linalg.norm(np.array(list(weights.values())))
            for (idoc, weight_d) in self.d_weights.items():
                scores[idoc] = 0
                for (term, t_weight) in weights.items():
                    if term in weight_d:
                        scores[idoc] = scores.get(idoc,0) + (weight_d[term] * t_weight)
                scores[idoc] = scores.get(idoc,0) / (normQ * d_norm[idoc])
        else:
            for (idoc, weight_d) in self.d_weights.items():
                scores[idoc] = 0
                for (term, t_weight) in weights.items():
                    if term in weight_d:
                        scores[idoc] = scores.get(idoc,0) + (weight_d[term] * t_weight)
                
        return scores
