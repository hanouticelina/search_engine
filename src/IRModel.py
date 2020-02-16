import numpy as np
from abc import abstractmethod
from Index import *
from TextRepresenter import *
from Parser import *
from Weighter import *


class IRModel:

    def __init__(self,index):
        self.index = index
    @abstractmethod
    def getScores(query):
        pass

    def getRanking(query):
        return sorted(self.getScores(query).items(), key = lambda r : (r[1], r[0]), reverse=True)

class Vectoriel(IRModel):
    def __init__(self,index,weighter,normalized=False):
        super().__init__(index)
        self.weighter = weighter
        self.normalized = normalized

    def normalize(self, d):
        norm = float(np.sum(d.values()))
        return {key = value / norm for (key, value) in d.items()}

    def getScores(query):
        scores = {}
        weights = self.weighter.getWeightsForQuery(query)

        if normalized :
            normQ = np.sqrt(np.sum(np.power(list(weights.values()),2)))
            for term in weights:
                t_weights = self.weighter.getWeightsForStem(term)
                for (doc,doc_w) in t_weights.items():
                    d_weights = self.weighter.getWeightsForDoc(doc)
                    normD = np.sqrt(np.sum(np.power(list(d_weights.values()),2)))
                    scores[doc] = scores.get(doc,0)+(doc_w * weights[term]) / (normD + normQ)
        else:
            for term in weights:
                t_weights = self.weighter.getWeightsForStem(term)
                for (doc, doc_w) in t_weights.items():
                    scores[doc] = scores.get(doc,0) + (doc_w * weights[term])
        return scores
