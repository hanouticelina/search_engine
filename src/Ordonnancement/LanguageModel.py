from abc import abstractmethod
import numpy as np

from Evaluation.Mesure import AveragePrecision
from Indexation.Parser import *
from Indexation.Index import *
from .Weighter import *
from .IRModel import *

class LanguageModel(IRModel):
    """
    Modèle de langue.
     ----------------------------------------------------
    """
    name = 'Jelinek'
    def __init__(self, indexer, lamb=0.8):
        super().__init__(indexer)
        self.lamb = lamb
        self.weighter = Weighter2(indexer)

        self.corpus_length = sum(len(d.get_text()) for d in self.indexer.docs.values())
        self.corpus_language = {}
        

    def getScores(self, query):
        weights = self.weighter.getWeightsForQuery(query)
        index, index_inverse = self.indexer.get_index()
        weights = {term: tf for (term, tf) in weights.items() if term in self.indexer.inv_index}
        
        scores = {}

        mc = {}
        for (term,t_weight) in weights.items():
            # p(t) sur tout le corpus, on compte le nombre d'apparition du terme dans le corpus et on divise sur le nombre total de mots dans le corpus
            mc[term] = sum(self.indexer.index[d.id].get(term, 0) for d in self.indexer.docs.values()) / self.corpus_length
        for (term,term_tf) in weights.items():
            # on parcourt chaque document contenant chaque terme de la requete
            for id in self.indexer.inv_index[term].keys():
                in_log = (self.lamb* self.indexer.index_n[id][term] + (1 - self.lamb) * mc[term])
                scores[id] = scores.get(id, 0) + term_tf * np.log(in_log)
        return scores

    def train_test_tuning(self, possibilities, train_queries):
        
        l_opt, score_max = 0.0, float("-inf")
        mesure = AveragePrecision()
        scores = np.zeros(len(possibilities))
        for i in range(len(possibilities)):
            self.lamb = possibilities[i]
            for query in train_queries:
                pred = np.array(self.getRanking(query.get_text()))[:,0]
                relevants = query.get_relevants()
                scores[i] += mesure.evalQuery(pred,relevants)
            scores[i] /= len(train_queries)

        self.lamb = possibilities[np.argmax(scores)]
