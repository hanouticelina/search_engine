from abc import abstractmethod
import numpy as np
from Evaluation.Mesure import AveragePrecision
from Indexation.Parser import *
from Indexation.Index import *
from .IRModel import *
from .Weighter import *


class OkapiBM25(IRModel):
    """
    Modèle OkapiBM25.
    ----------------------------------------------------
    """
    name = 'OkapiBM25'
    def __init__(self, indexer, weighter,k1=1.2, b=0.75):
        super().__init__(indexer)
        self.k1 = k1
        self.b = b
        self.weighter = weighter


    def getScores(self, query, pertinences=None):
        weights = self.weighter.getWeightsForQuery(query)
        weights = {term: tf for (term, tf) in weights.items() if term in self.indexer.inv_index}
        idf_word = {word: self.okapy_idf(word, pertinences) for word in weights.keys()}
        avgdl = sum(len(d.text) for d in self.indexer.docs.values()) / self.indexer.N
        scores = {}
        for term in weights.keys():
            for idoc in self.indexer.inv_index[term].keys():
                tf = self.indexer.index[idoc].get(term, 0)
                score = idf_word[term]
                D = len(self.indexer.docs[idoc].text)
                score *= tf / (tf + self.k1 * (1 - self.b + self.b * (D / avgdl)))
                scores[idoc] = scores.get(idoc, 0) + score
        return scores

    def okapy_idf(self, word, pertinence=None):
        N = self.indexer.N # nombre de documents dans la collection
        n = self.indexer.get_df(word) #nombre de documents contenant word
        if pertinence is None:
            return np.log(N - n + 0.5 / n + 0.5)
        # idf = self.indexer.idf(word)
        else:
            R, r = pertinence
            return np.log((r + 0.5) / (R - r + 0.5) * (N - n - R + r + .5) / n - r + 0.5)

        
    
    def train_test_tuning(self,possibilities, train_queries):
        rangeK, rangeB = possibilities[0],possibilities[1]
        k1_opt, b_opt, score_max = 0, 0, float("-inf")
        mesure = AveragePrecision()
        for k1 in rangeK:
            self.k1 = k1
            for b in rangeB:
                self.b = b
                score = 0
                for query in train_queries:
                    pred = np.array(self.getRanking(query.get_text()))[:,0]
                    relevants = query.get_relevants()
                    score += mesure.evalQuery(pred,relevants)
                score /= len(train_queries)
                if score > score_max:
                    score_max = score
                    k1_opt = k1
                    b_opt = b
        self.k1 = k1_opt
        self.b = b_opt
