import numpy as np
from abc import abstractmethod
from Index import *
from TextRepresenter import *
from Parser import *
from Weighter import *
from Mesure import AveragePrecision

class IRModel:
    """
    Classe abstraite d'un modèle d'ordonnancement
    """
    def __init__(self,index):
        self.indexer = index
    @abstractmethod
    def getName(self):
        """
        Retourne le nom du modèle
        """
        pass
    @abstractmethod
    def getScores(self,query):
        """
        Retourne les scores des documents pour une requete
        Args:
            - query : requete considérée
        """
        pass

    def getRanking(self,query):
        """
        Retourne une liste de couples (document, score) ordonnée par score décroissant
        """
        return sorted(self.getScores(query).items(), key = lambda r : (r[1], r[0]), reverse=True)



class Vectoriel(IRModel):
    """
    Modèle Vectoriel
    Parameters:
        - weighter : Objet Weighter
        - normalized : booléen permettant de définir la fonction de score, si vrai score cosine sinon produit scalaire
    """
    def __init__(self,index,weighter,normalized=False):
        super().__init__(index)
        self.weighter = weighter
        self.normalized = normalized
    def getName(self):
        return "Vectoriel"
    def normalize(self, d):
        norm = float(np.sum(d.values()))
        return {key: value / norm for (key, value) in d.items()}

    def getScores(self,query):
        scores = {}
        weights = self.weighter.getWeightsForQuery(query)
        weights = {term:tf for (term,tf) in weights.items() if term in self.indexer.inv_index}
        if self.normalized :
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


class LanguageModel(IRModel):
    """
    Modèle de langue
    """
    def __init__(self,indexer,lamb= 0.8):
        super().__init__(indexer)
        self.lamb = lamb
        self.weighter = Weighter1(indexer)

        self.corpus_length = sum(len(d.get_text()) for d in self.indexer.docs.values())
    def getName(self):
        return "Jelinek"
    def getScores(self, query):
        weights = self.weighter.getWeightsForQuery(query)
        weights = {term:tf for (term,tf) in weights.items() if term in self.indexer.inv_index}
        index, index_inverse = self.indexer.get_index()
        scores = {}

        mc = {}
        for term in weights.keys():

            mc[term] = sum(self.indexer.index[d.id].get(term,0) for d in self.indexer.docs.values()) / self.corpus_length

        for term in weights.keys():
            for id in self.indexer.inv_index[term].keys():
                scores[id] = scores.get(id,1) * ((1 - self.lamb) * self.indexer.index_n[id][term] +self .lamb * mc[term])
        return scores

    def fit(self, possibilities, donnees, labels):
        l_max, s_max = 0, float("-inf")
        for l in possibilities:
            self.lamb = l
            s = 0
            for k in range(len(donnees)):
                predictionModele = self.getRanking(donnees[k])
                s += AveragePrecision(predictionModele, labels[k])
            if s > s_max:
                s_max = s
                l_max = l
        self.lamb = l_max


class OkapiBM25(IRModel):
    """
    Modèle OkapiBM25
    """
    def __init__(self, indexer,k1=1.2,b=.75):
        super().__init__(indexer)
        self.k1 = k1
        self.b = b
        self.weighter = Weighter1(indexer)
    def getName(self):
        return 'OkapiBM25'
    def getScores(self, query, pertinences=None):
        weights = self.weighter.getWeightsForQuery(query)
        weights = {term:tf for (term,tf) in weights.items() if term in self.indexer.inv_index}
        idf_word = {word: self.okapy_idf(word, pertinences) for word in weights.keys()}
        avgdl = sum(len(d.text) for d in self.indexer.docs.values()) / self.indexer.N
        scores = {}
        for term in weights.keys():
            for idoc in self.indexer.inv_index[term].keys():
                tf = self.indexer.index[idoc].get(term,0)
                score = idf_word[term]
                D = len(self.indexer.docs[idoc].text)
                score *= tf / (tf + self.k1 * (1 - self.b + self.b*(len(self.indexer.docs[idoc].text)/avgdl)))
                scores[idoc] = scores.get(idoc, 0) + score
        return scores

    def okapy_idf(self,query, pertinence=None):
        N = self.indexer.N
        if pertinence is None :
            return np.log(N/len(query))

        else:
            R, r = pertinence
            return np.log((r + 0.5) / R - r+ 0.5) * (N - len(query) - R + r + .5) / (len(query) - r + 0.5)

    def fit(self, possibilities, donnees, labels):
        # from sklearn.model_selection import cross_validate

        rangeK = possibilities[0]
        rangeB = possibilities[1]
        k1_max, b_max, s_max = 0, 0, float("-inf")
        for k1 in rangeK:
            self.k1 = k1
            for b in rangeB:
                self.b = b
                s = 0
                for k in range(len(donnees)):
                    predictionModele = self.getRanking(donnees[k])
                    s += AveragePrecision(predictionModele, labels[k])

                if s > s_max:
                    s_max = s
                    k1_max = k1
                    b_max = b

        self.k1, self.b = k1_max, b_max
