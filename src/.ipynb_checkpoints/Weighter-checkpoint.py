"""Utiliser les méthodes définies dans Index et factoriser le code"""


import numpy as np
from abc import ABC, abstractmethod
from Index import *
from TextRepresenter import *
from Parser import *


class Weighter(ABC):
    """
    Classe abstraite de pondération
    """

    def __init__(self,index):
        self.index = index


    @abstractmethod
    def getWeightsForDoc(self,idDoc):
        """
        Retourne les poids des termes pour un document.
        Args:
            - idDoc : identifiant du document.
        """
        pass
    @abstractmethod
    def getWeightsForStem(self,stem):
        """
        Retourne les poids d'un terme pour tous les documents qui le contiennent.
        Args:
            - stem : terme considéré.
        """
        pass

    @abstractmethod
    def getWeightsForQuery(self,query):
        """
        Retourne les poids des termes d'une requete
        Args:
            - query : requete considéré
        """
        pass

class WeighterTF(Weighter):
    """
    Pondération TF
    """
    def getWeightsForDoc(self,idDoc):
        return self.index.index[idDoc]

    def getWeightsForStem(self,stem):
        return self.index.inv_index.get(stem,{})

    @abstractmethod
    def getWeightsForQuery(self,query):
        pass

class Weighter1(WeighterTF):
    """
    Binary Weighter 
    """
    def getWeightsForQuery(self,query):
        stemmer = PorterStemmer()
        return {w : 1 for w in stemmer.getTextRepresentation(query).keys()}

class Weighter2(WeighterTF):
    """
    TF Weighter
    """
    def getWeightsForQuery(self,query):
        stemmer = PorterStemmer()
        return stemmer.getTextRepresentation(query)


class Weighter3(WeighterTF):
    """
    TF-IDF Weighter
    """
    def getWeightsForQuery(self,query):
        stemmer = PorterStemmer()
        q = stemmer.getTextRepresentation(query)
        res = {}
        for w in q.keys():
            if w in self.index.get_index()[1].keys():
                res[w] = self.index.idf(w)
        return res

class Weighter4(Weighter):
    """
    Log Weighter
    """
    def getWeightsForDoc(self,idDoc):
        return {term: 1 + np.log(tf) for (term, tf) in self.index.get_index()[0][idDoc].items()}

    def getWeightsForStem(self,stem):
        return {doc: 1 + np.log(tf) for (doc, tf) in self.index.get_index()[1][stem].items()}

    def getWeightsForQuery(self,query):
        out = {}
        stemmer = PorterStemmer()
        q = stemmer.getTextRepresentation(query)
        for w in q.keys():
            if w in self.index.get_index()[1].keys():
                out[w] = self.index.idf(w)
        return out

class Weighter5(Weighter):
    """
    Log+ Weighter
    """
    def getWeightsForDoc(self,idDoc):
        return {term : (1 + np.log(tf)) * self.index.idf(term) for (term, tf) in self.index.get_index()[0][idDoc].items()}

    def getWeightsForStem(self,stem):
        if stem not in self.index.get_index()[1].keys():
            return {}
        return {doc : (1 + np.log(tf)) * self.index.idf(stem) for (doc,tf) in self.index.get_index()[1][stem].items()}

    def getWeightsForQuery(self,query):
        stemmer = PorterStemmer()
        q = stemmer.getTextRepresentation(query)
        idf_q = {w : self.index.idf(w) for w in q.keys()}
        return {term : (1 + np.log(tf)) * idf_q[term] for (term,tf) in q.items()}
