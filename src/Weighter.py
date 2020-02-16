import numpy as np
from abc import abstractmethod
from Index import *
from TextRepresenter import *
from Parser import *


class Weighter:

    def __init__(self,index):
        self.index = index


    @abstractmethod
    def getWeightsForDoc(idDoc):
        pass
    @abstractmethod
    def getWeightsForStem(stem):
        pass

    @abstractmethod
    def getWeightsForQuery(query):
        pass

class WeighterTF(Weighter):
    def __init__(self,index):
        super().__init__(self,index)

    def getWeightsForDoc(idDoc):
        return self.index.get_index()[0][idDoc]

    def getWeightsForStem(stem):
        return self.index.get_index()[1][stem]

    @abstractmethod
    def getWeightsForQuery(query):
        pass

class Weighter1(WeighterTF):

    def getWeightsForQuery(query):
        stemmer = PorterStemmer()
        return {w : 1 for w in stemmer.getTextRepresentation(query).keys()}

class Weighter2(WeighterTF):

    def getWeightsForQuery(query):
        stemmer = PorterStemmer()
        return stemmer.getTextRepresentation(query)


class Weighter3(WeighterTF):

    def getWeightsForQuery(query):
        stemmer = PorterStemmer()
        q = stemmer.getTextRepresentation(query)
        out = {}
        for w in q.keys():
            if w in self.index.get_index()[1].keys():
                out[w] = self.index.idf(w)
        return out

class Weighter4(Weighter):

    def getWeightsForDoc(idDoc):
        return {term: 1 + np.log(tf) for (term, tf) in self.index.get_index()[0][idDoc].items()}

    def getWeightsForStem(stem):
        return {doc: 1 + np.log(tf) for (doc, tf) in self.index.get_index()[1][stem].items()}

    def getWeightsForQuery(query):
        stemmer = PorterStemmer()
        q = stemmer.getTextRepresentation(query)
        for w in q.keys():
            if w in self.index.get_index[1].keys():
                out[w] = self.index.idf(w)
        return out

class Weighter5(Weighter):
    def __init__(self,index):
        super().__init__(self,index)

    def getWeightsForDoc(idDoc):
        return {term : 1+ np.log(tf) * self.index.idf[term] for (term, tf) in self.index.get_index[0][idDoc].items()}

    def getWeightsForStem(stem):
        if stem not in self.index.get_index[1].keys():
            return {}
        return {doc : 1 + np.log(tf) * self.index.idf(stem) for (doc,tf) in self.index.get_index()[1][stem].items()}

    def getWeightsForQuery(query):
        stemmer = PorterStemmer()
        q = stemmer.getTextRepresentation(query)
        idf_q = {w : self.index.idf(w) for w in q.keys()}
        return {term : 1+ np.log(tf) * idf_q[term] for (term,tf) in q.items()}
