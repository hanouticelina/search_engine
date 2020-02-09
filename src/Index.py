# Almost done TODO: bonus

import utils.porter as pt
import numpy as np
import copy
import TextRepresenter as tr
import collections

def counter(str):
    l = [pt.stem(w.lower()) for w in str.split(" ")]
    stemmer = tr.PorterStemmer()
    l = [word for word in l if word not in stemmer.stopWords]
    return dict(collections.Counter(l).items())

class IndexerSimple:

    def __init__(self,docs):
        self.docs = docs
        self.N = len(docs)
        self.index = None
        self.inv_index = None
        self.index_n = None
        self.inv_index_n = None
        self.df = None
        self.indexer()
        self.tf_idf = self.compute_tf_idf()


    def get_index(self,normalized=False):
        if normalized:
            return self.index_n, self.inv_index_n
        else:
            return self.index, self.inv_index


    def get_df(self):
        return self.df

    def get_tf_idf(self):
        return self.tf_idf

    def idf(self,w):
        return np.log((1 + self.N) / (1 + self.df[w]))

    def compute_tf_idf(self):
        return {d.id : {word: self.index[d.id][word] * self.idf(word) for word in self.index[d.id].keys()} for d in self.docs.values()}


    def indexer(self):
        index_file = open("index.txt", "w")
        inv_index_file = open("index_inverse.txt", "w")
        index = dict()
        inv_index = dict()
        df = dict()
        norm_index=dict()
        words = set() #vocabulary
        for (i,d) in self.docs.items():
            index[i] = counter(d.get_text())
            norm_index[i] = {key: value/sum(index[i].values()) for (key, value) in index[i].items()}
            words = words.union(set(index[i].keys()))
            index_file.write("{'"+str(i)+"': "+ str(index[i])+"}\n")
            for word in index[i]:
                if word not in inv_index.keys():
                    inv_index[word] = {}
                    inv_index[word][i] = index[i][word]
                    df[word] = 1
                else:
                    inv_index[word][i] = index[i][word]
                    df[word] += 1

        for word in df.keys():
            inv_index_file.write("{'"+word+"': "+ str(inv_index[word])+"}\n")
        inv_index_n = {w: {d.get_id(): norm_index[d.get_id()][w] for d in self.docs.values() if w in norm_index[d.get_id()]} for w in words}
        self.index = index
        self.inv_index = inv_index
        self.index_n = norm_index
        self.inv_index_n = inv_index_n
        self.df = df
        index_file.close()
        inv_index_file.close()

        def getStrDoc(self,doc):
            return doc.get_text()

        def getTfsForDoc(self,doc):
            try:
                return self.index[doc.id]
            except KeyError:
                print("le document n'existe pas dans l'index")
        def getTfIDFsForDoc(self,doc):
            try:
                return self.tf_idf[doc.id]
            except KeyError:
                print("le document n'existe pas dans l'index")

        def getTfsForStem(self,stem):
            try:
                return self.inv_index[stem]
            except KeyError:
                print()
        def getTfIDFsForStem(self,stem):
            return {d: self.tf_idf[d][stem] for d in self.inv_index[stem].keys()}
