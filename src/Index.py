# Almost done TODO: Index et index inversé normalisé + bonus

import utils.porter as pt
import numpy as np
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
        self.tf_idf = None
        self.indexer()
        self.compute_tf_idf()

    def get_index(self):
        return self.index

    def get_inv_index(self):
        return self.inv_index

    def get_df(self):
        return self.df

    def get_tf_idf(self):
        return self.tf_idf

    def tf(self,id, w):
        return self.index[id].get(w, 0)

    def idf(self,w):
        return np.log((1 + self.N) / (1 + self.df[w]))


    def tf_idf(self,id, w):
        return self.index[id][w] * self.idf(w)

    def compute_tf_idf(self):
        self.tf_idf = dict()
        for word in self.df.keys():
            for (i,d) in self.docs.items():
                self.tf_idf[word][i] = self.index[i][word] * self.idf(word)

    def getStrDoc(self,doc):
        return doc.get_text()

    def getTfsForDoc(self,doc):
        return self.index[doc.get_ind()]

    def getTfIDFsForDoc(self,doc):
        out = dict()

        for word in (self.index[doc.get_ind()]):
            out[word] = self.tf_idf(doc.get_ind(),w)
        return out

    def getTfsForStem(self,stem):
        return self.inv_index[stem]

    def getTfIDFsForStem(self,stem):
        return self.tf_idf[stem]

    def indexer(self):
        index_file = open("index.txt", "w")
        inv_index_file = open("index_inverse.txt", "w")
        ind = dict()
        inv = dict()
        df = dict()
        for (i,d) in self.docs.items():
            ind[i] = counter(d.get_text())
            index_file.write("{'"+str(i)+"': "+ str(ind[i])+"}\n")
            for word in ind[i]:
                if word not in inv.keys():
                    inv[word] = {}
                    inv[word][i] = ind[i][word]
                    df[word] = 1
                else:
                    inv[word][i] = ind[i][word]
                    df[word] += 1

        for word in df.keys():
            inv_index_file.write("{'"+word+"': "+ str(inv[word])+"}\n")
        self.index = ind
        self.inv_index = inv
        self.df = df
        index_file.close()
        inv_index_file.close()
