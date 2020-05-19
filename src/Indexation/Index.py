import sys
sys.path.append('./')
sys.path.append('../')
from utils import porter as pt
import numpy as np
import copy
import Indexation.TextRepresenter as tr
import collections


def counter(doc):
    """
    Processing d'un document.
    ------------------------------------------------------
    Args:
        - doc : document considéré.
    Return:
        - (mot,occurrence) pour chaque mot présent dans le document.
    """
    l = [pt.stem(w.lower()) for w in doc.split(" ")]
    stemmer = tr.PorterStemmer()
    l = [word for word in l if word not in stemmer.stopWords]
    return dict(collections.Counter(l).items())


class IndexerSimple:
    """
    Classe permettant l'indexation d'une collection de documents.
    ------------------------------------------------------
    Parameters:
        - docs : Documents de la collection.
        - N : Nombre de documents dans la collection.
        - index : index (identifiant_document : (token : nb_occurrences)).
        - inv_index : index inverse.
        - index_n : index normalisé.
        - inv_index_n : index inverse normalisé.
        - df : document frequency (token : nombre de documents dans lequel il apparait).
        - tf_idf : TF-IDF (token : (identifiant_document : tf-idf)).
    """

    def __init__(self, docs):
        self.docs = docs
        self.N = len(docs)
        self.index = None
        self.inv_index = None
        self.index_n = None
        self.inv_index_n = None
        self.df = None
        self.indexer()
        self.tf_idf = self.compute_tf_idf()

    def get_index(self, normalized=False):
        """
        Renvoie l'index et l'index inverse d'une collection.
        ------------------------------------------------------
        Args:
            - normalized : booléen, si vrai, la méthode renvoie l'index et l'index inverse normalisés.
        """
        if normalized:
            return self.index_n, self.inv_index_n
        else:
            return self.index, self.inv_index

    def get_df(self, w):
        """
        Renvoie le document frequency du token w.
        ------------------------------------------------------
        Args:
            - w : token considéré.
        """
        return len(self.inv_index[w]) if w in self.inv_index else 0

    def get_tf_idf(self):
        """
        Renvoie le dictionnaire tf-idf.
        """
        return self.tf_idf

    def idf(self, w):
        """
        Renvoie l'idf du token w.
        ------------------------------------------------------
        Args:
            - w : token considéré.
        """
        return np.log((1 + self.N) / (1 + self.get_df(w)))

    def compute_tf_idf(self):
        """
        Construit le dictionnaire tf-idf, qui associe à chaque document un 
        dictionnaire qui lui-même associe, pour chaque tokenqui apparait dans le document, son tf-idf.
        -----------------------------------------------------
        
        """
        return {d.id: {word: self.index[d.id][word] * self.idf(word) for word in self.index[d.id].keys()} for d in
                self.docs.values()}
    
    def indexer(self):
        """
        Construit l'index et l'index inverse et les ecrit dans des fichiers.
        -----------------------------------------------------
        """
        index_file = open("../index_files/index.txt", "w+")
        inv_index_file = open("../index_files/index_inverse.txt", "w+")
        index = dict()
        inv_index = dict()
        df = dict()
        norm_index = dict()
        words = set()  # vocabulary
        stemmer = tr.PorterStemmer()
        for (i, d) in self.docs.items():
            index[i] =  stemmer.getTextRepresentation(d.get_text()) #counter(d.get_text())
            norm_index[i] = {key: value / sum(index[i].values()) for (key, value) in index[i].items()}
            words = words.union(set(index[i].keys()))
            index_file.write("{'" + str(i) + "': " + str(index[i]) + "}\n")
            for word in index[i]:
                if word not in inv_index.keys():
                    inv_index[word] = {}
                    inv_index[word][i] = index[i][word]
                    df[word] = 1
                else:
                    inv_index[word][i] = index[i][word]
                    df[word] += 1

        for word in df.keys():
            inv_index_file.write("{'" + word + "': " + str(inv_index[word]) + "}\n")
        inv_index_n = {w: {d.get_id(): norm_index[d.get_id()][w] for d in self.docs.values() if w in norm_index[d.get_id()]} for w in words}
        self.index = index
        self.inv_index = inv_index
        self.index_n = norm_index
        self.inv_index_n = inv_index_n
        self.df = df
        index_file.close()
        inv_index_file.close()
        print("L'indexation a été effectuée avec succès ! taille du corpus : {}".format(len(self.index)))

    def getStrDoc(self, idoc):
        """
        Retourne le texte d'un document.
        ------------------------------------------------------
        Args:
            idoc : Identifiant du document considéré.
        """
        return self.docs[idoc].get_text()

    def getTfsForDoc(self, idoc):
        """
        Retourne la repr´esentation (stem-tf) d’un document partir de l’index.
        ------------------------------------------------------
        Args:
            idoc : Identifiant du document considéré.
        """
        try:
            return self.index[idoc]
        except KeyError:
            print("le document n'existe pas dans l'index")

    def getTfIDFsForDoc(self, idoc):
        """
         Retourne la représentation (stem-TFIDF) d’un document a partir de l’index.
         ------------------------------------------------------
         Args:
            doc : Document considéré.
        """
        try:
            return self.tf_idf[idoc]
        except KeyError:
            print("le document n'existe pas dans l'index")

    def getTfsForStem(self, stem):
        """
        Retourne la repr´esentation (doc-tf) d’un stem a partir de l’index inverse.
        ------------------------------------------------------
        Args:
            stem : terme considéré.
        """
        return self.inv_index.get(stem, {})


    def getTfIDFsForStem(self, stem):
        """
        Retourne la repr´esentation (doc-TFIDF) d’un stem a partir de l’index inverse.
        ------------------------------------------------------
        Args:
            stem : terme considéré.
        """
        return {d: self.tf_idf[d][stem] for d in self.inv_index[stem].keys()}

    def get_docs(self):
        """
        Retourne la collection de documents.
        -----------------------------------------------------
        """
        return self.docs
