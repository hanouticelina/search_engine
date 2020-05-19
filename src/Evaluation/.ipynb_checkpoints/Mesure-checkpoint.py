import numpy as np
from abc import ABC, abstractmethod
import sys
sys.path.append("..")
from Indexation.Index import *
from Indexation.TextRepresenter import *
from Indexation.Parser import *
from Ordonnancement.Weighter import *
from .Query import *

class EvalMesure(ABC):
    """
    Classe abstraite pour les mesures d'evaluation.
    -----------------------------------------------------
    """
    @abstractmethod
    def evalQuery(self,liste,relevants):
        """
        Calcule la mesure pour la liste de documents retournés par un modèle et la liste de documents pertinents.
        -----------------------------------------------------
        Args :
            - liste : liste de documents retournés par un modèle
            - relevants : liste de documents pertinents pour une requete
        """
        pass

    def eval_list_query(self, list_pred, list_relevants):
        """
        Retourne la moyenne et l'ecart type pour un modèle.
        -----------------------------------------------------
        Args :
            - list_pred : liste de documents retournés par un modèle.
            - list_relevants : liste de documents pertinents pour une requete.
        """
        
        res = [self.evalQuery(pred, labels) for pred, relevants in zip(list_pred, list_relevants)]

        return np.mean(res), np.std(res)


class PrecisionK(EvalMesure):
    """
    Classe associée à la precision au range k.
    -----------------------------------------------------
    """
    def __init__(self,k):
        self.k = k
    def evalQuery(self, liste,relevants):
        n = min(self.k, len(relevants))
        if len(relevants) == 0:
            # This query has no relevant documents, we return either None or 1 (?)
            return 1
        rel_th = relevants[:n]
        pred = liste[:n]
        nb_correct = 0
        for doc in pred:
            if doc in rel_th:
                nb_correct += 1
        return nb_correct / n


class RecallK(EvalMesure):
    """
    Classe associée au rappel au rang k.
    ------------------------------------------------------
    """
    def __init__(self,k):
        self.k = k

    def evalQuery(self, liste, relevants):
        n = min(self.k, len(relevants))
        if len(relevants) == 0:
            # This query has no relevant documents
            return 1
        rel_th = relevants[:n]
        pred = liste[:n]
        nb_correct = 0
        for doc in pred:
            if doc in rel_th:
                nb_correct += 1
        return nb_correct / len(relevants)

class FMesureK(EvalMesure):
    """
    Classe associée à la mesure-F au rang k.
    ------------------------------------------------------
    """
    def __init__(self, beta, k):
        self.beta = beta
        self.k = k

    def evalQuery(self, liste, relevants):
        if len(relevants)==0:
            # This query has no relevant documents
            return 1

        relevants = set(relevants)
        n = min(self.k, len(relevants))
        cpt = 0
        for p in liste[:n]:
            if p in relevants:
                cpt+=1

        if cpt==0:
            return 0

        precision = cpt/n
        rappel = cpt/len(relevants)

        return (1 + self.beta**2) * (precision * rappel)/\
               (self.beta**2 * precision + rappel)
class AveragePrecision(EvalMesure):
    """
    Classe associée à la precision moyenne.
    ------------------------------------------------------
    """
    def evalQuery(self, liste, relevants):
        n = len(relevants)
        if (len(relevants) == 0):
            # This query has no relevant documents
            return 1
        nb_correct = 0
        total = 0
        for k in range(len(liste)):
            if liste[k] in relevants:
                for i in range(k):
                    if liste[i] in relevants:
                        nb_correct +=1
                nb_correct /= (k+1)
                total += nb_correct
        return total * 1.0 / n

    def evalQueryApprox(self, liste, query):
        pass


class ReciprocalRank(EvalMesure):
    """
    Classe associée à la moyenne des rangs inverses.
    ------------------------------------------------------
    """
    def evalQuery(self, liste, relevants):
        if (len(relevants) == 0):
            # This query has no relevant documents
            return 1

        best = liste[0]
        if best not in relevants:
            return 0

        return 1 / (relevants.index(best) + 1)



class NDCG(EvalMesure):
    """
    Classe associée à la mesure NDCG.
    ------------------------------------------------------
    """
    def __init__(self, rg):
        self.rg = rg

    def iDCG(self):
        """
        Calcule l'iDCG (idéal DCG)
        """
        return 1 + np.sum([1 / np.log2(k + 1) for k in range(1,self.rg)])



    def DCG(self, liste, relevants):
        """
        Calcule le DCG.
        ------------------------------------------------------
        """
        rg = min(self.rg, len(relevants))
        relevants = set(relevants)
        sum_ = 1 if liste[0] in relevants else 0
        for k in range(1, rg):
            if liste[k] in relevants:
                sum_ += 1 / np.log2(k + 1)

        return sum_

    def evalQuery(self, liste, relevants):
        
        if len(relevants) == 0:
            # This query has no relevant documents
            return 1
        ndcg = self.DCG(liste, relevants) / self.iDCG()
        return ndcg
