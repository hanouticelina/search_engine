from abc import abstractmethod
import numpy as np
import sys
sys.path.append('./')
sys.path.append('..')
from Evaluation.Mesure import AveragePrecision
from Indexation.Parser import *
from Indexation.Index import *
from .Weighter import *


class IRModel:
    """
    Classe abstraite d'un modèle d'ordonnancement.
    ----------------------------------------------------
    """

    def __init__(self, index):
        self.indexer = index

    @abstractmethod
    def getName(self):
        """
        Retourne le nom du modèle.
        ----------------------------------------------------
        """
        pass

    @abstractmethod
    def getScores(self, query):
        """
        Retourne les scores des documents pour une requete.
        ----------------------------------------------------
        Args:
            - query : requete considérée.
        """
        pass

    def getRanking(self, query):
        """
        Retourne une liste de couples (document, score) ordonnée par score décroissant.
         ----------------------------------------------------
        """
        return sorted(self.getScores(query).items(), key=lambda r: (r[1], r[0]), reverse=True)









