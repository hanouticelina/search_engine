import numpy as np
from abc import abstractmethod
import sys
sys.path.append("..")
from Indexation.Index import *
from Indexation.TextRepresenter import *
from Indexation.Parser import *
from Ordonnancement.Weighter import *
import string

class Query:
    """
    Classe permettant de représenter une requete.
    -----------------------------------------------------
    Parameters:
        - id : identifiant de la requete
        - text : texte de la requete
        -list_relevants : liste des documents pertinents pour la requete
    """
    def __init__(self,id,text,relevants):
        self.id = id
        self.text = text
        self.list_relevants = relevants

    def get_id(self):
        return self.id

    def get_text(self):
        return self.text

    def get_relevants(self):
        return self.list_relevants

class QueryParser:
    """
    Classe permettant de lire les fichiers associés aux requetes et de jugements de pertinence et qui stocke cela
    dans une collection de Query.
    -----------------------------------------------------
    Parameters :
        - path : chemin vers les fichiers
        - parser : objet Parser permettant de parser le fichier
        - queries : dictionnaire associé à une requete (identifiant de la requete : Objet Query)
    """
    def __init__(self,path):
        self.path = path
        self.parser = Parser()
        self.queries = {}
        self.parser.buildDocCollectionSimple(path +".qry", balise=".W")
        list_relevants = self.loadRelevance(path+".rel")

        for (idoc, doc) in self.parser.collection.items():
            text = doc.get_text()
            self.queries[idoc] = Query(idoc,text,list_relevants.get(idoc,[]))
        print("Parsing des requetes effectué avec succès ! Nombre de requetes : {}".format(len(self.queries)))

    def loadRelevance(self,path):
        """
        Permet de lire et de parser le fichier associé aux jugements de pertinence.
        -----------------------------------------------------
        """
        relevants = {}
        with open(path) as file:
            for line in file:
                iq, idoc = line.split()[:2]
                iq = int(iq)
                idoc = int(idoc)
                if iq not in relevants:
                    relevants[iq] = []
                relevants[iq].append(idoc)
        return relevants

    def get_queries(self):
        return self.queries
