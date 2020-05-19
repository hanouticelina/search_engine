# TODO: modifier la méthode build afin de pouvoir extraire d'autres balises
import sys
sys.path.append('./')
sys.path.append('../')
from .utils import porter as pt
import numpy as np
import Indexation.TextRepresenter as tr
import collections
import re

other_balises = {'.I', '.T', '.B', '.A', '.K', '.W', '.X'}
balise_I = '.I'
balise_T = '.T'



class Document:
    """
    Classe permettant la représentation d'un document.
    ----------------------------------------------------
    Parameters:
        - id : identifiant du document
        - text : texte du document
    """
    def __init__(self,id,text):
        self.id = id
        self.text = text
        self.hyperliens = None
        self.other = None

    def get_id(self):
        return self.id

    def get_text(self):
        return self.text

    def get_hyperliens(self):
        return self.hyperliens

    def get_other(self):
        return self.other

    def set_hyperliens(self, hyperliens):
        self.hyperliens = hyperliens
    def set_other(self, champ):
        self.other = champ

class Parser:
    """
    Classe permettant de parser une collection et de la stocker dans un dictionnaire d'objets de type Document.
    """
    def __init__(self):
        self.collection = dict()

    def get_collection(self):
        return self.collection

    def display(self):
        for (idoc, doc) in self.collection.items():
            print("Document : {}\n".format(idoc))
            print("Text : {}".format(doc.get_text()))
            print("Hyperliens : {}".format(doc.get_hyperliens()))
            print("Other : {}".format(doc.get_other()))
            print("*************************************************")

    def set_collection(self, docs):
        """
        Méthode permettant de créer la collection à partir d'un ensemble de textes.
        -----------------------------------------------------
        Args:
            - docs : ensemble de textes.
        """
        for i, doc in enumerate(docs):
            self.collection[i] = Document(i, doc)

    def buildDocCollectionSimple(self,path, balise=balise_T,balise_fac=None):
        """
        Parse un fichier et stocke la collection sous forme d'un dictionnaire de Document.
        ------------------------------------------------------
        Args :
            - path : chemin vers le fichier
            - balise : balise contenant le texte qu'on souhaite recupérer
        """
        with open(path) as f:
            s = f.readline() #première ligne du fichier
            while s:
                if s.startswith(balise_I):
                    hyperliens = []
                    idoc = int(s.split()[1])
                    text = []
                    while(not(s.startswith(balise))):
                        s = f.readline()

                    if (s.startswith(balise)):
                        s = f.readline()
                        while not(s.startswith(".")) and s:
                            text.append(s[:-1]+" ")
                            s = f.readline()
                        text = ''.join(text)
                        text = text[:-1]
                        if len(text) > 0:
                            doc = Document(idoc,text)
                            self.collection[idoc] = doc
                    if balise_fac is not None:
                        while(not(s.startswith(balise_fac))):
                            s = f.readline()
                        champs = []
                        if (s.startswith(balise_fac)):
                            s = f.readline()
                            while not (s.startswith(".")) and s:
                                champs.append(s[:-1] + " ")
                                s = f.readline()
                            champs = ''.join(champs)
                            champs = champs[:-1]
                            if len(champs) > 0:
                                if idoc in self.collection.keys():
                                    self.collection[idoc].set_other(champs)
                    """while(not(s.startswith('.X'))):
                        s = f.readline()
                    if (s.startswith('.X')):
                        s = f.readline()
                        while not(s.startswith(".")) and s:
                            hyperlien = ''.join(s.split()[:1])
                            if len(hyperlien) > 0:
                                hyperliens.append(hyperlien)
                            s = f.readline()
                    if idoc in self.collection.keys():
                        self.collection[idoc].set_hyperliens(hyperliens)"""

                if not(s.startswith(balise_I)):
                    s = f.readline()
