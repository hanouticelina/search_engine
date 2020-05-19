# TODO: modifier la méthode build afin de pouvoir extraire d'autres balises

import utils.porter as pt
import numpy as np
import TextRepresenter as tr
import Index as ind
import collections
import re

other_balises = {'.I', '.T', '.B', '.A', '.K', '.W', '.X'}
balise_I = '.I'
balise_T = '.T'



class Document:
    """
    Classe permettant la représentation d'un document.
    Parameters:
        - id : identifiant du document
        - text : texte du document
    """
    def __init__(self,id,text):
        self.id = id
        self.text = text

    def get_id(self):
        return self.id
    def get_text(self):
        return self.text

class Parser:
    """
    Classe permettant de parser une collection et de la stocker dans un dictionnaire d'objets de type Document.
    """
    def __init__(self):
        self.collection = dict()

    def get_collection(self):
        return self.collection

    def display(self): #TODO
        pass
    def set_collection(self, docs):
        """
        Méthode permettant de créer la collection à partir d'un ensemble de textes
        Args:
            - docs : ensemble de textes
        """
        for i, doc in enumerate(docs):
            self.collection[i] = Document(i, doc)

    def buildDocCollectionSimple(self,path, balise=balise_T):
        """
        Parse un fichier et stocke la collection sous forme d'un dictionnaire de Document
        Args :
            - path : chemin vers le fichier
            - balise : balise contenant le texte qu'on souhaite recupérer
        """
        out = {}
        with open(path) as f:
            s = f.readline() #première ligne du fichier
            while s:
                if s.startswith(balise_I):

                    idoc = s.split()[1]
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
                            out[idoc] = doc
                if not(s.startswith(balise_I)):
                    s = f.readline()
        return out
