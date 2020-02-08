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

    def __init__(self,id,text):
        self.id = id
        self.text = text

    def get_id(self):
        return self.id
    def get_text(self):
        return self.text

class Parser:
    
    def __init__(self):
        self.collection = dict()

    def get_collection(self):
        return self.collection

    def display(self): #TODO
        pass


    def buildDocCollectionSimple(self,path, balise=balise_T):
        out = []
        with open(path) as f:
            s = f.readline() #première ligne du fichier
            while s:
                if s.startswith(balise_I):

                    idoc = s.split()[1]
                    text = []
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
                if not(s.startswith(balise_I)):
                    s = f.readline()
        return out
