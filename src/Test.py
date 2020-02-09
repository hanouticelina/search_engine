import utils.porter as pt
import numpy as np
import TextRepresenter as tr
import collections
import Index
import Parser as ps



def short_test_cacm():
    file = "./cacmShort-good.txt"
    try:
        parser = ps.Parser()
        _ = parser.buildDocCollectionSimple(file)
    except FileNotFoundError:
        pass

    indexer = Index.IndexerSimple(parser.collection)
    assert indexer.tf_idf.keys() == indexer.index.keys()
    ind, inv_ind = indexer.get_index()
    print("index : ",ind)
    print("\n\n")
    print("index inversé: ",inv_ind)
    print("\n\n\n")
    ind_n, inv_ind_n = indexer.get_index(normalized=True)
    print("index normalisé : ",ind_n)
    print("\n\n")
    print("index inversé normalisé: ",inv_ind_n)
    print("\n\n\n")

def long_test_cacm():
    file = "./data/cacm/cacm.txt"
    try:
        parser = ps.Parser()
        _ = parser.buildDocCollectionSimple(file)
    except FileNotFoundError:
        pass

    indexer = Index.IndexerSimple(parser.collection)
    assert indexer.tf_idf.keys() == indexer.index.keys()
    ind, inv_ind = indexer.get_index()
    print("index : ",ind)
    print("\n\n")
    print("index inversé: ",inv_ind)
    print("\n\n\n")
    ind_n, inv_ind_n = indexer.get_index(normalized=True)
    print("index normalisé : ",ind_n)
    print("\n\n")
    print("index inversé normalisé: ",inv_ind_n)

short_test_cacm()
long_test_cacm()
