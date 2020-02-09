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

short_test_cacm()
long_test_cacm()
