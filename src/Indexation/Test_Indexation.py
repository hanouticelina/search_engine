import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from Parser import *
from Index import *
from TextRepresenter import *
import argparse

def test_CACMShort():
    corpus_path = '../../cacmShort-good.txt'
    champs = None
    parser = None
    try:
        parser = Parser()
        parser.buildDocCollectionSimple(corpus_path)
    except FileNotFoundError:
        print("le fichier est introuvable")

    indexer = IndexerSimple(parser.collection)
    index, inv_index = indexer.get_index()
    assert 'algebra' in index['1']
    assert len(index['2']) == 6
    assert sum(index['11'].values()) == 8

    assert 'algebra' in indexer.index_n['1']
    assert abs(sum(indexer.index_n['2'].values()) - 1) < 1e-4

    assert inv_index['matrix'] == {'3': 1}
    assert len(inv_index['comput']) == 5

    assert indexer.inv_index_n['matrix'] == {'3': .2}
    assert len(indexer.inv_index_n['comput']) == 5

    tf_idf = indexer.compute_tf_idf()

    # tfidf à la même structure que ind
    assert tf_idf.keys() == index.keys()
    for i_doc in tf_idf.keys():
        assert tf_idf[i_doc].keys() == index[i_doc].keys()
    #print(indexer.getStrDoc('6'))

def parsing(corpus_path):
    champs = None
    parser = None
    try:
        parser = Parser()
        parser.buildDocCollectionSimple(corpus_path)
    except FileNotFoundError:
        print("le fichier est introuvable")

    indexer = IndexerSimple(parser.collection)
    return indexer

def main():
    """ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str, help = 'chemin vers le fichier à parser')
    args = ap.parse_args()
    path = args.path
    indexer = parsing(path)"""
    test_CACMShort()
    print("Le fichier a été parsé et l'indexation a été effectuée correctement")

    """"ap.add_argument('-ind', '--index', help="option pour afficher l'index",choices=['original','normalized', 'no'], default='no')
    ap.add_argument('-indinv', '--indexinverse', help="option pour afficher l'index inversé",choices=['original','normalized', 'no'], default='no')
    ap.add_argument('-tfidf', '--tfidf', help="option pour afficher le dictionnaire tfidf", choices=['yes', 'no'],default='no')
    
    indexer = test_bigFile(path)
    if args.index == 'original':
        print(indexer.get_index()[0])
    elif args.index == 'normalized':
        print(indexer.get_index(normalized=True)[0])
    else:
        
    print('------------------------------------------------------------------------')
    if args.indexinverse == 'original':
        print(indexer.get_index()[1])
    elif args.indexinverse == 'normalized':
        print(indexer.get_index(normalized=True)[1])
    else:
        print("\n")
    print('-----------------------------------------------------------------------')
    if args.indexinverse == 'yes':
        tf_idf = indexer.compute_tf_idf()
        print(tf_idf)
    else:
        print("\n")"""

if __name__ == "__main__":
    main()