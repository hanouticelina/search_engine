import sys
sys.path.append("..")
from .Mesure import *
from Ordonnancement import *
import numpy as np
from scipy import stats



class EvalIRModel:
    """Evaluation d'un modèle d'appariement avec une mesure d'evaluation.
    -----------------------------------------------------
     Parameters:
        - model : modèle d'appariement.
        - mesure : mesure d'evaluation.
    """
    def __init__(self, model, mesure):
        self.model = model
        self.mesure = mesure

    def eval(self, queryParser, verbose=False):
        """
        Methode pour l'evaluation du modèle.
        -----------------------------------------------------
        Args:
            - queryParser : Objet permettant de stocker les requetes.
            - Verbose : variable permettant de controller l'affichage.
        """
       
        queries = queryParser.get_queries()
        evals = []
        nb_queries = len(queries.items())
        for (idQ, query) in queries.items():
            ranking = np.array(self.model.getRanking(query.get_text()))[:,0]
            relevants = query.get_relevants()
            score = self.mesure.evalQuery(ranking,relevants)
            
            if score is not None: # On ne prend pas en compte les requetes n'ayant pas de documents pertinents
                evals.append(score)
            else:
                nb_queries -= 1
            if verbose : 
                
                print(" Query {} : {}".format(idQ, query.get_text()))
                print("\t 10 first documents returned by our model : {}".format(ranking[:10]))
                print("\t Relevant documents for this query : {}".format(relevants))
                if score is None:
                    print("\t Cette requete ne contient pas de documents pertinents !")
                else:
                    print("\t Score : {}".format(score))
                print("************************************************")
        evals = np.array(evals)
        mean_ =  np.sum(evals) / nb_queries*1.0
        std_ = np.sqrt(np.sum(np.square(evals-mean_)) / nb_queries)
        return evals, mean_, std_


# Bonus TME3 : paired t-test
def paired_ttest(scores1, scores2, threshold=0.05):
    """
    Methode permettant d'effectuer un paired t-test.
    -----------------------------------------------------
    Args:
        - score1 : scores obtenus à l'aide du premier modèle.
        - score2 : scores obtenus à l'aide du second modèle.
    """
    tstat, pvalue = stats.ttest_ind(scores1,scores2)
    if pvalue < threshold: 
        print("les performances des deux modèles sont significativement différentes à 95% de confiance : t = {}".format(tstat))
    else:
        print("les performances des deux modèles ne sont pas significativement différentes à 95% de confiance : t = {}".format(tstat))
            
    return pvalue, tstat


