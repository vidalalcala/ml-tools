import sklearn.metrics
import numpy as np
import rpy2.robjects.packages as packages
import rpy2.robjects.pandas2ri as pandas2ri


# R import and interfaces
p_roc = packages.importr('pROC')
pandas2ri.activate()


def roc_auc_score(y_true, y_score):
    """
    Calculates AUC with one standard deviation
    :param y_true: true response labels in {0,1}
    :param y_score: predictive score
    :return: auc: the AUC
             auc_std: AUC standard deviation
    """
    auc = sklearn.metrics.roc_auc_score(y_true, y_score)
    roc_object = p_roc.roc(y_true, y_score, algorithm=3)
    auc_var = p_roc.var(roc_object, method='delong')
    return auc, np.sqrt(auc_var[0])

