import sklearn.metrics
import numpy as np
import xgboost
import pandas as pd
import rpy2.robjects.packages as packages
import rpy2.robjects.pandas2ri as pandas2ri
import matplotlib
matplotlib.use('Agg')
import matplotlib.backends.backend_pdf as backend_pdf
import seaborn

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


class XGBClassifierGTX(xgboost.XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
        super(xgboost.XGBClassifier, self).__init__(max_depth,
                                                    learning_rate,
                                                    n_estimators,
                                                    silent,
                                                    objective,
                                                    booster,
                                                    n_jobs,
                                                    nthread,
                                                    gamma,
                                                    min_child_weight,
                                                    max_delta_step,
                                                    subsample,
                                                    colsample_bytree,
                                                    colsample_bylevel,
                                                    reg_alpha,
                                                    reg_lambda,
                                                    scale_pos_weight,
                                                    base_score,
                                                    random_state,
                                                    seed,
                                                    missing,
                                                    **kwargs)

    def predict_margins(self, X):
        """
        :param X: feature dataset
        :return:
        """
        booster = self.get_booster()
        booster_dump = booster.get_dump(dump_format='json')
        nb_trees = len(booster_dump)

        margins_dict = {}
        output_margin = True
        margins_dict[1] = self.predict_proba(X,
                                             output_margin=output_margin,
                                             ntree_limit=1)[:, 1]
        for i in range(1, nb_trees):
            predict_margins = self.predict_proba(X,
                                                 output_margin=output_margin,
                                                 ntree_limit=i)[:, 1]
            next_predict_margins = self.predict_proba(X,
                                                      output_margin=output_margin,
                                                      ntree_limit=(i + 1))[:, 1]
            margins_dict[i + 1] = next_predict_margins - predict_margins
        margins_df = pd.DataFrame(margins_dict)
        return margins_df.transpose()

    def plot_margins(self, X, file_name='margins.pdf'):
        margins = self.predict_margins(X)
        pp = backend_pdf.PdfPages(file_name)
        ax = seaborn.tsplot([margins[column] for column in margins])
        ax.set(xlabel='boost_step', ylabel='margin')
        pp.savefig()
        pp.close()
