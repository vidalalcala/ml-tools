import numpy as np
import mltools.metrics
import sklearn.datasets
import scipy.stats
import matplotlib
matplotlib.use('Agg')


def test_roc_auc_score():
    labels_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc, auc_std = mltools.metrics.roc_auc_score(labels_true,
                                                 scores)
    np.testing.assert_almost_equal(auc, 0.75)
    np.testing.assert_almost_equal(auc_std, 0.35355339059327379)


def test_predict_margins():
    digits = sklearn.datasets.load_digits(2)
    model = mltools.metrics.XGBClassifierGTX(objective='binary:logistic')
    X = digits['data']
    y = digits['target']
    model = model.fit(X, y)
    margins_df = model.predict_margins(X)
    margins_sum = margins_df.sum()
    scores = scipy.stats.logistic.cdf(margins_sum)
    predict_proba = model.predict_proba(X)[:, 1]
    np.testing.assert_almost_equal(predict_proba, scores)
    model.plot_margins(X)
