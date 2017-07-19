import numpy as np
import mltools.metrics


def test_roc_auc_score():
    labels_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc, auc_std = mltools.metrics.roc_auc_score(labels_true,
                                                 scores)
    np.testing.assert_almost_equal(auc, 0.75)
    np.testing.assert_almost_equal(auc_std, 0.35355339059327379)
