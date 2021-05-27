from sklearn.mixture import GaussianMixture
import numpy as np


def gaussian_mixture(train, test, components=2, max_iter=1000, scores_only=True):
    """
    Train Gaussian Mixture model.

    :param train:
    :param test:
    :param components:
    :param max_iter:
    :param scores_only:
    :return:
    """
    gm = GaussianMixture(n_components=components, max_iter=max_iter)
    model = gm.fit(train)

    y = np.reshape(np.array(test), (-1, 1))
    aic = model.aic(y)
    bic = model.bic(y)
    scores = model.predict_proba(test)
    means = model.means_
    covs = model.covariances_

    #print(scores)
    """std = np.sqrt(covs[0][0][0])
    exp_comp = np.exp(-0.5 * ((test[0][0] - means[0][0]) / std) ** 2)
    one = model.weights_[0] * exp_comp / (2. * np.sqrt(np.pi) * std)

    std = np.sqrt(covs[1][0][0])
    exp_comp = np.exp(-0.5 * ((test[0][0] - means[1][0]) / std) ** 2)
    two = model.weights_[1] * exp_comp / (2. * np.sqrt(np.pi) * std)

    print(one/(one+two), two/(one+two))"""

    if scores_only:
        return aic, bic, scores, means, covs
    return model, aic, bic, scores, means, covs




