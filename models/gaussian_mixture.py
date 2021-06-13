from sklearn.mixture import GaussianMixture as gm
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
    model = gm(n_components=components, max_iter=max_iter)
    model = model.fit(train)

    y = np.reshape(np.array(test), (-1, 1))
    aic = model.aic(y)
    bic = model.bic(y)
    scores = model.predict_proba(test)
    means = model.means_
    covs = model.covariances_

    if means[0][0] == means.max():
        high = 0
        low = 1
    else:
        high = 1
        low = 0

    results = [scores[0][low], scores[0][high], means[low][0], means[high][0], covs[low][0][0], covs[high][0][0], aic, bic]

    """std = np.sqrt(covs[0][0][0])
    exp_comp = np.exp(-0.5 * ((test[0][0] - means[0][0]) / std) ** 2)
    one = model.weights_[0] * exp_comp / (2. * np.sqrt(np.pi) * std)

    std = np.sqrt(covs[1][0][0])
    exp_comp = np.exp(-0.5 * ((test[0][0] - means[1][0]) / std) ** 2)
    two = model.weights_[1] * exp_comp / (2. * np.sqrt(np.pi) * std)

    print(one/(one+two), two/(one+two))"""

    if scores_only:
        return results
    return model, results


def remove_outliers(data, sigma=2, return_outliers=False):
    """
    Remove density outliers to avoid spurious fitting results.

    :param data: List of phase space densities.
    :param return_outliers: Bool to return outliers.

    :return: Depending on return_outliers parameter return only cleaned data or cleaned data and outliers as a
    separate value.
    """

    x = np.expand_dims(np.log10(data), axis=0).T

    m = np.mean(x)
    std = np.std(x)

    outliers = x[np.where((x < m - (2 * std)) | (x > m + (2 * std)))]
    x = x[np.where((x > m - (sigma * std)) & (x < m + (sigma * std)))]

    outliers = np.reshape(outliers, (outliers.shape[0], 1))
    x = np.reshape(x, (x.shape[0], 1))

    if return_outliers:
        return x, outliers
    return x
