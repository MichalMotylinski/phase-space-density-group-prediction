import numpy as np
from sklearn.mixture import GaussianMixture as gm


def gaussian_mixture(x, y, components=2, max_iter=1000, scores_only=True):
    """
    Train Gaussian Mixture model using phase space density data.

    :param x: Phase space densities array of target star neighbours
    :param y: Phase space density of target star
    :param components: Number of components to apply to Gaussian Mixture model
    :param max_iter: Number of step used by the best fit of expectation-maximization algorithm to reach the convergence.
    :param scores_only: Return model and scores or scores only

    :return: scores: List of scores as well as Gaussian Mixture attributes,
     model: Gaussian Mixture model object
    """

    model = gm(n_components=components, max_iter=max_iter)
    model = model.fit(x)

    # Return model attributes
    y = np.reshape(np.array(y), (-1, 1))
    aic = model.aic(y)
    bic = model.bic(y)
    means = model.means_
    covs = model.covariances_

    # Calculate model score for target star
    scores = model.predict_proba(y)

    # Assign overdensity and underdensity class to the return results of the model
    if means[0][0] == means.max():
        high = 0
        low = 1
    else:
        high = 1
        low = 0

    # Save all scores to an array
    scores = [scores[0][low], scores[0][high],
               means[low][0], means[high][0],
               covs[low][0][0], covs[high][0][0],
               aic, bic]

    if scores_only:
        return scores
    return model, scores


def remove_outliers(x, sigma=2, return_outliers=False):
    """
    Remove density outliers to avoid spurious fitting results.

    :param x: List of phase space densities.
    :param sigma: Number of standard deviations used to remove outliers outside given value.
    :param return_outliers: Bool to return outliers.

    :return: Depending on return_outliers parameter return only cleaned data or cleaned data and outliers as a
    separate value.
    """

    y = np.expand_dims(np.log10(x), axis=0).T

    # Calculate mean and sigma for the dataset
    m = np.mean(y)
    std = np.std(y)

    outliers = y[np.where((y < m - (sigma * std)) | (y > m + (sigma * std)))]
    # remove stars outside sigma
    y = y[np.where((y > m - (sigma * std)) & (y < m + (sigma * std)))]

    outliers = np.reshape(outliers, (outliers.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))

    if return_outliers:
        return y, outliers
    return y
