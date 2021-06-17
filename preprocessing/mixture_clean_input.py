import numpy as np


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

    outliers = x[np.where((x < m - (sigma * std)) | (x > m + (sigma * std)))]
    x = x[np.where((x > m - (sigma * std)) & (x < m + (sigma * std)))]

    outliers = np.reshape(outliers, (outliers.shape[0], 1))
    x = np.reshape(x, (x.shape[0], 1))

    if return_outliers:
        return x, outliers
    return x