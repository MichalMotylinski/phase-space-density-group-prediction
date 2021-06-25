from numba import njit, prange
import numpy as np
import time


@njit(parallel=False)
def to_sets(target, gaia, dist1=40, dist2=80):
    """
    Create two sets of neigbours of the target star within given range (by default 40pc and 80pc).

    :param target: phase space coordinates of the target star
    :param gaia: List of phase space coordinates for all stars in Gaia dataset
    :param dist1: First neighbourhood radius.
    :param dist2: Second neighbourhood radius.

    :return: Return 2 sets of neighbours according to the supplied dist1 and dist2 params
    """

    set1 = []
    set2 = []
    for i in range(gaia.shape[0]):
        # Calculate distance with an equivalent of np.linalg.norm function which is not supported by numba.
        z = np.zeros(shape=target[:3].shape)
        for j in range(target[:3].shape[0]):
            z[j] = target[j] - gaia[i][j]
        dist = np.sqrt(np.sum(z ** 2, 0))

        # Check if value fits into a predefined range and if so add value to an appropriate array.
        # TO DO: python lists are very inefficient and should be replaced.
        if dist < dist1:
            set1.append(gaia[i][:])

        if dist < dist2:
            set2.append(gaia[i][:])

    return set1, set2


@njit(parallel=True)
def calc_mah(set1_star, set2, set2_inv):
    """
    Calculate mahalanobis distance to all neighbours of the target star and select 20th closest.

    :param set1_star: Set1 star coordinates.
    :param set2: Entire second set of neighbours.
    :param set2_inv: Inverse second set of neighbours.

    :return: Mahalanobis distance to the 20th closest neighbour of the target.
    """
    # Create an array with infinite numbers. for 20 closest stars.
    # The idea here is that as the new values are being calculated the largest values will be replaced in the list.
    mahal_dist = np.full(20, np.inf, dtype="float64")

    for j in prange(set2.shape[0]):
        # Below is a manual implementation of mahalanobis distance as numba does not support scipy function.

        delta = set1_star - set2[j]
        m = np.dot(np.dot(delta, set2_inv), delta)

        # New value is compared to currently stored values and it replaces the highest value if new value < highest value in the set.
        if m < mahal_dist.max():
            mahal_dist[mahal_dist.argmax()] = m

    # Return 20th closest neighbour, which in this case is the last value in the set.
    return np.sqrt(mahal_dist.max())


@njit
def calc_dense(mah_dist_arr, dims=6):
    """
    Calculate phase space density for each neighbour.

    :param mah_dist_arr: List of mahalanobis distances from target star to all neighbours.
    :param dims: Number of dimensions (phase space density coordinates used to calculate mahalanobis distance).

    :return: List of phase space densities of all neighbours.
    """
    density = 20 / mah_dist_arr ** dims
    norm_density = density/np.median(density)
    return norm_density


def get_densities(hosts, gaia):
    """
    Calculate phase space density for a list of stars supplied in first argument

    :param hosts: List of exoplanet host stars
    :param gaia: List of cartesian coordinates for all gaia stars

    :return: List of phase space densities where first entry is target star and remaining values are its neighbours
    """

    densities = []
    o = 0
    for i in range(hosts.shape[0]):
        o = o+1
        target = gaia[i]

        # Generate sets of star neighbours
        set1, set2 = to_sets(target, gaia)
        set1 = np.array(set1)
        set2 = np.array(set2)

        # Get id of the target star and remove it from set2
        host_idx = np.where(set1 == target)[0][0]
        set2 = np.delete(set2, np.where(set2 == target)[0][0], 0)

        # Create inverted covariance matrix of set2
        set2_inv = np.atleast_2d(np.linalg.inv(np.cov(set2.T)))

        # Calculate mahalanobis distance for all neighbours from set1
        mah_dist_arr = np.zeros(set1.shape[0])
        for j in range(set1.shape[0]):
            mah_dist_arr[j] = (calc_mah(set1[j], set2, set2_inv))

        # Calculate densities for each neighbour from set1
        norm_density = calc_dense(np.array(mah_dist_arr), gaia.shape[1])

        # Extract host from the list and remove it from the list for further use
        host = norm_density[host_idx]
        norm_density = np.delete(norm_density, host_idx, 0)

        # Save both target host density and its neighbours densities to an array
        densities.append([hosts[i], host, norm_density])

    return densities
