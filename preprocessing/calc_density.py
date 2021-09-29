import math
from numba import njit, prange, cuda, float64
import numpy as np

# Global variables for cuda functions input
N_PARAMS = 6
N_DIST_CLOSE = 40
N_DIST_FAR = 80


@njit(parallel=False)
def to_sets_cpu(target, gaia):
    """
    Create two sets of neigbours of the target star within given range (by default 40pc and 80pc).

    :param target: Phase space coordinates of the target star
    :param gaia: List of phase space coordinates for all stars in Gaia dataset

    :return: Return 2 sets of neighbours according to the supplied dist1 and dist2 params
    """

    set1 = []
    set2 = []
    for i in range(gaia.shape[0]):
        # Calculate distance from target star to its neighbour
        z = np.zeros(shape=target[:3].shape)
        for j in range(target[:3].shape[0]):
            z[j] = target[j] - gaia[i][j]

        dist = np.sqrt(np.sum(z ** 2, 0))

        # Check if value fits into a predefined range and if so add value to an appropriate array.
        if dist < N_DIST_CLOSE:
            set1.append(gaia[i][:])

        if dist < N_DIST_FAR:
            set2.append(gaia[i][:])

    return set1, set2


@cuda.jit
def to_sets_gpu(target, gaia, set1, set2):
    """
    Create two sets of neigbours of the target star within given range (by default 40pc and 80pc).

    :param target: Phase space coordinates of the target star
    :param gaia: List of phase space coordinates for all stars in Gaia dataset
    :param set1: Numpy array to store list of neighbours within dist1.
    :param set2: Numpy array to store list of neighbours within dist2.
    """

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, gaia.shape[0], stride):
        # Calculate distance from target star to its neighbour
        s = 0
        for j in range(3):
            z = (target[j] - gaia[i][j]) ** 2
            s = s + z
        dist = math.sqrt(s)

        # Check if value fits into a predefined range and if so add value to an appropriate array.
        if dist < N_DIST_CLOSE:
            for k in range(set1.shape[1]):
                set1[i][k] = gaia[i][k]
        if dist < N_DIST_FAR:
            for k in range(set2.shape[1]):
                set2[i][k] = gaia[i][k]


@njit(parallel=True)
def calc_mah_cpu(set1_star, set2, set2_inv):
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


@cuda.jit
def calc_mah_gpu(set1, set2, set2_inv, dists):
    """
    Calculate mahalanobis distance to all neighbours of the target star using 6D coordinates and select 20th closest.

    :param set1: Entire first set of neighbours.
    :param set2: Entire second set of neighbours.
    :param set2_inv: Inverse second set of neighbours.
    :param dists: Array for calculated mahalanobis distances.
    """

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Loop over all close neighbours
    for i in range(start, set1.shape[0], stride):
        dist = cuda.local.array(shape=(20,), dtype=float64)
        for k in range(dist.shape[0]):
            dist[k] = 99999.
        max_val = 0

        delta = cuda.local.array(shape=(N_PARAMS,), dtype=float64)
        z = cuda.local.array(shape=(N_PARAMS,), dtype=float64)

        for j in range(set2.shape[0]):
            # Calculate mahalanobis distance
            # Subtract arrays
            for k in range(set2.shape[1]):
                delta[k] = set1[i][k] - set2[j][k]

            # Compute first dot product
            for k in range(delta.shape[0]):
                s = 0
                for g in range(set2_inv.shape[0]):
                    a = delta[g] * set2_inv[g][k]
                    s = s + a
                z[k] = s

            # Compute second dot product
            s = 0
            for k in range(z.shape[0]):
                a = z[k] * delta[k]
                s = s + a

            # Fill an array with the first 20 values and then continue comparing current value to the maximum value in
            # the array. If current value is lower then add it to array in place of the maximum value
            if j < 20:
                dist[j] = s
            elif j == 20:
                for k in range(dist.shape[0]):
                    if dist[k] > max_val:
                        max_val = dist[k]

            if max_val > s:
                for k in range(dist.shape[0]):
                    if dist[k] == max_val:
                        dist[k] = s
                        max_val = 0
                        for g in range(dist.shape[0]):
                            if dist[g] > max_val:
                                max_val = dist[g]
                        break

        # Save maximum value (20th nearest neighbour) to the list
        dists[i] = math.sqrt(max_val)


@njit
def calc_dense(mah_dist_arr, dims=6, nth_star=20):
    """
    Calculate phase space density for each neighbour.

    :param mah_dist_arr: List of mahalanobis distances from target star to all neighbours.
    :param dims: Number of dimensions (phase space density coordinates used to calculate mahalanobis distance).
    :param nth_star: Nth closest neighbour distance.

    :return: List of phase space densities of all neighbours.
    """

    density = nth_star / mah_dist_arr ** dims
    norm_density = density / np.median(density)
    return norm_density


def get_densities(labels, gaia, start=0, stop=1000, step=1, run_on_gpu=False):
    """
    Calculate phase space density for a list of stars supplied in first argument.

    :param labels: List of star labels (source_id) from Gaia dataset.
    :param gaia: List of cartesian coordinates for all gaia stars.
    :param start: Start from given integer.
    :param stop: Stop at given integer.
    :param step: Increment value.
    :param run_on_gpu: Set True to run pipeline on GPU.

    :return: List of target stars with their corresponding densities and densities of their neighbours.
    Second value returns list of dropped stars that have less than 400 neighbours within given distance.
    """

    global N_PARAMS

    dropped = []
    densities = []
    for i in range(start, stop, step):
        # Generate sets of star neighbours
        if run_on_gpu:
            target = np.ascontiguousarray(gaia[i])
            set1 = np.zeros(shape=gaia.shape)
            set2 = np.zeros(shape=gaia.shape)
            to_sets_gpu[16, 32](target, gaia, set1, set2)
            set1 = set1[~np.all(set1 == 0, axis=1)]
            set2 = set2[~np.all(set2 == 0, axis=1)]
        else:
            target = gaia[i]
            set1, set2 = to_sets_cpu(target, gaia)
            set1 = np.array(set1)
            set2 = np.array(set2)

        # Drop stars with less than 400 neighbours
        if set1.shape[0] < 400:
            dropped.append([labels[i], set1.shape[0], set2.shape[0]])
            continue

        # Get id of the target star and remove it from set2
        host_idx = np.where(set1 == target)[0][0]
        set2 = np.delete(set2, np.where(set2 == target)[0][0], 0)

        # Create inverted covariance matrix of set2
        set2_inv = np.atleast_2d(np.linalg.inv(np.cov(set2.T)))

        # Calculate mahalanobis distance for all neighbours from set1
        if run_on_gpu:
            N_PARAMS = set1.shape[1]
            mah_dist_arr = np.zeros(shape=(set1.shape[0],))
            calc_mah_gpu[80, 96](set1, set2, set2_inv, mah_dist_arr)
        else:
            mah_dist_arr = np.zeros(set1.shape[0], dtype="float64")
            for j in range(set1.shape[0]):
                mah_dist_arr[j] = (calc_mah_cpu(set1[j], set2, set2_inv))

        # Calculate densities for each neighbour from set1
        norm_density = calc_dense(mah_dist_arr, gaia.shape[1])

        # Extract host from the list and remove it from the list for further use
        host = norm_density[host_idx]
        norm_density = np.delete(norm_density, host_idx, 0)

        # Save both target host density and its neighbours densities to an array
        densities.append([labels[i], host, set1.shape[0], set2.shape[0], norm_density])

    return densities, dropped
