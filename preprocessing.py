from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
import coord_transform as coord
import re
from io import StringIO
from numba import njit

import models
import graphs


class GaiaDataset:
    """
    Use this class to fetch GAIA dataset from ESA.
    """

    def __init__(self, query=None, table="gaiaedr3.gaia_source", filename="gaiadr3_table"):
        self.query = query
        self.filename = filename
        self.table = table

    def __login(self):
        """
        Initiate UI frame for user to log in to ESA servers.
        """

        Gaia.login_gui()

    def __default_query(self):
        """
        Default query used only if no query supplied by the user.
        """

        self.query = f"""
        select source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, 
        dr2_radial_velocity, dr2_radial_velocity_error from {self.table} where source_id is not null and ra is 
        not null and ra_error is not null and dec is not null and dec_error is not null and parallax is not null and 
        parallax_error is not null and pmra is not null and pmra_error is not null and pmdec is not null and 
        pmdec_error is not null and dr2_radial_velocity is not null and dr2_radial_velocity_error is not null
        """

    def get_gaia(self, query=None):
        """
        Fetch data from GAIA table
        :param query: SQL query supplied by user
        """

        self.__login()
        if query is None:
            self.__default_query()

        if self.filename[-4:] != ".csv":
            self.filename = self.filename + ".csv"

        Gaia.launch_job_async(self.query).get_results().to_pandas().to_csv(self.filename, index=False)


def load_winter():
    """
    Read table with results from Andrew Winter. The table was saved in txt format.

    :returns: Pandas dataframe consisting of the following columns: exoplanet host name, number of stars, logPnull,
     logbetaM20, logPhigh.
    """

    data = ""
    with open("table1.dat.txt", "r") as f:
        for line in f:

            if line[30:33] == "   ":
                line = line[:30] + "---" + line[33:]
            if line[41:44] == "   ":
                line = line[:41] + "---" + line[44:]

            line = line[:20].replace(" ", "") + line[20:]
            line = line.replace(" ", ",")
            line = line.replace("|", ",")

            for i in line[:15]:
                if i == ",":
                    if line[line.index(i) - 1] != "," and line[line.index(i) + 1] != ",":
                        line = line[:line.index(i)] + " " + line[line.index(i) + 1:]
                    else:
                        break

            line = re.sub('\,+', ',', line)

            data = data + line
    labels = ["Host", "mass", "mass_error", "age", "age_error", "nstars", "logPnull", "BIC1-2", "logbetaM20",
              "logPhigh", "HJ", "Include"]
    hosts = pd.read_csv(StringIO(data), sep=",", names=labels)

    return hosts[["Host", "nstars", "logPnull", "logbetaM20", "logPhigh"]]


def gaia_exoplanets_cross(gaia_filename):
    """
    Cross match Gaia dataset with NASA exoplanet dataset and winter results (for comparison only).
    :returns: Cross matched dataset
    """

    exoplanets = pd.read_csv("exoplanets.csv", skiprows=24,
                             usecols=["pl_name", "hostname", "gaia_id", "pl_orbper", "pl_orbsmax", "pl_bmasse"])
    exoplanets.dropna(subset=["gaia_id"], inplace=True)
    exoplanets["source_id"] = exoplanets["gaia_id"].str.rsplit(" ", n=1, expand=True)[1].astype("int64")
    exoplanets.drop(["gaia_id"], axis=1, inplace=True)
    exoplanets["Host"] = exoplanets["hostname"].str.replace(" ", "")
    exoplanets.drop_duplicates(subset=["Host"], inplace=True)

    gaia = pd.read_csv(gaia_filename)

    exoplanets = pd.merge(exoplanets, gaia, on="source_id")
    exoplanets.drop(["pl_name", "hostname"], axis=1, inplace=True)

    hosts = load_winter()
    # Save crossmatch with Winter data
    hosts = pd.merge(hosts, exoplanets, on="Host")
    hosts.to_csv("hosts.csv")

    hosts.drop(hosts.columns[0:8], axis=1, inplace=True)

    # Remove exoplanet hosts
    gaia = gaia[~gaia["source_id"].isin(hosts["source_id"])]
    # Further reduction of the data
    gaia = gaia[4.5 < gaia["parallax"] / gaia["parallax_error"]]

    # Concatenate exoplanet hosts back, however at the top of the dataframe.
    gaia = pd.concat([hosts, gaia])

    # Calculate distance in pc and drop any stars with negative or null distance
    gaia["distance_pc"] = (1. / gaia["parallax"]) * 1000
    gaia = gaia[gaia["distance_pc"] > 0]

    # Convert from degrees to pc
    gaia["ra"] = (gaia["ra"] * np.pi) / 180.
    gaia["dec"] = (gaia["dec"] * np.pi) / 180.

    # Convert from spherical to cartesian coordinates
    gaia["x"], gaia["y"], gaia["z"] = coord.sph2cart(gaia["distance_pc"], gaia["ra"], gaia["dec"])
    gaia["vx"], gaia["vy"], gaia["vz"] = coord.vsph2cart(gaia["dr2_radial_velocity"], gaia["pmra"], gaia["pmdec"],
                                                         gaia["distance_pc"], gaia["ra"], gaia["dec"])

    gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
    gaia.to_csv("gaia.csv", index=False)

    return gaia


@njit
def to_sets(target, gaia, dist1=40, dist2=80):
    """
    Create two sets of neigbours of the target star within given range (by default 40pc and 80pc).

    :param target: 6D phase space coordinates of the target star
    :param gaia: List of 6D coordinates for all stars in Gaia dataset
    :param dist1: First neighbourhood radius.
    :param dist2: Second neighbourhood radius.

    :return: Return 2 sets of neighbours according to the supplied dist1 and dist2 params
    """

    set1 = []
    set2 = []
    for j in range(gaia.shape[0]):
        # Calculate distance with an equivalent of np.linalg.norm function which is not supported by numba.
        z = np.zeros(shape=target[:3].shape)
        for i in range(len(target)):
            z[i] = target[i] - gaia[j][i]
        dist = np.sqrt(np.sum(z ** 2, 0))

        # Check if value fits into a predefined range and if so add value to an appropriate array.
        # TO DO: python lists are very inefficient and should be replaced.
        if dist < dist1:
            set1.append(gaia[j][:])

        if dist < dist2:
            set2.append(gaia[j][:])

    return set1, set2


@njit
def calc_mah(i, set1, set2, set2_inv):
    """
    Calculate mahalanobis distance to all neighbours of the target star and select 20th closest.

    :param i: Star index in set1.
    :param set1: Entire first set of neighbours.
    :param set2: Entire second set of neighbours.
    :param set2_inv: Inverse second set of neighbours.

    :return: Mahalanobis distance to the 20th closest neighbour of the target.
    """
    # Create an array with infinite numbers. for 20 closest stars.
    # The idea here is that as the new values are being calculated the largest values will be replaced in the list.
    mahal_dist = np.full(20, np.inf, dtype="float64")

    for j in range(set2.shape[0]):
        # Below is a manual implementation of mahalanobis distance as numba does not support scipy function.
        set2_inv = np.atleast_2d(set2_inv)
        delta = set1[i] - set2[j]
        m = np.dot(np.dot(delta, set2_inv), delta)

        # New value is compared to currently stored values and if it replaces the highest value if new value < highest value in the set.
        if np.sqrt(m) < mahal_dist.max():
            mahal_dist[mahal_dist.argmax()] = np.sqrt(m)

    # Return 20th closest neighbour, which in this case is the last value in the set.
    return mahal_dist.max()


@njit
def calc_dense(mah_dist_arr):
    """
    Calculate phase space density for each neighbour.

    :param mah_dist_arr: List of mahalanobis distances from target star to all neighbours.

    :return: List of phase space densities of all neighbours.
    """
    density = 20 / mah_dist_arr ** 6
    norm_density = density/np.median(density)
    return norm_density


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


def exoplanet_hosts_densities(i, gaia):
    #for i in range(hosts.shape[0]):
    """if hosts[i] != "HD285968":
        continue"""
    target = gaia[i]

    set1, set2 = to_sets(target, gaia)

    set1 = np.array(set1)
    set2 = np.array(set2)

    host_idx = np.where(set1 == target)[0][0]
    set2 = np.delete(set2, np.where(set2 == target)[0][0], 0)

    set2_cov_mat = np.cov(set2.T)  # Calculate the covariance matrix
    set2_inv = np.linalg.inv(set2_cov_mat)  # Invert

    mah_dist_arr = np.zeros(set1.shape[0])
    for j in range(set1.shape[0]):
        # Save calculated distances
        mah_dist_arr[j] = (calc_mah(j, set1, set2, set2_inv))

    # Calculate densities for each star within 40pc
    norm_density = calc_dense(np.array(mah_dist_arr))

    # Extract host from the list and remove it from the list for further use
    host = norm_density[host_idx]
    norm_density = np.delete(norm_density, host_idx, 0)

    return host, norm_density


def exoplanets_loop(hosts, gaia):
    densities = []
    for i in range(hosts.shape[0]):
        res = exoplanet_hosts_densities(i, gaia)
        densities.append([hosts[i], res[0], res[1]])

    return densities


def random_600_n(hosts, gaia):
    for i in range(hosts.shape[0]):
        if hosts[i] != "HD285968":
            continue
        target = gaia[i]

        set1, set2 = to_sets(target, gaia)

        set1 = np.array(set1)
        set2 = np.array(set2)

        host_idx = np.where(set1 == target)[0][0]

        set1 = np.random.permutation(set1)
        set1 = np.vstack((target, set1))


        set2 = np.delete(set2, np.where(set2 == target)[0][0], 0)
        #set2 = np.random.permutation(set2)[:1000]
        print(set1.shape, set2.shape)
        set2_cov_mat = np.cov(set2.T)  # Calculate the covariance matrix
        set2_inv = np.linalg.inv(set2_cov_mat)  # Invert

        if set1.shape[0] >= 600:
            mah_dist_arr = np.zeros(600)
        else:
            mah_dist_arr = np.zeros(set1.shape[0])
        for j in range(set1.shape[0]):
            if j == 600:
                break
            # Save calculated distances
            mah_dist_arr[j] = (calc_mah(j, set1, set2, set2_inv))

        # Calculate densities for each star within 40pc
        norm_density = calc_dense(np.array(mah_dist_arr))

        host = np.expand_dims(np.log10(norm_density[0]), axis=0).T
        norm_density = np.delete(norm_density, 0, 0)
        print(set1.shape, set2.shape, mah_dist_arr.shape)
        # Remove outliers (outside 2 standard deviation)
        norm_density = remove_outliers(norm_density)
        #norm_density = np.reshape(norm_density, (norm_density.shape[0], 1))
        #aic, bic, scores, means, covs = models.gaussian_mixture(norm_density, [host], 2, 1000)
        #print(aic, bic, scores, means, covs)
        model, aic, bic, scores, means, covs = models.gaussian_mixture(norm_density, [host], 2, 1000, scores_only=False)
        graphs.best_fit_mixture(model, norm_density, [hosts[i], host])
