import pandas as pd
from pathlib import Path
import time
import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import models
import graphs


def download_gaiadr2():
    query = f"""
            select source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, 
            radial_velocity, radial_velocity_error from gaiadr2.gaia_source where source_id is not null and ra is 
            not null and ra_error is not null and dec is not null and dec_error is not null and parallax is not null and 
            parallax_error is not null and pmra is not null and pmra_error is not null and pmdec is not null and 
            pmdec_error is not null and radial_velocity is not null and radial_velocity_error is not null
            """

    g = preprocessing.GaiaDataset(query=query, filename="gaiadr2_table")
    g.get_gaia()


def exoplanet_pipeline_with_winter_crossmatch():
    """
    This function is for testing purposes only.
    The entire pipeline below takes considerable amount of time to compute (~8 hours depending on hardware)
    This function can be used to generate new data.
    Please use use_precomputed_6d_densities() to work on precomputed density values.

    :return: Density values for 1065 exoplanets and their neighbours, Winter-Gaia-NASA exoplanet archive crossmatch
     table containing 6D coordinates only and Winter-Gaia-NASA exoplanet archive crossmatch table with data from all 3
      sources combined.
    """
    # Cross match datasets and generate new ones.
    if Path("gaia.csv").is_file():
        gaia = pd.read_csv("gaia.csv").to_numpy()
    else:
        gaia = preprocessing.gaia_exoplanets_cross("gaiadr2_table.csv").to_numpy()
    hosts = pd.read_csv("hosts.csv").Host.to_numpy()

    densities = preprocessing.exoplanets_loop(hosts, gaia)

    with open("densities.data", "wb") as f:
        pickle.dump(densities, f)


def crossmatch_only():
    """
    Use this function to generate crossmatch tables only.
    :return:
    """
    preprocessing.gaia_exoplanets_cross("gaiadr2_table.csv").to_numpy()


def use_precomputed_6d_densities():
    """
    Please use this function to work on precomputed density list for 1065 exoplanets.
    :return:
    """
    with open("densities.data", "rb") as f:
        densities = pickle.load(f)

    start = time.perf_counter()
    o = 0
    for i in densities:
        # Uncomment below if statement to target specific star.
        """if i[0] != "WASP-12":
            continue"""
        # Compute log10 of the host density and expand dimensions for further use
        target = np.expand_dims(np.log10(i[1]), axis=0).T

        # Remove outliers outside sigma
        data = preprocessing.remove_outliers(i[2], sigma=2)

        # Apply gaussian mixture model to the data
        model, aic, bic, scores, means, covs = models.gaussian_mixture(data, [target], components=2, scores_only=False)

        # Create list consisting of star name and its density for graph drawing
        host = [i[0], target]

        # Draw best fit mixture
        graphs.best_fit_mixture(model, data, host)

    end = time.perf_counter()
    print(end-start)


crossmatch_only()
use_precomputed_6d_densities()