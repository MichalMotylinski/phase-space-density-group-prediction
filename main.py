import pandas as pd
from pathlib import Path
import time
from os import path

from preprocessing.exoplanets_gaia_crossmatch import gaia_exoplanets_cross, transform_to_cart
from preprocessing.download_gaia import GaiaDataset
from preprocessing.calc_density import get_densities
from models.gaussian_mixture import remove_outliers, gaussian_mixture
from models.classification import random_forest as rfc
from models.regression import random_forest as rfr
from models.regression import RadialVelocityRegression, ann
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

import graphs


def download_gaiadr2():
    query = f"""
            select source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, 
            pmdec_error, radial_velocity, radial_velocity_error from gaiadr2.gaia_source where source_id is not null 
            and ra is not null and ra_error is not null and dec is not null and dec_error is not null and parallax 
            is not null and parallax_error is not null and pmra is not null and pmra_error is not null and pmdec is 
            not null and pmdec_error is not null and radial_velocity is not null and radial_velocity_error is not null
            """

    g = GaiaDataset(query=query, filename="gaiadr2_table")
    g.get_gaia()


def exoplanet_gaia_crossmatch(transform_type="6d"):
    """
    This function is for testing purposes only.
    The entire pipeline below takes considerable amount of time to compute (~8 hours depending on hardware)
    This function can be used to generate new data.
    Please use use_precomputed_6d_densities() to work on precomputed density values.

    :return: Density values for 1065 exoplanets and their neighbours, Winter-Gaia-NASA exoplanet archive crossmatch
     table containing 6D coordinates only and Winter-Gaia-NASA exoplanet archive crossmatch table with data from all 3
      sources combined.
    """

    crossmatch_dir = "data/crossmatch"

    # Cross match datasets and generate new ones.
    if Path(path.join(crossmatch_dir, "gaiadr2_exoplanet_cross_spherical.csv")).is_file():
        gaia = pd.read_csv(path.join(crossmatch_dir, "gaiadr2_exoplanet_cross_spherical.csv"))
    else:
        gaia = gaia_exoplanets_cross("gaiadr2.csv", return_data=True)
    hosts = pd.read_csv(path.join(crossmatch_dir, "exoplanet_hosts.csv")).Host.to_numpy()

    if Path(path.join(crossmatch_dir, f"gaiadr2_exoplanet_cross_cartesian_{transform_type}.csv")).is_file():
        gaia = pd.read_csv(path.join(crossmatch_dir, f"gaiadr2_exoplanet_cross_cartesian_{transform_type}.csv")).to_numpy()
    else:
        gaia = transform_to_cart(gaia, "gaiadr2", setting=transform_type).to_numpy()

    return hosts, gaia


def calculate_densities(transform_type="5d"):

    hosts, gaia = exoplanet_gaia_crossmatch(transform_type)
    densities = get_densities(hosts, gaia)

    with open(f"data/densities/densities_{transform_type}.data", "wb") as f:
        pickle.dump(densities, f)

    return densities


def derive_classification_features(transform_type="5d"):
    """
    Please use this function to work on precomputed density list for 1065 exoplanets.
    :return:
    """

    with open(f"data/densities/densities_{transform_type}.data", "rb") as f:
        densities = pickle.load(f)

    o = 0
    results = []
    dropped = []
    for i in densities:
        if i[2].shape[0] < 10:
            dropped.append(i)
            continue
        # Compute log10 of the host density and expand dimensions for further use
        target = np.expand_dims(np.log10(i[1]), axis=0).T

        # Remove outliers outside sigma
        data = remove_outliers(i[2], sigma=2)

        # Apply gaussian mixture model to the data
        model, scores = gaussian_mixture(data, [target], components=2, scores_only=False)

        # Create list consisting of star name and its density for graph drawing
        scores.insert(0, i[1])
        scores.insert(0, i[0])

        results.append(scores)

        o = o+1
        #results.append()
        # Draw best fit mixture
        host = [i[0], i[1]]
        #graphs.best_fit_mixture(model, data, host)

    with open(f"data/densities/dropped_densities_{transform_type}.data", "wb") as f:
        pickle.dump(dropped, f)
    df = pd.DataFrame(results, columns=["Host", "density", "Plow", "Phigh", "mean_low", "mean_high", "cov_low",
                                        "cov_high", "aic", "bic"])
    df.to_csv(f"data/classification/features_{transform_type}.csv")


def labels(row):
    if row['Plow'] >= 0.84:
        return '0'
    elif row['Phigh'] >= 0.84:
        return '2'
    else:
        return '1'


def classification_task():

    results_dir = "results/classification"
    transform_type = "5d"
    d6 = pd.read_csv("6d_data", index_col=0)
    d5 = pd.read_csv("5d_data", index_col=0)

    d6["class"] = d6.apply(lambda row: labels(row), axis=1)
    label = d6["class"]
    d5 = d5[["density", "Phigh"]]

    scores = rfc(d5, label, transform_type)
    columns = ["Classifier", "Features", "Accuracy", "Precision", "Recall", "F1-score"]
    df = pd.DataFrame([scores], columns=columns)

    if Path(path.join(results_dir, f"rf_{transform_type}.csv")).is_file():
        df.to_csv(path.join(results_dir, f"rf_{transform_type}.csv"), mode='a', header=False, index=False)
    else:
        df.to_csv(path.join(results_dir, f"rf_{transform_type}.csv"), index=False)


def regression_task():
    crossmatch_dir = "data/crossmatch"

    df = pd.read_csv(path.join(crossmatch_dir, "gaiadr2_exoplanet_cross_spherical.csv"))
    df = df[:1000000]

    output = df["dr2_radial_velocity"]
    df = pd.read_csv(path.join(crossmatch_dir, "gaiadr2_exoplanet_cross_cartesian_5d.csv"))
    df = df[:1000000]

    rfr(df, output)
    #ann(df, output)
    """x_train, y_train, x_val, y_val, x_test, y_test = RadialVelocityRegression().train_test_split(df, output)
    model = RadialVelocityRegression().assemble_model(df.shape[1])
    history, model = RadialVelocityRegression().train_model(x_train, y_train, x_val, y_val, model)
    
    y_pred = model.predict(x_test)

    print(y_test[:20])
    print("A")
    print(y_pred[0][:20])

    print(('R2 score for velocity: ', r2_score(y_test, y_pred[0])))"""



regression_task()

"""start = time.perf_counter()
#calculate_densities("5d")
derive_classification_features("5d")
end = time.perf_counter()
print(end-start)"""

"""start = time.perf_counter()
calculate_densities("6d")
derive_classification_features("6d")
end = time.perf_counter()
print(end-start)"""
#classification_task()
"""with open(f"data/densities/densities_5d.data", "rb") as f:
    densities = pickle.load(f)
l = []
sizes = []
for i in densities:
    l.append(i[2])

    if i[2].shape[0] > 50:
        sizes.append(i[2].shape[0])
for i in sizes:
    if i < 100:
        print(i)
print(sizes)"""