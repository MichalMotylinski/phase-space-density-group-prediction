import numpy as np
import pandas as pd
from os import path
import os
from .read_winter import load_winter

crossmatch_dir = "data/crossmatch"


def gaia_exoplanets_cross(gaia_filename, return_data=False, save_to_file=True):
    """
    Cross match Gaia dataset with NASA exoplanet dataset and winter results (for comparison only).
    :returns: Cross matched dataset
    """

    datasets_dir = "data/initial_datasets"

    exoplanets = pd.read_csv(path.join(datasets_dir, "exoplanets.csv"), skiprows=24,
                             usecols=["pl_name", "hostname", "gaia_id", "pl_orbper", "pl_orbsmax", "pl_bmasse"])
    exoplanets.dropna(subset=["gaia_id"], inplace=True)
    exoplanets["source_id"] = exoplanets["gaia_id"].str.rsplit(" ", n=1, expand=True)[1].astype("int64")
    exoplanets.drop(["gaia_id"], axis=1, inplace=True)
    exoplanets["Host"] = exoplanets["hostname"].str.replace(" ", "")
    exoplanets.drop_duplicates(subset=["Host"], inplace=True)

    gaia = pd.read_csv(path.join(datasets_dir, gaia_filename))

    exoplanets = pd.merge(exoplanets, gaia, on="source_id")
    exoplanets.drop(["pl_name", "hostname"], axis=1, inplace=True)
    """hosts = load_winter()
    # Save crossmatch with Winter data
    hosts = pd.merge(hosts, exoplanets, on="Host")
    hosts.to_csv("hosts.csv")
    hosts.drop(hosts.columns[0:8], axis=1, inplace=True)"""
    print(exoplanets)
    # Remove exoplanet hosts
    gaia = gaia[~gaia["source_id"].isin(exoplanets["source_id"])]
    # Further reduction of the data
    gaia = gaia[4.5 < gaia["parallax"] / gaia["parallax_error"]]

    # Concatenate exoplanet hosts back, however at the top of the dataframe.
    gaia = pd.concat([exoplanets, gaia])

    # Calculate distance in pc and drop any stars with negative or null distance
    gaia["distance_pc"] = (1. / gaia["parallax"]) * 1000
    gaia = gaia[gaia["distance_pc"] > 0]

    # Convert from degrees to pc
    gaia["ra"] = (gaia["ra"] * np.pi) / 180.
    gaia["dec"] = (gaia["dec"] * np.pi) / 180.

    gaia.drop(gaia.columns[:5], axis=1, inplace=True)

    if save_to_file:
        gaia.to_csv(path.join(crossmatch_dir, f"{gaia_filename.split('.')[0]}_exoplanet_cross_spherical.csv"),
                    index=False)
        exoplanets.to_csv(path.join(crossmatch_dir, "exoplanet_hosts.csv"), index=False)

    if return_data:
        return gaia


def transform_to_cart(gaia, table_name, setting="6d", predicted_radial_velocity=None):
    gaia["x"], gaia["y"], gaia["z"] = sph2cart(gaia["distance_pc"], gaia["ra"], gaia["dec"])
    if predicted_radial_velocity:
        # 5D coords with predicted radial velocity [hard]
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(predicted_radial_velocity, gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
        gaia.to_csv(path.join(crossmatch_dir, f"{table_name}_exoplanet_cross_cartesian_predicted_rv.csv"), index=False)

    if setting == "6d":
        # Convert from spherical to cartesian coordinates [easy]
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(gaia["dr2_radial_velocity"], gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
        gaia.to_csv(path.join(crossmatch_dir, f"{table_name}_exoplanet_cross_cartesian_6d.csv"), index=False)
    else:
        # 5D coords [average]
        gaia[["vx", "vy"]]= gaia[["pmra", "pmdec"]]
        gaia.drop(gaia.columns[:-5], axis=1, inplace=True)
        gaia.to_csv(path.join(crossmatch_dir, f"{table_name}_exoplanet_cross_cartesian_5d.csv"), index=False)

    return gaia


# Function to convert positions from spherical to cartesian coordinates
def sph2cart(r, ra, dec):
    x = r * np.cos(ra) * np.cos(dec)  # Why is this not sin(ra)*sin(dec)?
    y = r * np.sin(ra) * np.cos(dec)
    z = r * np.sin(dec)

    return x, y, z


# Function to convert positions from cartesian to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    dec = np.arcsin(z / r)
    ra = np.arctan2(y, x)

    return r, ra, dec


# Function to convert velocities from cartesian to spherical coordinates
def vcart2vsph(vx, vy, vz, x, y, z):
    r = np.sqrt(x * x + y * y * z * z)
    R = np.sqrt(x * x + y * y)
    rdot = x * vx + y * vy + z * vz
    rdot /= r
    radot = vx * y - vy * x
    radot /= R * R
    decdot = z * (x * vx + y * vy) - R * R * vz
    decdot /= (R * r * r)

    return rdot, radot, decdot


# Function to convert velocities from spherical to cartesian coordinates
def vsph2cart(rdot, radot, decdot, r, ra, dec):
    xdot = np.cos(ra) * np.cos(dec) * rdot - \
           r * np.sin(ra) * np.cos(dec) * radot \
           - r * np.cos(ra) * np.sin(dec) * decdot
    ydot = np.sin(ra) * np.cos(dec) * rdot + \
           r * np.cos(ra) * np.cos(dec) * radot - \
           r * np.sin(ra) * np.sin(dec) * decdot
    zdot = np.sin(dec) * rdot + r * np.cos(dec) * decdot

    return xdot, ydot, zdot