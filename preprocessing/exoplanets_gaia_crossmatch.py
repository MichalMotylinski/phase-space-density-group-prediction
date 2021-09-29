import numpy as np
from os import path
import pandas as pd


def gaia_exoplanets_cross(gaia_filename, crossmatch_dir, save_gaia_id=False, return_data=False, save_spherical=True):
    """
    Cross match Gaia dataset with NASA exoplanet dataset.

    :param gaia_filename: Name of the file to read Gaia information from
    :param crossmatch_dir: Path to crossmatch directory
    :param save_gaia_id: Bool to save source_id values in a separate file
    :param return_data: Return crossmatched gaia data for further use
    :param save_spherical: Save crossmatched data to a CSV format

    :return: Cross matched dataset
    """

    # Path to downloaded datasets
    datasets_dir = "data/initial_datasets"

    # Read Exoplanets data
    exoplanets = pd.read_csv(path.join(datasets_dir, "exoplanets.csv"), skiprows=28,
                             usecols=["pl_name", "hostname", "gaia_id", "pl_orbper", "pl_orbsmax", "pl_bmasse"])
    # Process Exoplanets data
    exoplanets.dropna(subset=["gaia_id"], inplace=True)
    exoplanets["source_id"] = exoplanets["gaia_id"].str.rsplit(" ", n=1, expand=True)[1].astype("int64")
    exoplanets.drop(["gaia_id"], axis=1, inplace=True)
    exoplanets["Host"] = exoplanets["hostname"].str.replace(" ", "")
    exoplanets.drop_duplicates(subset=["Host"], inplace=True)

    # Read Gaia data
    gaia = pd.read_csv(path.join(datasets_dir, gaia_filename))

    # Add Gaia information to Exoplanet hosts
    exoplanets = pd.merge(exoplanets, gaia, on="source_id")
    exoplanets.drop(["pl_name", "hostname"], axis=1, inplace=True)

    # Remove exoplanet hosts from Gaia
    gaia = gaia[~gaia["source_id"].isin(exoplanets["source_id"])]
    # Further reduction of the data
    gaia = gaia[4.5 < gaia["parallax"] / gaia["parallax_error"]]

    # Concatenate exoplanet hosts back, however at the top of the dataframe. This way for testing purposes we later
    # iterate only over first 1065 entries that are exoplanet hosts.
    gaia = pd.concat([exoplanets, gaia])

    # Calculate distance in pc and drop any stars with negative or null distance
    gaia["distance_pc"] = (1. / gaia["parallax"]) * 1000
    gaia = gaia[gaia["distance_pc"] > 0]

    # Convert from degrees to pc
    gaia["ra"] = (gaia["ra"] * np.pi) / 180.
    gaia["dec"] = (gaia["dec"] * np.pi) / 180.

    if save_gaia_id:
        gaia[["source_id", "Host"]].to_csv(path.join(crossmatch_dir, f"{gaia_filename.split('.')[0]}_star_labels.csv"), index=False)

    # Drop all unnecessary data leaving only 6 coordinates, their errors and distance
    gaia.drop(gaia.columns[:5], axis=1, inplace=True)

    # Save transformed data to a new file
    if save_spherical:
        gaia.to_csv(path.join(crossmatch_dir, f"{gaia_filename.split('.')[0]}_exoplanet_cross_spherical.csv"),
                    index=False)
        exoplanets.to_csv(path.join(crossmatch_dir, f"{gaia_filename.split('.')[0]}_exoplanet_hosts.csv"), index=False)

    if return_data:
        return gaia


def transform_to_cart(gaia, table_name, crossmatch_dir, setting="6d", predicted_radial_velocity=None):
    """

    :param gaia: Gaia dataset
    :param table_name: Name of the Gaia table used
    :param setting: Option used to convert either from 5D or 6D spherical to cartesian
    :param predicted_radial_velocity: Optional - True when using predicted radial velocity coordinate
    :return: Return Gaia dataset with converted coordinates
    """

    # First 3 coordinates remain the same for all options
    gaia["x"], gaia["y"], gaia["z"] = sph2cart(gaia["distance_pc"], gaia["ra"], gaia["dec"])

    if predicted_radial_velocity:
        # 5D coords with predicted radial velocity [hard]
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(predicted_radial_velocity, gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
        setting = "predicted_rv"

    if setting == "6d":
        # Convert from spherical to cartesian coordinates [easy]
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(gaia["dr2_radial_velocity"], gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
    elif setting == "5d_drop_rv":
        # 5D coords [average]
        gaia[["vx", "vy"]]= gaia[["pmra", "pmdec"]]
        gaia.drop(gaia.columns[:-5], axis=1, inplace=True)
    elif setting == "5d_drop_vz":
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(gaia["dr2_radial_velocity"], gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
        gaia.drop(["vz"], axis=1, inplace=True)
    elif setting == "5d_drop_vy":
        # 5D coords [average]
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(gaia["dr2_radial_velocity"], gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
        gaia.drop(["vy"], axis=1, inplace=True)
    elif setting == "5d_drop_vx":
        # 5D coords [average]
        gaia["vx"], gaia["vy"], gaia["vz"] = vsph2cart(gaia["dr2_radial_velocity"], gaia["pmra"], gaia["pmdec"],
                                                       gaia["distance_pc"], gaia["ra"], gaia["dec"])
        gaia.drop(gaia.columns[:-6], axis=1, inplace=True)
        gaia.drop(["vx"], axis=1, inplace=True)

    # Save to CSV
    gaia.to_csv(path.join(crossmatch_dir, f"{table_name}_exoplanet_cross_cartesian_{setting}.csv"), index=False)

    return gaia


# Function to convert positions from spherical to cartesian coordinates
def sph2cart(r, ra, dec):
    x = r * np.cos(ra) * np.cos(dec)
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
    xdot = np.cos(ra) * np.cos(dec) * rdot - r * np.sin(ra) * np.cos(dec) * radot - r * np.cos(ra) * np.sin(dec) * decdot
    ydot = np.sin(ra) * np.cos(dec) * rdot + r * np.cos(ra) * np.cos(dec) * radot - r * np.sin(ra) * np.sin(dec) * decdot
    zdot = np.sin(dec) * rdot + r * np.cos(dec) * decdot

    return xdot, ydot, zdot
