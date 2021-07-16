from astroquery.gaia import Gaia


class GaiaDataset:
    """
    Use this class to fetch GAIA dataset from ESA.
    """

    def __init__(self, query=None, table="gaiaedr3.gaia_source", filename="gaiaedr3"):
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

        Gaia.launch_job_async(self.query).get_results().to_pandas().to_csv("data/initial_datasets/" + self.filename,
                                                                           index=False)
