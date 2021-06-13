import re
from io import StringIO

import pandas as pd


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
