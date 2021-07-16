import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def star_neighbours_2d(set1, set2):
    sns.scatterplot(x=set2["x"], y=set2["y"], color="g")
    sns.scatterplot(x=set1["x"], y=set1["y"], color="r")
    plt.legend(["set2", "set1"])
    plt.show()