import math
import matplotlib.pyplot as plt
import numpy as np
import os


def best_fit_mixture(model, data, host, phigh, fig_dir=None, show_graph=False, save_graph=False):
    """
    Draw combined best fit mixture.

    :param model: Gaussian Mixture model object
    :param data: Calculated densities
    :param host: Target star
    :param phigh: Value of Phigh
    :param fig_dir: Path to graph directory
    :param show_graph: Bool to output the graphs on screen
    :param save_graph: Bool to save graphs in a folder

    :return: Plot a best fit mixture graph.
    """

    fig = plt.figure(figsize=(50, 5), facecolor="w")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)
    ax = fig.add_subplot(131)

    x = np.linspace(math.floor(data.min()), math.ceil(data.max()), data.shape[0])
    logprob = model.score_samples(x.reshape(-1, 1))

    responsibilities = model.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.axvline(x=host[1], color="r", label=f"{host[0]}: {host[1][0]}", ymax=0.7)

    ax.hist(data, 30, density=True, histtype="stepfilled", alpha=0.5)

    # Add combined kde line
    ax.plot(x, pdf, "-k")
    # Add individual lines for low and high density
    ax.plot(x, pdf_individual, "--k")

    plt.title("Best-fit Mixture")
    ax.set_xlabel("Phase space density")
    ax.set_ylabel("Probability density function")
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, max(pdf) + 0.2, 0.2))

    ax.text(0.01, 0.9, f"{host[0]}: {format(host[1][0], '.2g')}", fontsize=14, transform=ax.transAxes)
    ax.text(0.01, 0.8, r"$P_{high}$" + f" = {format(phigh, '.2g')}", fontsize=14, transform=ax.transAxes)

    fig.set_size_inches(30, 5)
    if save_graph:
        i = 0
        while os.path.exists(f"figures/{fig_dir}/{host[0]}_{i}.png"):
            i += 1
        plt.savefig(f"figures/{fig_dir}/{host[0]}_{i}", dpi=100, bbox_inches="tight", pad_inches=0.1)
    if show_graph:
        plt.show()
    fig.clf()
    plt.close()


def combined_fit_mixture(model, data, host, n_gaussians, fig_dir=None, show_graph=False, save_graph=False):
    """
    Draw combined best fit mixture.

    :param model: Gaussian Mixture model object
    :param data: Calculated densities
    :param host: Target star
    :param n_gaussians: Number of gaussian mixture graphs to draw
    :param fig_dir: Path to graph directory
    :param show_graph: Bool to output the graphs on screen
    :param save_graph: Bool to save graphs in a folder

    :return: Plot a best fit mixture graph for multiple sets of data.
    """

    fig = plt.figure(figsize=(50, 5), facecolor="w")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)
    ax = fig.add_subplot(131)
    colors = ["r", "b", "g", "k", "y"]
    for i in range(n_gaussians):
        x = np.linspace(math.floor(data[i].min()), math.ceil(data[i].max()), data[i].shape[0])
        logprob = model[i].score_samples(x.reshape(-1, 1))

        responsibilities = model[i].predict_proba(x.reshape(-1, 1))
        pdf = np.exp(logprob)

        pdf_individual = responsibilities * pdf[:, np.newaxis]

        plt.axvline(x=host[i][1], color=colors[i], label=f"{host[i][0]}: {host[i][1]}")

        # Add combined kde line
        ax.plot(x, pdf, f"-{colors[i]}")
        # Add individual lines for low and high density
        ax.plot(x, pdf_individual, f"--{colors[i]}")

    plt.title("Best-fit Mixture")
    ax.set_xlabel("Phase space density")
    ax.set_ylabel("Probability density function")
    plt.legend()
    fig.set_size_inches(50, 8)
    if save_graph:
        plt.savefig(f"{fig_dir}/{host[0][0].rsplit('_', 1)[1]}", dpi=100, bbox_inches="tight", pad_inches=0.1)
    if show_graph:
        plt.show()
    fig.clf()
    plt.close()
