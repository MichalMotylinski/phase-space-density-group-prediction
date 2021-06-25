import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math


def best_fit_mixture(model, data, host):
    """
    Draw best fit mixture.

    :param model: Gaussian Mixture model object
    :param data: Calculated densities
    :param host: Target star
    :return: Plot a best fit mixture graph
    """
    fig = plt.figure(figsize=(50, 5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)
    ax = fig.add_subplot(131)

    x = np.linspace(math.floor(data.min()), math.ceil(data.max()), data.shape[0])
    logprob = model.score_samples(x.reshape(-1, 1))

    responsibilities = model.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.axvline(x=host[1], color="r", label=f"{host[0]}: {host[1]}")

    ax.hist(data, 30, density=True, histtype='stepfilled', alpha=0.5)

    # Add combined kde line
    ax.plot(x, pdf, '-k')
    # Add individual lines for low and high density
    ax.plot(x, pdf_individual, '--k')
    #ax.text(0.04, 0.96, "Best-fit Mixture", ha='left', va='top', transform=ax.transAxes)
    plt.title("Best-fit Mixture")
    ax.set_xlabel('Phase space density')
    ax.set_ylabel('Probability density function')
    plt.legend()
    plt.show()
