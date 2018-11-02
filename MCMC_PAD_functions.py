
import numpy as np
import matplotlib.pyplot as plt
import emcee as mc
import corner
import time
from chainconsumer import ChainConsumer
from chainconsumer import analysis


def log_function_nd(theta):
    """
    This function is an arbitrary function as a placeholder for an n-dimensional posterior (not normalised)
    Don't read too much into it - it is just an example function with a single peak (for simplicity)
    :param theta: a list/array of the parameters of the function
    :return: log value of the function
    """

    dim = np.array(theta).shape[0]
    cov_diag = np.linspace(0.1, 0.15, dim) + 0.1
    y_vec = np.linspace(0.1, 0.15, dim) ** np.linspace(0.1, 0.11, dim) + 1.0
    log_residual = np.log(np.abs(theta)) - y_vec

    return - np.inner(log_residual, log_residual / cov_diag)


def plot_normalised_1d(theta_0_1d, log_posterior_1d):
    """
    This plotting function keeps the notebook clear of messy code
    :param theta_0_1d: parameter values
    :param log_posterior_1d: log posterior values
    """

    _ = plt.fill(theta_0_1d, np.exp(log_posterior_1d) / np.sum(np.exp(log_posterior_1d)), alpha=0.4)
    _ = plt.plot(theta_0_1d, np.exp(log_posterior_1d) / np.sum(np.exp(log_posterior_1d)), marker='x', ms=15, lw=1)
    _ = plt.ylabel(r'$P(\theta_0| \rm{other\ stuff})$', fontsize=17), plt.xlabel(r'$\theta_0$', fontsize=17)
    _ = plt.show()


def plot_normalised_2d(theta_0, theta_1, log_posterior_2d, n_grid):
    """
    This plotting function keeps the notebook clear of messy code
    :param theta_0:
    :param theta_1:
    :param log_posterior_2d:
    :param n_grid:
    """

    _ = plt.figure(figsize=(6, 6))
    _ = plt.imshow(np.exp(np.reshape(log_posterior_2d, (n_grid, n_grid))).T, origin='lower')
    _ = plt.xlabel(r'$\theta_0$', fontsize=17), plt.ylabel(r'$\theta_1$', fontsize=17)
    _ = plt.title(r'$P(\theta_0\ |\ \rm{other\ stuff})$', fontsize=17)
    _ = plt.show()

