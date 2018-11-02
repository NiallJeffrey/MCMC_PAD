
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


def plot_normalised_2d(log_posterior_2d, n_grid):
    """
    This plotting function keeps the notebook clear of messy code
    :param log_posterior_2d:
    :param n_grid:
    """
    posterior_2d = np.reshape(np.exp(log_posterior_2d) / np.sum(np.exp(log_posterior_2d)), (n_grid, n_grid)).T
    _ = plt.figure(figsize=(6, 6))
    _ = plt.imshow(posterior_2d, origin='lower')
    _ = plt.xlabel(r'$\theta_0$', fontsize=17), plt.ylabel(r'$\theta_1$', fontsize=17)
    _ = plt.suptitle(r'$P(\theta_0\ |\ \rm{other\ stuff})$', fontsize=17)
    _ = plt.title('(The axis labels are ticks not parameter values)', fontsize = 17)
    _ = plt.show()


def plot_chainconsumer_2d(theta_0, theta_1, log_posterior):
    """
    Plotting function of 2d contours on a grid with ChainConsumer
    :param theta_0:
    :param theta_1:
    :param log_posterior:
    """
    c = ChainConsumer()

    weights_array1 = np.reshape(np.exp(log_posterior), -1)
    weights_array1 += np.min(weights_array1[np.where(weights_array1 > 0)])

    c.add_chain([theta_0, theta_1],
                parameters=[r'$\theta_0$', r'$\theta_1$'],
                weights=weights_array1, grid=True)

    c.configure(kde=[1.], sigmas=[1, 2, 3],
                contour_label_font_size=19,
                label_font_size=15, shade=True)

    fig = c.plotter.plot(figsize=(6, 6))


def initial_parameters(theta, relative_sigma):
    """
    This is not randomise the initial position of the
    :param theta: list/array of parameter values
    :param relative_sigma: controls variance of random draws
    :return: the theta array but with random shifts
    """
    theta = np.array(theta)
    return np.random.normal(theta, np.abs(theta * relative_sigma))


def posterior6d_hard_prior(theta, prior_min, prior_max):
    """
    This just adds a hard prior to the previous function
    :param theta: proposed parameter array
    :param prior_min: minimum value for any parameter
    :param prior_max: maximum value for any parameter
    :return: log_posterior
    """

    if any(val > prior_max for val in theta):
        return 1e18 * (log_function_nd(theta))
    elif any(val < prior_min for val in theta):
        return 1e18 * (log_function_nd(theta))
    else:
        return log_function_nd(theta)


def break_condition(i, max, total_length, time_val):
    """
    This returns True if i >= max, and prints the percentage of it wrt total_length
    :param i:
    :param max:
    :param total_length:
    """

    if i >= max:
        print('time to do ' + str(float(i) * 100./float(total_length)) +\
              '% is ' + str(time.time() - time_val))
        return True