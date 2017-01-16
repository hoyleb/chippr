import numpy as np

import chippr
from chippr import utils as u

def mean(population):
    """
    Calculates the mean of a population

    Parameters
    ----------
    population: np.array, float
        population over which to calculate the mean

    Returns
    -------
    mean: np.array, float
        mean value over population
    """
    shape = np.shape(population)
    flat = population.reshape(np.prod(shape[:-1]), shape[-1])
    mean = np.mean(flat, axis=0)
    return mean

def single_parameter_gr_stat(chain):
    """
    Calculates the Gelman-Rubin test statistic of convergence of an MCMC chain over one parameter

    Parameters
    ----------
    chain: numpy.ndarray, float
        single-parameter chain

    Returns
    -------
    R_hat: float
        potential scale reduction factor
    """
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    xb = np.mean(chain, axis=1)
    xbb = np.mean(xb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1.) * np.sum((xbb - xb)**2., axis=0)
    var_x = (n - 1.) / n * W + 1. / n * B
    R_hat = np.sqrt(var_x / W)
    return R_hat

def multi_parameter_gr_stat(sample):
    """
    Calculates the Gelman-Rubin test statistic of convergence of an MCMC chain over multiple parameters

    Parameters
    ----------
    sample: numpy.ndarray, float
        multi-parameter chain output

    Returns
    -------
    Rs: numpy.ndarray, float
        vector of the potential scale reduction factors
    """
    dims = np.shape(sample)
    (n_walkers, n_iterations, n_params) = dims
    n_burn_ins = n_iterations / 2
    chain_ensemble = sample.reshape(n_iterations, n_walkers, n_params)
    chain_ensemble = chain_ensemble[n_burn_ins:, :]
    Rs = np.zeros((n_params))
    for i in range(n_params):
        chains = chain_ensemble[:, :, i].T
        Rs[i] = single_parameter_gr_stat(chains)
    return Rs

def gr_test(sample, threshold=u.gr_threshold):
    """
    Performs the Gelman-Rubin test of convergence of an MCMC chain

    Parameters
    ----------
    sample: numpy.ndarray, float
        chain output
    threshold: float, optional
        Gelman-Rubin test statistic criterion (usually around 1)

    Returns
    -------
    test_result: boolean
        True if burning in, False if post-burn in
    """
    gr = multi_parameter_gr_test(sample)
    return np.max(gr) > threshold
