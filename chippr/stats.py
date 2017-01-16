import numpy as np

import chippr
from chippr import defaults as d
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

def gr_test(sample, threshold=d.gr_threshold):
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
    gr = multi_parameter_gr_stat(sample)
    return np.max(gr) > threshold

def cft(xtimes, lag):
    """
    Helper function to calculate autocorrelation time for chain of MCMC samples

    Parameters
    ----------
    xtimes: numpy.ndarray, float
        single parameter values for a single walker over all iterations
    lag: int
        maximum lag time in number of iterations

    Returns
    -------
    ans: numpy.ndarray, float
        autocorrelation time for one time lag for one parameter of one walker
    """
    lent = len(xtimes) - lag
    allt = xrange(lent)
    ans = np.array([xtimes[t+lag] * xtimes[t] for t in allt])
    return ans

def cf(xtimes):#xtimes has ntimes elements
    """
    Helper function to calculate autocorrelation time for chain of MCMC samples

    Parameters
    ----------
    xtimes: numpy.ndarray, float
        single parameter values for a single walker over all iterations

    Returns
    -------
    cf: numpy.ndarray, float
        autocorrelation time over all time lags for one parameter of one walker
    """
    cf0 = np.dot(xtimes, xtimes)
    allt = xrange(len(xtimes) / 2)
    cf = np.array([sum(cft(xtimes,lag)[len(xtimes) / 2:]) for lag in allt]) / cf0
    return cf

def cfs(x, mode):#xbinstimes has nbins by ntimes elements
    """
    Helper function for calculating autocorrelation time for MCMC chains

    Parameters
    ----------
    x: numpy.ndarray, float
        input parameter values of length number of iterations by number of walkers if mode='walkers' or dimension of parameters if mode='bins'
    mode: string
        'bins' for one autocorrelation time per parameter, 'walkers' for one autocorrelation time per walker

    Returns
    -------
    cfs: numpy.ndarray, float
        autocorrelation times for all walkers if mode='walkers' or all parameters if mode='bins'
    """
    if mode == 'walkers':
        xbinstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xbinstimes]) / len(xbinstimes)
    if mode == 'bins':
        xwalkerstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xwalkerstimes]) / len(xwalkerstimes)
    return cfs

def acors(xtimeswalkersbins, mode='bins'):
    """
    Calculates autocorrelation time for MCMC chains

    Parameters
    ----------
    xtimeswalkersbins: numpy.ndarray, float
        emcee chain values of dimensions (n_iterations, n_walkers, n_parameters)
    mode: string, optional
        'bins' for one autocorrelation time per parameter, 'walkers' for one autocorrelation time per walker

    Returns
    -------
    taus: numpy.ndarray, float
        autocorrelation times by bin or by walker depending on mode
    """
    if mode == 'walkers':
        xwalkersbinstimes = np.swapaxes(xtimeswalkersbins, 1, 2)#nwalkers by nbins by nsteps
        taus = np.array([1. + 2. * sum(cfs(xbinstimes, mode)) for xbinstimes in xwalkersbinstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    if mode == 'bins':
        xbinswalkerstimes = xtimeswalkersbins.T#nbins by nwalkers by nsteps
        taus = np.array([1. + 2. * sum(cfs(xwalkerstimes, mode)) for xwalkerstimes in xbinswalkerstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    return taus
