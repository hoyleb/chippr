def make_true_nz(test_name):
    """
    Function to create true redshift distribution to be shared among several test cases

    Parameters
    ----------
    test_name: string
        name used to look up parameters for making true_nz

    Returns
    -------
    true_nz: chippr.gmix object
        gaussian mixture probability distribution

    Notes
    -----
    test_name is currently ignored but will soon be used to load parameters for making true_nz instead of hardcoded values.
    """
    true_amps = np.array([0.20, 0.35, 0.55])
    true_means = np.array([0.5, 0.2, 0.75])
    true_sigmas = np.array([0.4, 0.2, 0.1])

    true_nz = chippr.gmix(true_amps, true_means, true_sigmas, limits=(0., 1.))

    return(true_nz)

def set_up_prior(data):
    """
    Function to create prior distribution from data

    Parameters
    ----------
    data: dict
        catalog dictionary containing bin endpoints, log interim prior, and log interim posteriors

    Returns
    -------
    prior: chippr.mvn object
        prior distribution as multivariate normal
    """
    zs = data['bin_ends']
    log_nz_intp = np.exp(data['log_interim_prior'])
    log_z_posts = np.exp(data['log_interim_posteriors'])

    z_difs = zs[1:]-zs[:-1]
    z_mids = (zs[1:]+zs[:-1])/2.
    n_bins = len(z_mids)

    prior_var = np.eye(n_bins)
    for k in range(n_bins):
        prior_var[k] = 1. * np.exp(-0.5 * (z_mids[k] - z_mids) ** 2 / 0.05 ** 2)
    prior_mean = log_nz_intp
    prior = mvn(prior_mean, prior_var)
    return prior

def just_plot(given_key):
    """
    Function to make plot from a previously generate log_z_dens object

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = all_tests[given_key]
    test_name = test_info['name']
    true_nz = test_info['truth']

    test_name = test_name[:-1]
    param_file_name = test_name + '.txt'

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)

    prior = set_up_prior(data)
    n_bins = len(data['log_interim_prior'])
    n_ivals = 2 * n_bins
    initial_values = prior.sample(n_ivals)

    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, params=param_file_name, vb=True)

    nz.info = nz.read('nz.p')
    nz.plot_estimators()

    nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import os
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    name_file = 'which_inf_tests.txt'

    with open(name_file) as tests_to_run:
        all_tests = {}
        for test_name in tests_to_run:
            true_nz = make_true_nz(test_name)
            test_info = {}
            test_info['name'] = test_name
            test_info['truth'] = true_nz
            all_tests[test_name] = test_info

    nps = mp.cpu_count()-1
    pool = mp.Pool(nps)
    pool.map(just_plot, all_tests.keys())
