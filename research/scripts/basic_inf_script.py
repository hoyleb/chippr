def make_true():
    """
    Function to create true redshift distribution to be shared among several test cases

    Returns
    -------
    true_nz: chippr.gmix object
        gaussian mixture probability distribution
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

def the_loop(given_key):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    true_nz: chippr.gmix object, optional
        gaussian mixture probability distribution
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

    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, vb=True)

    nz_stacked = nz.calculate_stacked()
    nz_mmap = nz.calculate_mmap()
    nz_mexp = nz.calculate_mexp()
    nz_stats = nz.compare()

    nz.plot_estimators()

    nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import os
    import multiprocessing as mp

    import chippr
    from chippr import *

    true_nz = make_true()

    result_dir = os.path.join('..', 'results')
    name_file = 'which_inf_tests.txt'

    with open(name_file) as tests_to_run:
        all_tests = {}
        for test_name in tests_to_run:
            test_info = {}
            test_info['name'] = test_name
            test_info['truth'] = true_nz
            all_tests[test_name] = test_info

    nps = mp.cpu_count()-1
    pool = mp.Pool(nps)
    pool.map(the_loop, all_tests.keys())
