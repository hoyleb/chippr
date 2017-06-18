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
    with open(os.path.join(os.path.join(test_dir, saved_location), 'true_params.p'), 'r') as true_file:
        true_nz_params = pickle.load(true_file)
    chippr.gmix(true_nz_params['true_amps'], true_nz_params['true_means'], true_nz_params['true_sigmas'],
        limits=(data.z_min, data.z_max))

    prior = set_up_prior(data)
    n_bins = len(data['log_interim_prior'])
    n_ivals = 2 * n_bins
    initial_values = prior.sample(n_ivals)

    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, vb=True)

    nz.info = nz.read('nz.p')
    nz.plot_estimators()

    nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import pickle
    import os
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    test_name = 'null_test\n'

    all_tests = {}
    test_info = {}
    test_info['name'] = test_name
    all_tests[test_name] = test_info

    just_plot(test_name)
