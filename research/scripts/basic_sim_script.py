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

def make_true_zs(true_nz):
    """
    Function to create true redshifts to be shared among several test cases

    Returns
    -------
    true_zs: numpy.ndarray, float
        array of true redshift values
    """
    N =10**4
    true_zs = true_nz.sample(N)
    return(true_zs)

def make_catalog(given_key):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    with open(name_file) as tests_to_run:

        for test_name in tests_to_run:
            test_info = all_tests[given_key]
            test_name = test_info['name']
            true_nz = test_info['true_nz']
            true_zs = test_info['true_zs']

            test_name = test_name[:-1]
            param_file_name = test_name + '.txt'

            params = chippr.sim_utils.ingest(param_file_name)
            params = defaults.check_sim_params(params)

            bin_ends = np.array([params['bin_min'], params['bin_max']])
            weights = np.array([1.])

            interim_prior = chippr.discrete(bin_ends, weights)

            test_dir = os.path.join(result_dir, test_name)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir)

            posteriors = chippr.catalog(param_file_name, loc=test_dir)
            output = posteriors.create(true_zs, interim_prior)

            data = np.exp(output['log_interim_posteriors'])

            posteriors.write()

if __name__ == "__main__":

    import numpy as np
    import os
    import shutil
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    name_file = 'which_sim_tests.txt'

    with open(name_file) as tests_to_run:
        all_tests = {}
        for test_name in tests_to_run:
            true_nz = make_true_nz(test_name)
            true_zs = make_true_zs(true_nz)
            test_info = {}
            test_info['name'] = test_name
            test_info['true_nz'] = true_nz
            test_info['true_zs'] = true_zs
            all_tests[test_name] = test_info

    nps = mp.cpu_count()-1
    pool = mp.Pool(nps)
    pool.map(make_catalog, all_tests.keys())
