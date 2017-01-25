def make_true():
    """
    Function to create true redshifts to be shared among several test cases

    Returns
    -------
    true_zs: numpy.ndarray, float
        array of true redshift values
    """
    true_amps = np.array([0.20, 0.35, 0.55])
    true_means = np.array([0.5, 0.2, 0.75])
    true_sigmas = np.array([0.4, 0.2, 0.1])

    true_nz = chippr.gmix(true_amps, true_means, true_sigmas, limits=(0., 1.))

    N =10**4

    true_zs = true_nz.sample(N)

    return(true_zs)

def make_catalog(true_zs):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    true_zs: numpy.ndarray, float
        array of true redshift values
    """
    result_dir = os.path.join('..', 'results')
    name_file = 'which_tests.txt'

    with open(name_file) as tests_to_run:

        for test_name in tests_to_run:

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

    import chippr
    from chippr import *

    true_zs = make_true()

    make_catalog(true_zs)
