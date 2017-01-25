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

def the_loop(true_nz=None):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    true_nz: chippr.gmix object, optional
        gaussian mixture probability distribution
    """
    result_dir = os.path.join('..', 'results')
    name_file = 'which_inf_tests.txt'

    with open(name_file) as tests_to_run:

        for test_name in tests_to_run:

            test_name = test_name[:-1]
            param_file_name = test_name + '.txt'

            test_dir = os.path.join(result_dir, test_name)
            simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
            saved_location = 'data'
            saved_type = '.txt'
            data = simulated_posteriors.read(loc=saved_location, style=saved_type)

            zs = data['bin_ends']
            nz_intp = np.exp(data['log_interim_prior'])
            z_posts = np.exp(data['log_interim_posteriors'])

            z_difs = zs[1:]-zs[:-1]
            z_mids = (zs[1:]+zs[:-1])/2.
            n_bins = len(z_mids)

            prior_var = np.eye(n_bins)
            for k in range(n_bins):
                prior_var[k] = 1. * np.exp(-0.5 * (z_mids[k] - z_mids) ** 2 / 0.05 ** 2)
            prior_mean = nz_intp
            prior = mvn(prior_mean, prior_var)

            nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, vb=True)

            nz_stacked = nz.calculate_stacked()
            nz_mmap = nz.calculate_mmap()
            nz_mexp = nz.calculate_mexp()
            nz_stats = nz.compare()

            nz.info['estimators'].keys()
            nz.plot_estimators('final_plot.png')
            
            nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import os

    import chippr
    from chippr import *

    true_nz = make_true()

    the_loop(true_nz=true_nz)
