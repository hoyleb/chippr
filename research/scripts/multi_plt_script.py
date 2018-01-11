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

def make_true(true_nz_info):
    """
    Function to create true redshift distribution to be shared among several
    test cases

    Parameters
    ----------
    given_key: string
        name of test case for which true n(z) is to be made

    Returns
    -------
    true_nz: chippr.gmix object
        gaussian mixture probability distribution

    Notes
    -----
    test_name is currently ignored but will soon be used to load parameters for
    making true_nz instead of hardcoded values.
    """
    # test_info = all_tests[given_key]

    if true_nz_info['format'] == 'discrete':
        # z0 = 0.3
        # def smooth_func(z):
        #     return 1/(2 * z_0) * (z/z_0)**2 * np.exp(-z/z_0)
        # true_amps = np.array([0.150,0.822,1.837,2.815,3.909,
        #                       5.116,6.065,6.477,6.834,7.304,
        #                       7.068,6.771,6.587,6.089,5.165,
        #                       4.729,4.228,3.664,3.078,2.604,
        #                       2.130,1.683,1.348,0.977,0.703,
        #                       0.521,0.339,0.283,0.187,0.141,
        #                       0.104,0.081,0.055,0.043,0.034])
        # true_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], len(test_info['bin_ends']))
        # true_grid_mids = (true_grid[1:] + true_grid[:-1]) / 2.
        # f = spi.interp1d(true_grid_mids, true_amps)
        # bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
        # bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]
        # fine_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], 10 * (len(test_info['bin_ends'] - 1) + 1))
        # true_means = bin_mids
        # true_amps = smooth_func(fine_grid)# f(bin_mids)
        # true_sigmas = bin_difs

        true_funcs = [chippr.discrete(true_nz_info['grid'], true_nz_info['amps'])]
        true_amps = [1.]

        # true_dict = {'format', 'discrete', 'bins': fine_grid, 'amps': true_amps}
    elif true_nz_info['format'] == 'gauss':
        # bin_range = max(test_info['bin_ends']) - min(test_info['bin_ends'])
        # true_amps = np.array([0.20, 0.35, 0.55])
        # true_means = np.array([0.5, 0.2, 0.75]) * bin_range + min(test_info['bin_ends'])
        # true_sigmas = np.array([0.4, 0.2, 0.1]) * bin_range
        #
        # n_mix_comps = len(true_amps)
        # true_funcs = []
        # for c in range(n_mix_comps):
        #     true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
        # true_nz = chippr.gmix(true_amps, true_funcs,
        #     limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))
        #
        # true_dict = {'format': 'gmix', 'amps': true_amps, 'means': true_means, 'sigmas': true_sigmas}
        n_mix_comps = len(true_nz_info['amps'])
        true_funcs = []
        for c in range(n_mix_comps):
            true_funcs.append(chippr.gauss(true_nz_info['means'][c], true_nz_info['sigmas'][c]**2))
        true_amps = true_nz_info['amps']
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(min(true_nz_info['bins']), max(true_nz_info['bins'])))

    # true_dict['bins'] = test_info['bins']

    # true_zs = true_nz.sample(test_info['params']['n_galaxies'])
    # true_dict['zs'] = true_zs
    # test_info['truth'] = true_dict

    return(test_info, true_nz)

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

    test_name = test_name[:-1]
    param_file_name = test_name + '.txt'

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)

    with open(os.path.join(os.path.join(test_dir, saved_location), 'true_params.p'), 'r') as true_file:
        true_nz_params = pickle.load(true_file)
    # true_amps = true_nz_params['amps']
    # true_means = true_nz_params['means']
    # true_sigmas =  true_nz_params['sigmas']
    # n_mix_comps = len(true_amps)
    # true_funcs = []
    # for c in range(n_mix_comps):
    #     true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    # true_nz = chippr.gmix(true_amps, true_funcs,
    #         limits=(simulated_posteriors.params['bin_min'], simulated_posteriors.params['bin_max']))
    true_info, true_nz = make_true(true_nz_params)

    true_data = os.path.join(test_dir, saved_location)
    with open(os.path.join(true_data, 'true_vals.txt'), 'rb') as csvfile:
        tuples = (line.split(None) for line in csvfile)
        alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
    true_vals = np.array(alldata)
    bin_mids = (data['bin_ends'][1:] + data['bin_ends'][:-1]) / 2.
    catalog_plots.plot_true_histogram(true_vals.T[0], plot_loc=os.path.join(test_dir, 'plots'), true_func=true_nz)
    catalog_plots.plot_obs_scatter(true_vals.T, np.exp(data['log_interim_posteriors']), bin_mids, plot_loc=os.path.join(test_dir, 'plots'))

    prior = set_up_prior(data)
    n_bins = len(data['log_interim_prior'])
    n_ivals = 2 * n_bins
    # initial_values = prior.sample(n_ivals)

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
    name_file = 'which_plt_tests.txt'

    with open(name_file) as tests_to_run:
        all_tests = {}
        for test_name in tests_to_run:
            test_info = {}
            test_info['name'] = test_name
            all_tests[test_name] = test_info

    nps = mp.cpu_count()
    pool = mp.Pool(nps)
    pool.map(just_plot, all_tests.keys())
