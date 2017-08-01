def set_params(n_gals):
    """
    The parameters of all these cases will be the same (and vanilla), except for the number of galaxies being different
    """
    params = {}

    params['smooth_truth'] = 0
    params['interim_prior'] = 'flat'
    params['n_galaxies'] = n_gals
    params['n_bins'] = 25
    params['bin_min'] = 0.
    params['bin_max'] = 1.
    params['variable_sigmas'] = 0
    params['constant_sigma'] = 0.03
    params['catastrophic_outliers'] = '0'
    params['gr_threshold'] = 1.1
    params['n_accepted'] = 10000
    params['n_burned'] = 1000
    params['n_walkers'] = 100

    return params

def make_true(given_key):
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
    test_info = all_tests[given_key]

    if test_info['params']['smooth_truth'] == 1:
        true_amps = np.array([0.150,0.822,1.837,2.815,3.909,
                              5.116,6.065,6.477,6.834,7.304,
                              7.068,6.771,6.587,6.089,5.165,
                              4.729,4.228,3.664,3.078,2.604,
                              2.130,1.683,1.348,0.977,0.703,
                              0.521,0.339,0.283,0.187,0.141,
                              0.104,0.081,0.055,0.043,0.034])
        true_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], len(true_amps) + 1)
        true_grid_mids = (true_grid[1:] + true_grid[:-1]) / 2.
        f = spi.interp1d(true_grid_mids, true_amps)
        bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
        bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]
        true_amps = f(bin_mids)
        true_means = bin_mids
        true_sigmas = bin_difs
    else:
        bin_range = max(test_info['bin_ends']) - min(test_info['bin_ends'])
        true_amps = np.array([0.20, 0.35, 0.55])
        true_means = np.array([0.5, 0.2, 0.75]) * bin_range + min(test_info['bin_ends'])
        true_sigmas = np.array([0.4, 0.2, 0.1]) * bin_range

    n_mix_comps = len(true_amps)
    true_funcs = []
    for c in range(n_mix_comps):
        true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))

    true_dict = {'amps': true_amps, 'means': true_means, 'sigmas': true_sigmas}
    true_dict['bins'] = test_info['bin_ends']

    # true_zs = true_nz.sample(test_info['params']['n_galaxies'])
    # true_dict['zs'] = true_zs
    test_info['truth'] = true_dict

    return(test_info)

def make_interim_prior(given_key):
    """
    Function to make the histogram-parametrized interim prior

    Parameters
    ----------
    given_key: string
        name of test case for which interim prior is to be made

    Returns
    -------
    interim_prior: chippr.discrete or chippr.gauss or chippr.gmix object
        the discrete distribution that will be the interim prior
    """
    test_info = all_tests[given_key]

    if test_info['params']['interim_prior'] == 'template':
        bin_range = max(test_info['bin_ends']) - min(test_info['bin_ends'])
        int_amps = np.array([0.35, 0.5, 0.15])
        int_means = np.array([0.1, 0.5, 0.9]) * bin_range + min(test_info['bin_ends'])
        int_sigmas = np.array([0.1, 0.1, 0.1]) * bin_range
        n_mix_comps = len(int_amps)
        int_funcs = []
        for c in range(n_mix_comps):
            int_funcs.append(chippr.gauss(int_means[c], int_sigmas[c]**2))
        interim_prior = chippr.gmix(int_amps, int_funcs,
            limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))
    elif test_info['params']['interim_prior'] == 'training':
        int_amps = np.array([0.150,0.822,1.837,2.815,3.909,
                              5.116,6.065,6.477,6.834,7.304,
                              7.068,6.771,6.587,6.089,5.165,
                              4.729,4.228,3.664,3.078,2.604,
                              2.130,1.683,1.348,0.977,0.703,
                              0.521,0.339,0.283,0.187,0.141,
                              0.104,0.081,0.055,0.043,0.034])
        int_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], len(int_amps) + 1)
        int_grid_mids = (int_grid[1:] + int_grid[:-1]) / 2.
        f = spi.interp1d(int_grid_mids, int_amps)
        bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
        bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]
        int_amps = f(bin_mids)
        int_means =bin_mids
        int_sigmas = bin_difs
        n_mix_comps = len(int_amps)
        int_funcs = []
        for c in range(n_mix_comps):
            int_funcs.append(chippr.gauss(int_means[c], int_sigmas[c]**2))
        interim_prior = chippr.gmix(int_amps, int_funcs,
                limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))
    else:
        bin_ends = np.array([test_info['params']['bin_min'], test_info['params']['bin_max']])
        weights = np.array([1.])
        interim_prior = chippr.discrete(bin_ends, weights)

    return(interim_prior)

def make_catalog(given_key):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = all_tests[given_key]
    test_name = test_info['name']

    test_dir = os.path.join(result_dir, test_name)
    test_info['dir'] = test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    param_file_name = test_name + '.txt'
    params = chippr.utils.ingest(param_file_name)
    params = defaults.check_sim_params(params)
    params = check_extra_params(params)
    test_info['params'] = params

    test_info['bin_ends'] = np.linspace(test_info['params']['bin_min'],
                                test_info['params']['bin_max'],
                                test_info['params']['n_bins'] + 1)

    test_info = make_true(given_key)
    true_amps = test_info['truth']['amps']
    true_means = test_info['truth']['means']
    true_sigmas =  test_info['truth']['sigmas']
    n_mix_comps = len(true_amps)
    true_funcs = []
    for c in range(n_mix_comps):
        true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(test_info['params']['bin_min'], test_info['params']['bin_max']))

    interim_prior = make_interim_prior(given_key)

    posteriors = chippr.catalog(param_file_name, loc=test_dir)
    output = posteriors.create(true_nz, interim_prior, N=test_info['params']['n_gals'])
    # data = np.exp(output['log_interim_posteriors'])
    posteriors.write()
    data_dir = posteriors.data_dir
    with open(os.path.join(data_dir, 'true_params.p'), 'w') as true_file:
        pickle.dump(test_info['truth'], true_file)

def check_prob_params(params):
    """
    Sets parameter values pertaining to components of probability

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for probability

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for probability
    """
    if 'prior_mean' not in params:
        params['prior_mean'] = 'interim'
    else:
        params['prior_mean'] = params['prior_mean'][0]
    if 'no_prior' not in params:
        params['no_prior'] = 0
    else:
        params['no_prior'] = int(params['no_prior'][0])
    if 'no_data' not in params:
        params['no_data'] = 0
    else:
        params['no_data'] = int(params['no_data'][0])
    return params

def set_up_prior(data, params):
    """
    Function to create prior distribution from data

    Parameters
    ----------
    data: dict
        catalog dictionary containing bin endpoints, log interim prior, and log
        interim posteriors
    params: dict
        dictionary of parameter values for creation of prior

    Returns
    -------
    prior: chippr.mvn object
        prior distribution as multivariate normal
    """
    zs = data['bin_ends']
    log_nz_intp = data['log_interim_prior']
    log_z_posts = data['log_interim_posteriors']

    z_difs = zs[1:]-zs[:-1]
    z_mids = (zs[1:]+zs[:-1])/2.
    n_bins = len(z_mids)

    n_pdfs = len(log_z_posts)

    a = 1.# / n_bins
    b = 20.#1. / z_difs ** 2
    c = a / n_pdfs
    prior_var = np.eye(n_bins)
    for k in range(n_bins):
        prior_var[k] = a * np.exp(-0.5 * b * (z_mids[k] - z_mids) ** 2)
    prior_var += c * np.identity(n_bins)

    prior_mean = log_nz_intp
    prior = mvn(prior_mean, prior_var)
    if params['prior_mean'] == 'sample':
        new_mean = prior.sample_one()
        prior = mvn(new_mean, prior_var)
        print(params['prior_mean'], prior_mean, new_mean)
    else:
        print(params['prior_mean'], prior_mean)

    return (prior, prior_var)

def do_inference(given_key):
    """
    Function to do inference from a catalog of photo-z interim posteriors

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = all_tests[given_key]
    test_name = test_info['name']

    test_name = test_name[:-1]
    param_file_name = test_name + '.txt'

    params = chippr.utils.ingest(param_file_name)
    params = check_prob_params(params)
    params = defaults.check_inf_params(params)
    print(params)

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)
    zs = data['bin_ends']
    z_difs = zs[1:]-zs[:-1]
    with open(os.path.join(os.path.join(test_dir, saved_location), 'true_params.p'), 'r') as true_file:
        true_nz_params = pickle.load(true_file)
    true_amps = true_nz_params['amps']
    true_means = true_nz_params['means']
    true_sigmas =  true_nz_params['sigmas']
    n_mix_comps = len(true_amps)
    true_funcs = []
    for c in range(n_mix_comps):
        true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(min(zs), max(zs)))

    (prior, cov) = set_up_prior(data, params)

    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, vb=True)

    nz_stacked = nz.calculate_stacked()
    print('stacked: '+str(np.dot(np.exp(nz_stacked), z_difs)))
    nz_mmap = nz.calculate_mmap()
    print('MMAP: '+str(np.dot(np.exp(nz_mmap), z_difs)))
    nz_mexp = nz.calculate_mexp()
    print('MExp: '+str(np.dot(np.exp(nz_mexp), z_difs)))
    nz_mmle = nz.calculate_mmle(nz_stacked, no_data=params['no_data'], no_prior=params['no_prior'])
    print('MMLE: '+str(np.dot(np.exp(nz_mmle), z_difs)))

    nz_stats = nz.compare()
    nz.plot_estimators()
    nz.write('nz.p')

    # n_bins = len(nz_mmle)
    if params['n_walkers'] is not None:
        n_ivals = params['n_walkers']
    else:
        n_ivals = 10 * n_bins
    initial_values = prior.sample(n_ivals)
    log_z_dens_plots.plot_ivals(initial_values, nz.info, nz.plot_dir)
    # nz_samps = nz.calculate_samples(initial_values, no_data=params['no_data'], no_prior=params['no_prior'])

    nz_stats = nz.compare()

    nz.plot_estimators()
    nz.write('nz.p')

def one_scale(n_gals):
    """
    Combines the catalog generation and inference steps for parallelization to establish scaling behavior
    """

    set_params(n_gals)
    make_catalog(n_gals)
    do_inference(n_gals)
    print('FINISHED ' + str(n_gals))
    return

if __name__ == "__main__":

    import numpy as np
    import pickle
    import os
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results/scaling')
    catalog_sizes = [100, 1000, 10000, 100000]
    test_name = 'scaling'

    # all_tests = {}
    # for n_gals in catalog_sizes:
    #     test_info = {}
    #     test_info['size'] = n_gals
    #     all_tests[str(n_gals)] = test_info

    nps = mp.cpu_count()
    pool = mp.Pool(nps)
    pool.map(one_scale, catalog_sizes)
