def set_shared_params():
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
    # params = {}
    #
    # params['n_bins'] = 25
    # params['bin_min'] = 0.
    # params['bin_max'] = 1.
    #
    # params['smooth_truth'] = 0
    # params['interim_prior'] = 'flat'
    # params['variable_sigmas'] = 0
    # params['constant_sigma'] = 0.03
    # params['catastrophic_outliers'] = '0'
    # params['outlier_fraction'] = 0.
    #
    # params['gr_threshold'] = 1.1
    # params['n_accepted'] = 10000
    # params['n_burned'] = 1000
    # params['n_walkers'] = 100
    paramfile = 'scaling.txt'
    params = chippr.utils.ingest(paramfile)
    params = chippr.defaults.check_sim_params(params)
    params = chippr.defaults.check_inf_params(params)

    bin_range = params['bin_max'] - params['bin_min']
    true_amps = np.array([0.20, 0.35, 0.55])
    true_means = np.array([0.5, 0.2, 0.75]) * bin_range + params['bin_min']
    true_sigmas = np.array([0.4, 0.2, 0.1]) * bin_range

    # n_mix_comps = len(true_amps)
    # true_funcs = []
    # for c in range(n_mix_comps):
    #     true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    # true_nz = chippr.gmix(true_amps, true_funcs,
    #         limits=(params['bin_min'], params['bin_max']))

    true_dict = {'amps': true_amps, 'means': true_means, 'sigmas': true_sigmas}
    params['truth'] = true_dict

    bin_ends = np.array([params['bin_min'], params['bin_max']])
    weights = np.array([1.])
    # interim_prior = chippr.discrete(bin_ends, weights)
    interim_dict = {'bin_ends': bin_ends, 'weights': weights}
    params['interim'] = interim_dict

    params['prior_mean'] = 'interim'
    params['no_prior'] = 0
    params['no_data'] = 0

    return(params)

def set_unique_params(n_gals):
    """
    The parameters of all these cases will be the same (and vanilla), except for the number of galaxies being different
    """
    params = start_params
    params['n_gals'] = n_gals
    params['name'] = str(10**n_gals)

    test_dir = os.path.join(result_dir, params['name'])
    params['dir'] = test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    return params

def make_catalog(params):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = {}
    test_info['params'] = params
    test_info['name'] = params['name']
    test_info['dir'] = params['dir']

    test_info['bin_ends'] = np.linspace(test_info['params']['bin_min'],
                                test_info['params']['bin_max'],
                                test_info['params']['n_bins'] + 1)

    true_amps = test_info['params']['truth']['amps']
    true_means = test_info['params']['truth']['means']
    true_sigmas =  test_info['params']['truth']['sigmas']
    n_mix_comps = len(true_amps)
    true_funcs = []
    for c in range(n_mix_comps):
        true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(test_info['params']['bin_min'], test_info['params']['bin_max']))
    test_info['truth'] = test_info['params']['truth']

    bin_ends = np.array([test_info['params']['bin_min'], test_info['params']['bin_max']])
    weights = np.array([1.])
    interim_prior = chippr.discrete(bin_ends, weights)

    posteriors = chippr.catalog(params=params, loc=test_info['dir'])
    output = posteriors.create(true_nz, interim_prior, N=test_info['params']['n_gals'])
    # data = np.exp(output['log_interim_posteriors'])

    posteriors.write()
    data_dir = posteriors.data_dir
    with open(os.path.join(data_dir, 'true_params.p'), 'w') as true_file:
        pickle.dump(test_info['truth'], true_file)

    return

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

def do_inference(params):
    """
    Function to do inference from a catalog of photo-z interim posteriors

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_name = params['name']
    param_file_name = test_name + '.txt'

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=params, loc=test_dir)
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

    n_ivals = params['n_walkers']
    initial_values = prior.sample(n_ivals)
    log_z_dens_plots.plot_ivals(initial_values, nz.info, nz.plot_dir)

    # nz_samps = nz.calculate_samples(initial_values, no_data=params['no_data'], no_prior=params['no_prior'])

    nz_stats = nz.compare()
    nz.plot_estimators()
    nz.write('nz.p')

    return

def one_scale(n_gals):
    """
    Combines the catalog generation and inference steps for parallelization to establish scaling behavior
    """
    print('STARTED ' + str(n_gals))
    params = set_unique_params(n_gals)
    print('SET PARAMS ' + str(params['name']))

    sim_profile = os.path.join(params['dir'], 'sim_profile.txt')
    sys.stdout = open(sim_profile, 'w')
    pr = cProfile.Profile()
    pr.enable()
    make_catalog(params)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    sys.stdout = sys.__stdout__
    print('MADE CATALOG ' + str(params['name']))

    inf_profile = os.path.join(params['dir'], 'inf_profile.txt')
    sys.stdout = open(inf_profile, 'w')
    pr = cProfile.Profile()
    pr.enable()
    do_inference(params)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    # with open(inf_profile, 'w') as profiler:
    #     profiler.write(str(ps.print_stats()))
    sys.stdout = sys.__stdout__
    print('FINISHED INFERENCE ' + str(params['name']))

    return

if __name__ == "__main__":

    import numpy as np
    import pickle
    import os
    import sys
    import shutil
    import cProfile
    import StringIO
    import pstats
    import psutil
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results/scaling')
    catalog_sizes = [2]#, 3, 4, 5]

    start_params = set_shared_params()
    start_params['raw'] = 0

    nps = mp.cpu_count()
    pool = mp.Pool(nps)
    pool.map(one_scale, catalog_sizes)
