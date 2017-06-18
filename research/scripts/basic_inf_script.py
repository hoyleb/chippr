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
    b = 30.#1. / z_difs ** 2
    c = a / n_pdfs
    prior_var = np.eye(n_bins)
    for k in range(n_bins):
        prior_var[k] = a * np.exp(-0.5 * b * (z_mids[k] - z_mids) ** 2)
    prior_var += c * np.identity(n_bins)

    prior_mean = log_nz_intp
    prior = mvn(prior_mean, prior_var)
    if params['prior_mean'] is 'sample':
        prior_mean = prior.sample_one()
        prior = mvn(prior_mean, prior_var)

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
    print('script sees params: '+str(params))

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)
    zs = data['bin_ends']
    z_difs = zs[1:]-zs[:-1]
    with open(os.path.join(os.path.join(test_dir, saved_location), 'true_params.p'), 'r') as true_file:
        true_nz_params = pickle.load(true_file)
    true_nz = chippr.gmix(true_nz_params['amps'], true_nz_params['means'], true_nz_params['sigmas'],
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
    nz.plot_estimators()
    nz.write('nz.p')

    #start_mean = mvn(nz_mmle, cov).sample_one()
    start = prior#mvn(data['log_interim_prior'], cov)

    n_bins = len(nz_mmle)
    if params['n_walkers'] is not None:
        n_ivals = params['n_walkers']
    else:
        n_ivals = 10 * n_bins
    initial_values = start.sample(n_ivals)

    nz_samps = nz.calculate_samples(initial_values, no_data=params['no_data'], no_prior=params['no_prior'])

    nz_stats = nz.compare()

    nz.plot_estimators()
    nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import pickle
    import os

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    test_name = 'null_test\n'
    all_tests = {}
    test_info = {}
    test_info['name'] = test_name
    all_tests[test_name] = test_info

    do_inference(test_name)
