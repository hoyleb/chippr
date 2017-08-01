import sys

seed = 42

eps = sys.float_info.min
log_eps = sys.float_info.min_exp

min_x = 0.
max_x = 1.

n_bins = 10

n_gals = 4

constant_sigma = 0.03

gr_threshold = 1.2

n_accepted = 3
n_burned = 2

plot_colors = 5
dpi = 250

def check_sim_params(params={}):
    """
    Checks simulation parameter dictionary for various keywords and sets to
    default values if not present

    Parameters
    ----------
    params: dict, optional
        dictionary containing initial key/value pairs for simulation of catalog

    Returns
    -------
    params: dict
        dictionary containing final key/value pairs for simulation of catalog
    """
    print('checking sim params')
    params = check_basic_setup(params)
    print('basic setup params ok')
    params = check_variable_sigmas(params)
    print('variable sigma params ok')
    params = check_catastrophic_outliers(params)
    print('catastrophic outlier params ok')
    return params

def check_basic_setup(params):
    """
    Sets parameter values pertaining to basic constants of simulation

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for simulation

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for simulation

    Notes
    -----
    Currently, the number of galaxies is really log10 the number of galaxies.  This will one day be changed to permit arbitrary numbers of galaxies!
    """
    if 'n_gals' not in params:
        params['n_gals'] = n_gals
    elif params['raw'] == 1:
        params['n_gals'] = int(params['n_gals'][0])
    if 'n_bins' not in params:
        params['n_bins'] = n_bins
    elif params['raw'] == 1:
        params['n_bins'] = int(params['n_bins'][0])
    if 'bin_min' not in params:
        params['bin_min'] = min_x
    elif params['raw'] == 1:
        params['bin_min'] = float(params['bin_min'][0])
    if 'bin_max' not in params:
        params['bin_max'] = max_x
    elif params['raw'] == 1:
        params['bin_max'] = float(params['bin_max'][0])
    return params

def check_variable_sigmas(params):
    """
    Sets parameter values pertaining to widths of Gaussian PDF components

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for simulation

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for simulation
    """
    if 'variable_sigmas' not in params:
        params['variable_sigmas'] = 0
    elif params['raw'] == 1:
        params['variable_sigmas'] = int(params['variable_sigmas'][0])
    if not params['variable_sigmas']:
        if 'constant_sigma' not in params:
            params['constant_sigma'] = constant_sigma
        elif params['raw'] == 1:
            params['constant_sigma'] = float(params['constant_sigma'][0])
    return params

def check_catastrophic_outliers(params):
    """
    Sets parameter values pertaining to presence of a catastrophic outlier
    population

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for simulation

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for simulation
    """
    if 'catastrophic_outliers' not in params:
        params['catastrophic_outliers'] = '0'
    elif params['raw'] == 1:
        params['catastrophic_outliers'] = str(params['catastrophic_outliers'][0])
    if 'outlier_fraction' not in params:
        params['outlier_fraction'] = 0.
    elif params['raw'] == 1:
        params['outlier_fraction']  = float(params['outlier_fraction'][0])
    if params['outlier_fraction'] > 0. and params['raw'] == 1:
        params['outlier_mean'] = float(params['outlier_mean'][0])
        params['outlier_sigma'] = float(params['outlier_sigma'][0])
    else:
        params['outlier_fraction'] = 0.
    return params

def check_inf_params(params={}):
    """
    Checks inference parameter dictionary for various keywords and sets to
    default values if not present

    Parameters
    ----------
    params: dict, optional
        dictionary containing initial key/value pairs for inference

    Returns
    -------
    params: dict
        dictionary containing final key/value pairs for inference
    """
    params = check_sampler_params(params)
    return params

def check_sampler_params(params):
    """
    Sets parameter values pertaining to basic constants of inference

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for inference

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for inference
    """
    if 'gr_threshold' not in params:
        params['gr_threshold'] = gr_threshold
    elif params['raw'] == 1:
        params['gr_threshold'] = float(params['gr_threshold'][0])
    if 'n_accepted' not in params:
        params['n_accepted'] = 10 ** n_accepted
    elif params['raw'] == 1:
        params['n_accepted'] = 10 ** int(params['n_accepted'][0])
    if 'n_burned' not in params:
        params['n_burned'] = 10 ** n_burned
    elif params['raw'] == 1:
        params['n_burned'] = 10 ** int(params['n_burned'][0])
    if 'n_walkers' not in params:
        params['n_walkers'] = None
    elif params['raw'] == 1:
        params['n_walkers'] = int(params['n_walkers'][0])
    return params
