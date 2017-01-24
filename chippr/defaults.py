import sys

seed = 42

eps = sys.float_info.epsilon

min_x = 0.
max_x = 1.

n_bins = 10
# n_items = 4

constant_sigma = 0.05

gr_threshold = 1.1

n_accepted = 10**3
n_burned = 10**2

def check_sim_params(params={}):
    """
    Checks simulation parameter dictionary for various keywords and sets to default values if not present

    Parameters
    ----------
    params: dict, optional
        dictionary containing initial key/value pairs for simulation of catalog

    Returns
    -------
    params: dict
        dictionary containing final key/value pairs for simulation of catalog
    """
    params = check_basic_setup(params)
    params = check_variable_sigmas(params)
    params = check_catastrophic_outliers(params)
    print('this far')
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
    """
    # if 'n_items' not in params:
    #     params['n_items'] = n_items
    # else:
    #     params['n_items'] = 10 ** int(params['n_items'][0])
    if 'n_bins' not in params:
        params['n_bins'] = n_bins
    else:
        params['n_bins'] = int(params['n_bins'][0])
    if 'bin_min' not in params:
        params['bin_min'] = min_x
    else:
        params['bin_min'] = float(params['bin_min'][0])
    if 'bin_max' not in params:
        params['bin_max'] = max_x
    else:
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
    else:
        params['variable_sigmas'] = int(params['variable_sigmas'][0])
    if not params['variable_sigmas']:
        if 'constant_sigma' not in params:
            params['constant_sigma'] = constant_sigma
        else:
            params['constant_sigma'] = float(params['constant_sigma'][0])
    return params

def check_catastrophic_outliers(params):
    """
    Sets parameter values pertaining to presence of a catastrophic outlier population

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for simulation

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for simulation
    """
    if 'outlier_fraction' not in params:
        params['outlier_fraction'] = 0.
    else:
        params['outlier_fraction']  = float(params['outlier_fraction'][0])
    if params['outlier_fraction'] > 0.:
        params['outlier_mean'] = float(params['outlier_mean'][0])
        params['outlier_sigma'] = float(params['outlier_sigma'][0])
    else:
        params['outlier_fraction'] = 0.
    return params
