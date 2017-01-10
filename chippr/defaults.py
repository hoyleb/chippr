# Module containing hardcoded default values in central location for easy modification later

import sys

global seed
seed = 42

global eps
eps = sys.float_info.epsilon

global min_x, max_x
min_x = 0.
max_x = 1.

global n_bins
n_bins = 10

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
    params = check_variable_sigmas(params)
    params = check_catastrophic_outliers(params)
    return params

def check_variable_sigmas(params):
    if 'variable_sigma' not in params:
        params['variable_sigma'] = 0
    else:
        params['variable_sigma'] = bool(params['variable_sigmas'][0])
    if not params['variable_sigma']:
        if 'constant_sigma' not in params:
            params['constant_sigma'] = 0.05
        else:
            params['constant_sigma'] = float(params['constant_sigma'][0])
    return params

def check_catastrophic_outliers(params):
    if 'catastrophic_outliers' not in params:
        params['catastrophic_outliers'] = 0
    else:
        params['catastrophic_outliers']  = bool(params['catastrophic_outliers'][0])
    if params['catastrophic_outliers']:
        params['outlier_fraction'] = float(params['outlier_fraction'][0])
        params['outlier_mean'] = float(params['outlier_mean'][0])
        params['outlier_sigma'] = float(params['outlier_sigma'][0])
    else:
        params['outlier_fraction'] = 0.
    return params
