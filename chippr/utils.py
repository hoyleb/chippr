# Module containing generally handy functions used by simulation and inference modules

import numpy as np

import chippr
from chippr import defaults as d

# Choose random seed at import, hopefully carries through everywhere
np.random.seed(d.seed)

def safe_log(arr, threshold=d.eps):
    """
    Takes the natural logarithm of an array that might contain zeros.

    Parameters
    ----------
    arr: ndarray, float
        array of values to be logged
    threshold: float, optional
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: ndarray
        logged values, with small value replacing un-loggable values
    """
    arr = np.asarray(arr)
    # if type(arr) == np.ndarray:
    arr[arr < threshold] = threshold
    # else:
    #     arr = max(threshold, arr)
    logged = np.log(arr)
    return logged

def ingest(in_info):
    """
    Function reading in parameter file to define functions necessary for
    generation of posterior probability distributions

    Parameters
    ----------
    in_info: string or dict
        string containing path to plaintext input file or dict containing
        likelihood input parameters

    Returns
    -------
    in_dict: dict
        dict containing keys and values necessary for posterior probability
        distributions
    """
    if type(in_info) == str:
        with open(in_info) as infile:
            lines = (line.split(None) for line in infile)
            in_dict = {defn[0] : defn[1:] for defn in lines}
    else:
        in_dict = in_info
    return in_dict
